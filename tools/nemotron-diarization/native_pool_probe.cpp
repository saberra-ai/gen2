// Diagnostic only: private C++ headers must match the exact qualified SDK.
// No production runtime patch. Compare one pool-lifetime variable at four threads.
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <vector>

#include "diar_pipeline.h"
#include "ggml-cpu.h"
#include "nemo_speech/asr.h"

namespace {
using Clock = std::chrono::steady_clock;
using NewPool = ggml_threadpool_t (*)(ggml_threadpool_params*);
using FreePool = void (*)(ggml_threadpool_t);
using SetPool = void (*)(ggml_backend_t, ggml_threadpool_t);

ggml_runtime::Params parameters() {
    ggml_runtime::Params params;
    params.use_gpu = false;
    params.gpu_device_idx = 0;
    params.pe_bin_path = const_cast<char*>("");
    if (params.n_threads != 4) throw std::runtime_error("expected four-thread native default");
    return params;
}

struct Pool {
    ggml_threadpool_t value = nullptr;
    FreePool free = nullptr;
    ~Pool() { if (value) free(value); }
    Pool() = default;
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
};

class Backend {
    // Destruction order: detach/pause, destroy backend handles, then free pool.
    Pool pool_;
    ggml_runtime::BackendManager manager_{parameters()};
    ggml_backend_t cpu_ = nullptr;
    SetPool attach_ = nullptr;
public:
    explicit Backend(bool persistent) {
        for (auto backend : manager_.get_backends()) {
            auto device = ggml_backend_get_device(backend);
            if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                if (cpu_) throw std::runtime_error("multiple CPU backends");
                cpu_ = backend;
            }
        }
        if (!cpu_) throw std::runtime_error("CPU backend missing");
        if (persistent) {
            auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu_));
            auto create = reinterpret_cast<NewPool>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
            pool_.free = reinterpret_cast<FreePool>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free"));
            attach_ = reinterpret_cast<SetPool>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
            if (!create || !pool_.free || !attach_) throw std::runtime_error("thread-pool registry functions missing");
            auto params = ggml_threadpool_params_default(4);
            pool_.value = create(&params);
            if (!pool_.value) throw std::runtime_error("thread-pool creation failed");
            attach_(cpu_, pool_.value);
        }
    }
    ~Backend() { if (pool_.value) attach_(cpu_, nullptr); }
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    ggml_runtime::BackendManager& manager() { return manager_; }
};

uint64_t micros(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}

void publish(const std::filesystem::path& path, const void* data, size_t size) {
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) throw std::runtime_error("fresh output file required");
    try {
        auto bytes = static_cast<const char*>(data);
        while (size) {
            auto count = write(fd, bytes, size);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("output write failed");
            bytes += count;
            size -= static_cast<size_t>(count);
        }
        if (fsync(fd)) throw std::runtime_error("output sync failed");
    } catch (...) { close(fd); throw; }
    close(fd);
}
} // namespace

int main(int argc, char** argv) {
    try {
        const char* version = nemo_speech_asr_version();
        if (!version || std::strcmp(version, "nemo-speech-asr 0.2.0+gen2-valid-mel-v1"))
            throw std::runtime_error("qualified runtime version required");
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            for (bool persistent : {false, true}) {
                for (int index = 0; index < 10; ++index) { Backend backend(persistent); }
            }
            std::cout << "{\"model_constructed\":false,\"frontend_executed\":false,"
                         "\"backend_lifetimes\":20,\"persistent_pool_lifetimes\":10,\"threads\":4}\n";
            return 0;
        }
        if (argc != 5 || (std::string(argv[1]) != "disposable" && std::string(argv[1]) != "persistent"))
            throw std::runtime_error("usage: native_pool_probe disposable|persistent MODEL PCM_F32 OUTPUT.json, or --self-test");
        const bool persistent = std::string(argv[1]) == "persistent";
        const uint32_t endian = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian) != 1) throw std::runtime_error("little-endian host required");
        if (std::filesystem::file_size(argv[3]) != 192000 * sizeof(float))
            throw std::runtime_error("exactly twelve seconds of float32 PCM required");
        std::vector<float> pcm(192000);
        std::ifstream input(argv[3], std::ios::binary);
        input.read(reinterpret_cast<char*>(pcm.data()), pcm.size() * sizeof(float));
        if (!input || !std::all_of(pcm.begin(), pcm.end(), [](float v) { return std::isfinite(v) && std::abs(v) <= 1; }))
            throw std::runtime_error("finite normalized PCM required");
        unsigned char hash[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256(pcm.data(), static_cast<CC_LONG>(pcm.size() * sizeof(float)), hash);
        std::ostringstream digest;
        for (auto byte : hash) digest << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
        Backend backend(persistent);
        auto load_start = Clock::now();
        nemo_speech::asr::DiarModel model(backend.manager(), argv[2]);
        const auto load_micros = micros(load_start);
        nemo_speech::asr::DiarGeometry geometry{264, 264, 9, 222, 0, 4};
        geometry = model.resolved_geometry(geometry);
        geometry.validate(model.cfg().num_speakers, model.cfg().scoring.sil_frames_per_spk,
                          model.cfg().encoder.pos_emb_max_len);
        nemo_speech::asr::DiarStream stream(model, geometry);
        auto started = Clock::now();
        for (size_t offset = 0; offset < pcm.size(); offset += 16640)
            stream.feed_audio(pcm.data() + offset, std::min<size_t>(16640, pcm.size() - offset));
        stream.finish();
        const auto elapsed = micros(started);
        const auto& probabilities = stream.frame_probs();
        if (model.cfg().num_speakers != 8 || stream.n_frames() != 1200 || stream.frame_probs_base() != 0 || probabilities.size() != 9600
                || !std::all_of(probabilities.begin(), probabilities.end(), [](float v) { return std::isfinite(v) && 0 <= v && v <= 1; }))
            throw std::runtime_error("unexpected valid-frame output");
        std::ostringstream json;
        json << "{\"scope\":\"private-C++ diagnostic, not public API/release qualification\","
             << "\"persistent_pool\":" << (persistent ? "true" : "false") << ",\"threads\":4,"
             << "\"audio_samples\":192000,\"input_pcm_f32le_sha256\":\"" << digest.str() << "\","
             << "\"frames\":1200,\"channels\":8,\"retained_first_frame\":0,"
             << "\"geometry\":{\"chunk\":9,\"right\":4,\"left\":0,\"fifo\":264,\"cache\":264,\"update\":222},"
             << "\"load_micros\":" << load_micros << ",\"elapsed_micros\":" << elapsed << "}\n";
        const std::string raw_path = std::string(argv[4]) + ".f32", text = json.str();
        publish(raw_path, probabilities.data(), probabilities.size() * sizeof(float));
        publish(argv[4], text.data(), text.size());
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
