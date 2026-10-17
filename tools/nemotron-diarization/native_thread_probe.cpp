// Qualification only. All private C++ headers must match the owned-pool SDK.
// The newly compiled BackendManager owns its pool; no second pool is attached.
#include <CommonCrypto/CommonDigest.h>
#include <mach/mach.h>
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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>
#include "diar_pipeline.h"
#include "nemo_speech/asr.h"

namespace {
using Clock = std::chrono::steady_clock;
ggml_runtime::Params parameters(int threads) {
    if (threads != 3 && threads != 4) throw std::runtime_error("only three or four threads admitted");
    ggml_runtime::Params params;
    if (params.n_threads != 4) throw std::runtime_error("native default is not four threads");
    params.n_threads = threads;
    params.use_gpu = false;
    params.gpu_device_idx = 0;
    params.pe_bin_path = const_cast<char*>("");
    return params;
}
unsigned thread_count() {
    thread_act_array_t values = nullptr;
    mach_msg_type_number_t count = 0;
    if (task_threads(mach_task_self(), &values, &count) != KERN_SUCCESS)
        throw std::runtime_error("cannot inspect own task threads");
    for (unsigned i = 0; i < count; ++i) mach_port_deallocate(mach_task_self(), values[i]);
    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(values), count * sizeof(thread_t));
    return count;
}
uint64_t micros(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count();
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
void lifetime(int threads) {
    ggml_runtime::BackendManager manager(parameters(threads));
    if (manager.get_params().n_threads != threads) throw std::runtime_error("manager thread count differs");
    unsigned cpus = 0;
    for (auto backend : manager.get_backends()) {
        auto device = ggml_backend_get_device(backend);
        const auto kind = ggml_backend_dev_type(device);
        if (kind == GGML_BACKEND_DEVICE_TYPE_CPU) ++cpus;
        else if (kind != GGML_BACKEND_DEVICE_TYPE_ACCEL || std::string(ggml_backend_dev_name(device)) != "BLAS")
            throw std::runtime_error("unexpected device outside CPU/BLAS");
    }
    if (cpus != 1) throw std::runtime_error("exactly one CPU backend required");
}
struct LifetimeObservation {
    int threads;
    unsigned attempt;
    std::optional<unsigned> immediate, settled;
    unsigned waited = 0;
};
void self_test(bool inject_failure) {
    std::vector<LifetimeObservation> observations;
    observations.reserve(20);
    std::optional<unsigned> baseline;
    unsigned completed = 0;
    const char* phase = "warmup";
    auto emit = [&](bool complete) {
        std::cout << "{\"complete\":" << (complete ? "true" : "false")
                  << ",\"phase\":\"" << phase << "\",\"backend_lifetimes\":" << completed
                  << ",\"model_constructed\":false,\"frontend_executed\":false,\"warmed_thread_baseline\":";
        if (baseline) std::cout << *baseline; else std::cout << "null";
        std::cout << ",\"thread_growth_allowance\":2,\"cases\":[";
        for (int threads : {4, 3}) {
            std::cout << (threads == 3 ? "," : "") << "{\"threads\":" << threads << ",\"observations\":[";
            bool first = true;
            for (const auto& row : observations) {
                if (row.threads != threads) continue;
                std::cout << (first ? "" : ",") << "{\"attempt\":" << row.attempt << ",\"immediate\":";
                first = false;
                if (row.immediate) std::cout << *row.immediate; else std::cout << "null";
                std::cout << ",\"settled\":";
                if (row.settled) std::cout << *row.settled; else std::cout << "null";
                std::cout << ",\"requested_wait_ms\":" << row.waited << '}';
            }
            std::cout << "]}";
        }
        std::cout << "]}\n" << std::flush;
    };
    try {
        for (int threads : {4, 3}) { lifetime(threads); ++completed; }
        phase = "baseline";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        baseline = thread_count();
        for (int threads : {4, 3}) {
            for (unsigned i = 0; i < 10; ++i) {
                observations.push_back({threads, i+1, std::nullopt, std::nullopt, 0});
                auto& row = observations.back();
                phase = "lifetime";
                lifetime(threads);
                ++completed;
                phase = "immediate-count";
                row.immediate = thread_count();
                phase = "settled-count";
                do {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    row.waited += 10;
                    row.settled = thread_count();
                } while (*row.settled > *baseline + 2 && row.waited < 100);
                phase = "growth-check";
                if (*row.settled > *baseline + 2) throw std::runtime_error("backend lifetimes accumulate threads");
            }
        }
        if (inject_failure) {
            phase = "injected-failure-control";
            throw std::runtime_error("injected lifetime evidence control");
        }
        phase = "complete";
    } catch (...) { emit(false); throw; }
    emit(true);
}
} // namespace

int main(int argc, char** argv) {
    try {
        const char* version = nemo_speech_asr_version();
        if (!version || std::strcmp(version, "nemo-speech-asr 0.2.0+gen2-valid-mel-pool-v1"))
            throw std::runtime_error("exact owned-pool runtime required");
        if (argc == 2 && (std::string(argv[1]) == "--self-test" || std::string(argv[1]) == "--self-test-failure-control")) {
            self_test(std::string(argv[1]) == "--self-test-failure-control"); return 0;
        }
        if (argc != 5 || (std::string(argv[1]) != "3" && std::string(argv[1]) != "4"))
            throw std::runtime_error("usage: native_thread_probe 3|4 MODEL PCM_F32 OUTPUT.json, or --self-test[ -failure-control]");
        const int threads = argv[1][0] - '0';
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
        ggml_runtime::BackendManager backend(parameters(threads));
        auto load_start = Clock::now();
        nemo_speech::asr::DiarModel model(backend, argv[2]);
        const auto load_micros = micros(load_start);
        nemo_speech::asr::DiarGeometry geometry{264, 264, 9, 222, 0, 4};
        geometry = model.resolved_geometry(geometry);
        geometry.validate(model.cfg().num_speakers, model.cfg().scoring.sil_frames_per_spk, model.cfg().encoder.pos_emb_max_len);
        nemo_speech::asr::DiarStream stream(model, geometry);
        auto started = Clock::now();
        for (size_t offset = 0; offset < pcm.size(); offset += 16640)
            stream.feed_audio(pcm.data()+offset, std::min<size_t>(16640, pcm.size()-offset));
        stream.finish();
        const auto elapsed = micros(started);
        const auto& probabilities = stream.frame_probs();
        if (model.cfg().num_speakers != 8 || stream.n_frames() != 1200 || stream.frame_probs_base() != 0 || probabilities.size() != 9600
                || !std::all_of(probabilities.begin(), probabilities.end(), [](float v) { return std::isfinite(v) && 0 <= v && v <= 1; }))
            throw std::runtime_error("unexpected valid-frame output");
        std::ostringstream json;
        json << std::setprecision(17) << "{\"scope\":\"matched-private-C++ thread diagnostic; not public API/release qualification\","
             << "\"threads\":" << backend.get_params().n_threads << ",\"runtime_version\":\"" << version << "\","
             << "\"audio_samples\":192000,\"audio_duration_seconds\":12,\"input_repeats\":1,\"requested_gpu_index\":-1,"
             << "\"input_pcm_f32le_sha256\":\"" << digest.str() << "\","
             << "\"native_frame_count\":1200,\"retained_first_frame\":0,\"retained_rows\":1200,"
             << "\"geometry\":{\"chunk\":9,\"right\":4,\"left\":0,\"fifo\":264,\"cache\":264,\"update\":222},"
             << "\"load_micros\":" << load_micros << ",\"elapsed_micros\":" << elapsed << ",\"turns\":[";
        bool first = true;
        double previous_start = 0;
        for (const auto& turn : stream.segments()) {
            if (!std::isfinite(turn.t0) || !std::isfinite(turn.t1) || turn.t0 < previous_start
                    || turn.t0 < 0 || turn.t1 < turn.t0 || turn.t1 > 12. + 1e-4
                    || turn.speaker < 0 || turn.speaker >= 8)
                throw std::runtime_error("invalid native turn before rounding clamp");
            previous_start = turn.t0;
            if (!first) json << ',';
            first = false;
            json << "{\"start_seconds\":" << std::clamp(turn.t0, 0., 12.) << ",\"end_seconds\":"
                 << std::clamp(turn.t1, 0., 12.) << ",\"speaker\":" << turn.speaker+1 << '}';
        }
        json << "]}\n";
        const std::string raw_path = std::string(argv[4])+".f32", text = json.str();
        publish(raw_path, probabilities.data(), probabilities.size()*sizeof(float));
        publish(argv[4], text.data(), text.size());
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
