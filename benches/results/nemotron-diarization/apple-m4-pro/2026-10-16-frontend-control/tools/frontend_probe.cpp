// Qualification-only CPU frontend access. Does not load a neural model or
// change the stable C ABI. Compile against the pinned native C++ SDK.
#include "fe.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static std::vector<float> read_f32(const char* path, size_t maximum) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open frontend input");
    const auto bytes = input.tellg();
    if (bytes <= 0 || bytes % 4 != 0 || bytes / 4 > static_cast<std::streamoff>(maximum))
        throw std::runtime_error("invalid or oversized frontend input");
    std::vector<float> values(static_cast<size_t>(bytes) / 4);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(values.data()), bytes))
        throw std::runtime_error("cannot read complete frontend input");
    for (float value : values)
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite frontend input");
    return values;
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: frontend-probe PCM.f32 BANK.f32 OUTPUT.f32");
        static_assert(sizeof(float) == 4);
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian) != 1)
            throw std::runtime_error("frontend diagnostic requires little-endian host");
        if (std::filesystem::exists(argv[3])) throw std::runtime_error("refusing to overwrite frontend output");
        auto pcm = read_f32(argv[1], 16000 * 120);
        auto bank = read_f32(argv[2], 128 * 257);
        if (bank.size() != 128 * 257) throw std::runtime_error("frontend bank must be [128,257]");
        for (float value : bank)
            if (value < 0) throw std::runtime_error("negative frontend bank weight");
        MelSpecConfig config;
        config.sample_rate = 16000;
        config.n_fft = 512;
        config.n_mels = 128;
        config.window_size = 0.025f;
        config.window_stride = 0.01f;
        config.preemph = 0.97f;
        config.log_zero_guard = 1.0f / 16777216.0f;
        config.stft_center_window = true;
        config.hann_periodic = false;
        config.mask_invalid_frames = false;
        MelSpectrogramExtractor frontend(config, nullptr);
        frontend.set_mel_basis(bank.data(), 128, 257);
        if (frontend.uses_gpu()) throw std::runtime_error("frontend unexpectedly selected GPU");
        std::vector<float> features;
        int frames = 0;
        frontend.compute(pcm.data(), pcm.size(), features, frames, true, false);
        if (frames != static_cast<int>(pcm.size() / 160 + 1) || features.size() != static_cast<size_t>(frames) * 128)
            throw std::runtime_error("unexpected centered frontend shape");
        for (float value : features)
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite frontend output");
        std::ofstream output(argv[3], std::ios::binary);
        if (!output.write(reinterpret_cast<const char*>(features.data()), features.size() * 4))
            throw std::runtime_error("cannot write frontend output");
        output.close();
        if (!output) throw std::runtime_error("cannot close frontend output");
        std::cout << "{\"audio_samples\":" << pcm.size() << ",\"frames\":" << frames
                  << ",\"mel_bins\":128,\"center\":true,\"normalize\":false,\"uses_gpu\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
