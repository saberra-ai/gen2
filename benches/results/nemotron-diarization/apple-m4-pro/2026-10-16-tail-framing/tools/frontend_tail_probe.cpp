// Qualification-only access to the pinned streaming frontend. No neural model.
#include "fe.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static std::vector<float> read_f32(const char* path, size_t maximum) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open input");
    const auto bytes = input.tellg();
    if (bytes <= 0 || bytes % 4 || bytes / 4 > static_cast<std::streamoff>(maximum))
        throw std::runtime_error("invalid input size");
    std::vector<float> values(static_cast<size_t>(bytes) / 4);
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(values.data()), bytes))
        throw std::runtime_error("incomplete input");
    for (float value : values)
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite input");
    return values;
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: frontend-tail-probe PCM.f32 BANK.f32 OUTPUT.f32");
        static_assert(sizeof(float) == 4);
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian) != 1)
            throw std::runtime_error("little-endian host required");
        if (std::filesystem::exists(argv[3])) throw std::runtime_error("refusing output overwrite");
        auto pcm = read_f32(argv[1], 16000 * 120);
        auto bank = read_f32(argv[2], 128 * 257);
        if (pcm.size() < 320 || bank.size() != 128 * 257)
            throw std::runtime_error("invalid PCM or bank shape");
        for (float value : bank)
            if (value < 0) throw std::runtime_error("negative bank value");
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
        if (frontend.uses_gpu()) throw std::runtime_error("unexpected GPU frontend");
        std::vector<float> audio, features;
        std::vector<int64_t> available;
        auto release = [&]() {
            std::vector<float> next;
            const auto first = static_cast<int64_t>(features.size() / 128);
            const int count = produce_new_mel_frames(frontend, audio, 0, first, next);
            if (count < 0 || next.size() != static_cast<size_t>(count) * 128)
                throw std::runtime_error("stream frontend shape differs");
            features.insert(features.end(), next.begin(), next.end());
        };
        // Match the frozen native helper's real transport pushes. This calls
        // upstream FE directly; chunk scheduling below the NN is not executed.
        for (size_t start = 0; start < pcm.size(); start += 16640) {
            const auto end = std::min(start + 16640, pcm.size());
            audio.insert(audio.end(), pcm.begin() + start, pcm.begin() + end);
            release();
            available.push_back(static_cast<int64_t>(features.size() / 128));
        }
        const size_t before_finish = features.size() / 128;
        // Exact source finish() tail: zero AFTER preemphasis via decay in PCM.
        float tail = audio.back();
        for (int k = 0; k < frontend.n_fft() / 2; ++k) {
            tail *= config.preemph;
            audio.push_back(tail);
        }
        release();
        const size_t frames = features.size() / 128;
        if (frames != pcm.size() / 160 + 1)
            throw std::runtime_error("unexpected final raw frame count");
        for (float value : features)
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite output");
        std::ofstream output(argv[3], std::ios::binary);
        if (!output.write(reinterpret_cast<const char*>(features.data()), features.size() * 4))
            throw std::runtime_error("cannot write output");
        output.close();
        if (!output) throw std::runtime_error("cannot close output");
        std::cout << "{\"audio_samples\":" << pcm.size() << ",\"raw_frames\":" << frames
                  << ",\"before_finish_frames\":" << before_finish
                  << ",\"mel_bins\":128,\"push_samples\":16640,\"uses_gpu\":false,\"available_after_push\":[";
        for (size_t i = 0; i < available.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << available[i];
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
