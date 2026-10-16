// Qualification-only fixed V3 graph access, no frontend or native AOSC.
#include "sortformer_model.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<float> read_f32(const char* path, size_t maximum, bool allow_empty) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open graph input");
    const auto bytes = input.tellg();
    if (bytes < 0 || bytes % 4 || bytes / 4 > static_cast<std::streamoff>(maximum) || (!allow_empty && bytes == 0))
        throw std::runtime_error("invalid graph input size");
    std::vector<float> values(static_cast<size_t>(bytes) / 4);
    input.seekg(0);
    if (bytes && !input.read(reinterpret_cast<char*>(values.data()), bytes))
        throw std::runtime_error("incomplete graph input");
    for (float value : values)
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite graph input");
    return values;
}

static void write_f32(const std::string& path, const std::vector<float>& values, bool probabilities) {
    for (float value : values)
        if (!std::isfinite(value) || (probabilities && (value < 0 || value > 1)))
            throw std::runtime_error("invalid graph output value");
    std::ofstream output(path, std::ios::binary);
    if (!output.write(reinterpret_cast<const char*>(values.data()), values.size() * 4))
        throw std::runtime_error("cannot write graph output");
    output.close();
    if (!output) throw std::runtime_error("cannot close graph output");
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: calibration-graph-probe MODEL MEL.f32 FIFO.f32 OUTPUT-PREFIX");
        static_assert(sizeof(float) == 4);
        const uint16_t endian = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian) != 1)
            throw std::runtime_error("little-endian host required");
        const std::string output = argv[4];
        for (const auto* suffix : {".json", ".predictions.f32", ".embeddings.f32"})
            if (std::filesystem::exists(output + suffix))
                throw std::runtime_error("refusing graph output overwrite");
        auto mel = read_f32(argv[2], 104 * 128, false);
        auto fifo = read_f32(argv[3], 9 * 512, true);
        const size_t mel_rows = mel.size() / 128;
        const size_t fifo_rows = fifo.size() / 512;
        if (mel.size() % 128 || fifo.size() % 512 ||
            !((mel_rows == 104 && fifo_rows == 0) || ((mel_rows == 39 || mel_rows == 40) && fifo_rows == 9)))
            throw std::runtime_error("inputs outside fixed calibration graph cases");
        ggml_runtime::Params params;
        params.use_gpu = false;
        params.n_threads = 4;
        ggml_runtime::BackendManager backend(params);
        nemo_speech::asr::BatchingConfig batching;
        batching.enabled = false;
        nemo_speech::asr::SortformerModel model(backend, argv[1], batching);
        const auto& cfg = model.cfg();
        if (!cfg.is_v3() || !cfg.high_resolution || !cfg.learnable_silence || cfg.num_speakers != 8 ||
            cfg.n_mels != 128 || cfg.encoder.d_model != 512 || cfg.encoder.subsampling_factor != 8 ||
            cfg.upsample_factor != 8 || cfg.output_subsampling_factor != 1)
            throw std::runtime_error("graph model is not the frozen V3 contract");
        auto result = model.run_chunk(mel.data(), static_cast<int>(mel_rows), nullptr, 0,
                                      fifo.empty() ? nullptr : fifo.data(), static_cast<int>(fifo_rows));
        const int chunk_frames = static_cast<int>((mel_rows + 7) / 8);
        const int total_frames = chunk_frames + static_cast<int>(fifo_rows);
        if (result.chunk_frames != chunk_frames || result.total_frames != total_frames ||
            result.native_total_frames != total_frames * 8 ||
            result.chunk_embs.size() != static_cast<size_t>(chunk_frames) * 512 ||
            result.native_preds.size() != static_cast<size_t>(total_frames) * 8 * 8)
            throw std::runtime_error("unexpected graph output shape");
        write_f32(output + ".predictions.f32", result.native_preds, true);
        write_f32(output + ".embeddings.f32", result.chunk_embs, false);
        std::ofstream metadata(output + ".json");
        metadata << "{\"mel_rows\":" << mel_rows << ",\"fifo_rows\":" << fifo_rows
                 << ",\"speaker_cache_rows\":0,\"chunk_encoder_rows\":" << chunk_frames
                 << ",\"total_encoder_rows\":" << total_frames
                 << ",\"native_probability_rows\":" << result.native_total_frames
                 << ",\"speakers\":8,\"hidden_size\":512,\"cpu_threads\":4,\"gpu_requested\":false,\"batching\":false}\n";
        metadata.close();
        if (!metadata) throw std::runtime_error("cannot close graph metadata");
        std::cout << "{\"fixed_native_graph_call_completed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
