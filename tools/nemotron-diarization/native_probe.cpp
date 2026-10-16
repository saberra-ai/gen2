// N0 experiment only: compile against the pinned installed public header.
// This shim keeps upstream C struct layouts out of the Gen2 coexistence driver.
#include "nemo_speech/diar.h"
#include "wav_reader.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
void check(nemo_speech_asr_status status) {
    if (status != NEMO_SPEECH_ASR_OK) throw std::runtime_error(nemo_speech_asr_last_error());
}
using Model = std::unique_ptr<nemo_speech_diar_model, decltype(&nemo_speech_diar_destroy)>;
using Stream = std::unique_ptr<nemo_speech_diar_stream, decltype(&nemo_speech_diar_stream_close)>;
struct ProbeModel { Model model; bool official_geometry; };
}

extern "C" const char* n0_last_error() { return error.c_str(); }
extern "C" void n0_close(void* model) {
    delete static_cast<ProbeModel*>(model);
}
extern "C" void* n0_open(const char* path, int official_geometry) noexcept {
    try {
        nemo_speech_diar_model_config cfg{};
        cfg.size = sizeof(cfg);
        cfg.model_path = path;
        cfg.gpu = -1;
        cfg.preset = "v3-streaming";
        if (official_geometry != 0 && official_geometry != 1)
            throw std::runtime_error("unknown qualification geometry");
        cfg.left_context_frames = 0;
        if (official_geometry) {
            // Official Transformers low_latency geometry, explicitly matched.
            // The SDK's v3-streaming default is 13/1, FIFO80, update40 instead.
            cfg.chunk_frames = 9;
            cfg.right_context_frames = 4;
            cfg.fifo_frames = 264;
            cfg.spkcache_frames = 264;
            cfg.update_period_frames = 222;
        }
        nemo_speech_diar_model* model = nullptr;
        check(nemo_speech_diar_create(&cfg, &model));
        Model owned(model, nemo_speech_diar_destroy);
        if (nemo_speech_diar_num_speakers(model) != 8 ||
            std::abs(nemo_speech_diar_seconds_per_frame(model) - .01) > 1e-9)
            throw std::runtime_error("not the locked V3 eight-speaker model");
        return new ProbeModel{std::move(owned), official_geometry == 1};
    } catch (const std::exception& e) { error = e.what(); return nullptr; }
    catch (...) { error = "unknown native failure"; return nullptr; }
}

extern "C" int n0_run(void* pointer, const char* wav, const char* output_path) noexcept {
    try {
        auto* owner = static_cast<ProbeModel*>(pointer);
        if (!owner) throw std::runtime_error("null model");
        auto* model = owner->model.get();
        std::vector<float> audio;
        std::string message;
        if (!examples::read_wav_mono_16k(wav, audio, message)) throw std::runtime_error(message);
        if (audio.empty() || audio.size() > 16000 * 120 ||
            !std::all_of(audio.begin(), audio.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("N0 probe accepts finite 16 kHz audio up to 120 seconds");
        auto started = std::chrono::steady_clock::now();
        nemo_speech_diar_stream* raw = nullptr;
        check(nemo_speech_diar_stream_open(model, &raw));
        Stream stream(raw, nemo_speech_diar_stream_close);
        for (size_t off = 0; off < audio.size(); off += 2560)
            check(nemo_speech_diar_stream_push_f32(raw, audio.data() + off,
                                                  std::min(size_t(2560), audio.size() - off), 16000));
        check(nemo_speech_diar_stream_finish(raw));
        size_t count = 0;
        check(nemo_speech_diar_segments(raw, nullptr, nullptr, 0, &count));
        if (count > 100000) throw std::runtime_error("excessive segment count");
        std::vector<nemo_speech_diar_segment> segments(count);
        if (count) check(nemo_speech_diar_segments(raw, nullptr, segments.data(), count, &count));
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const auto frames = nemo_speech_diar_frame_count(raw);
        if (frames < 0 || frames > 13000 || nemo_speech_diar_frame_probs_start(raw) != 0)
            throw std::runtime_error("unexpected frame count/compaction for short N0 probe");
        std::vector<float> probabilities(static_cast<size_t>(frames) * 8);
        check(nemo_speech_diar_frame_probs(raw, probabilities.data(), probabilities.size()));
        std::ofstream probability_output(std::string(output_path) + ".f32", std::ios::binary);
        probability_output.exceptions(std::ios::failbit | std::ios::badbit);
        probability_output.write(reinterpret_cast<const char*>(probabilities.data()),
                                 probabilities.size() * sizeof(float));
        probability_output.close();
        std::ofstream output(output_path);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output << std::setprecision(17) << "{\"seconds\":" << seconds
               << ",\"frames\":" << nemo_speech_diar_frame_count(raw)
               << ",\"audio_seconds\":" << audio.size() / 16000.0
               << ",\"geometry\":{\"chunk\":" << (owner->official_geometry ? 9 : 13)
               << ",\"right\":" << (owner->official_geometry ? 4 : 1)
               << ",\"left\":0,\"fifo\":" << (owner->official_geometry ? 264 : 80)
               << ",\"cache\":264,\"update\":" << (owner->official_geometry ? 222 : 40)
               << "},\"segments\":[";
        for (size_t i = 0; i < count; ++i) {
            const auto& s = segments[i];
            if (!std::isfinite(s.start_time) || !std::isfinite(s.end_time) ||
                s.end_time < s.start_time || s.speaker < 1 || s.speaker > 8)
                throw std::runtime_error("invalid native segment");
            if (i) output << ',';
            output << "{\"start\":" << s.start_time << ",\"end\":" << s.end_time
                   << ",\"speaker\":" << s.speaker << '}';
        }
        output << "]}\n";
        output.close();
        return 0;
    } catch (const std::exception& e) { error = e.what(); return 1; }
    catch (...) { error = "unknown native failure"; return 1; }
}
