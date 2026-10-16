// Candidate-only V3 finish regression. Does not alter the public C ABI.
#include "diar_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

using nemo_speech::asr::DiarGeometry;
using nemo_speech::asr::DiarModel;
using nemo_speech::asr::DiarStream;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static std::vector<float> read_f32(const char* name, size_t elements) {
    std::ifstream input(name, std::ios::binary | std::ios::ate);
    require(bool(input), "cannot open fixed candidate input");
    require(input.tellg() == static_cast<std::streamoff>(elements * 4), "candidate input size differs");
    std::vector<float> values(elements);
    input.seekg(0);
    require(bool(input.read(reinterpret_cast<char*>(values.data()), elements * 4)), "truncated candidate input");
    for (float value : values) require(std::isfinite(value), "nonfinite candidate input");
    return values;
}

// Persist available tensors before validating them, including partial failures.
static void write_f32(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary);
    require(bool(output), "cannot open candidate tensor");
    if (!values.empty()) output.write(reinterpret_cast<const char*>(values.data()), values.size() * 4);
    output.close();
    require(bool(output), "cannot close candidate tensor");
}

static void validate(const std::vector<float>& values, bool probabilities) {
    for (float value : values)
        require(std::isfinite(value) && (!probabilities || (value >= 0 && value <= 1)), "invalid candidate tensor");
}

static void event(std::ofstream& report, const std::string& name, const char* status) {
    report << "{\"case\":\"" << name << "\",\"event\":\"" << status << "\"}\n";
    report.flush();
    require(bool(report), "cannot persist candidate event");
}

static void snapshot(DiarStream& stream, const std::filesystem::path& prefix) {
    write_f32(prefix.string() + ".f32", stream.frame_probs());
    std::ofstream metadata(prefix.string() + ".json");
    metadata << "{\"frames\":" << stream.n_frames() << ",\"stable_frames\":" << stream.stable_frames()
             << ",\"retained_base\":" << stream.frame_probs_base()
             << ",\"elements\":" << stream.frame_probs().size() << "}\n";
    metadata.close();
    require(bool(metadata), "cannot persist candidate snapshot");
}

static float max_difference(const std::vector<float>& left, const std::vector<float>& right) {
    require(left.size() == right.size(), "candidate probability shape mismatch");
    float maximum = 0;
    for (size_t i = 0; i < left.size(); ++i) maximum = std::max(maximum, std::fabs(left[i] - right[i]));
    return maximum;
}

static std::vector<float> reference(DiarModel& model, const std::vector<float>& valid_mel, int left_context,
                                     const std::filesystem::path& output, std::ofstream& report) {
    // Independent graph assembly: explicit 111 valid rows, no DiarStream finish.
    const std::string label = "direct-left" + std::to_string(left_context);
    std::vector<float> first_mel(valid_mel.begin(), valid_mel.begin() + 104 * 128);
    write_f32(output / (label + "-first.mel.f32"), first_mel);
    event(report, label + "-first", "started");
    auto first = model.model().run_chunk(first_mel.data(), 104, nullptr, 0, nullptr, 0);
    write_f32(output / (label + "-first.predictions.f32"), first.native_preds);
    write_f32(output / (label + "-first.embeddings.f32"), first.chunk_embs);
    event(report, label + "-first", "returned");
    validate(first.native_preds, true);
    validate(first.chunk_embs, false);
    require(first.chunk_frames == 13 && first.total_frames == 13 && first.native_total_frames == 104 &&
            first.native_preds.size() == 104 * 8 && first.chunk_embs.size() == 13 * 512, "first direct reference shape differs");
    event(report, label + "-first", "completed");
    const int begin_mel = 72 - left_context * 8;
    const int count = 111 - begin_mel;
    std::vector<float> final_mel(valid_mel.begin() + begin_mel * 128, valid_mel.end());
    std::vector<float> fifo(first.chunk_embs.begin(), first.chunk_embs.begin() + 9 * 512);
    write_f32(output / (label + "-final.mel.f32"), final_mel);
    write_f32(output / (label + "-final.fifo.f32"), fifo);
    event(report, label + "-final", "started");
    auto last = model.model().run_chunk(final_mel.data(), count, nullptr, 0, fifo.data(), 9);
    write_f32(output / (label + "-final.predictions.f32"), last.native_preds);
    write_f32(output / (label + "-final.embeddings.f32"), last.chunk_embs);
    event(report, label + "-final", "returned");
    validate(last.native_preds, true);
    validate(last.chunk_embs, false);
    const int chunk_frames = (count + 7) / 8;
    const int total_frames = 9 + chunk_frames;
    const int offset = (9 + left_context) * 8;
    require(last.chunk_frames == chunk_frames && last.total_frames == total_frames &&
            last.native_total_frames == total_frames * 8 && last.native_preds.size() == static_cast<size_t>(total_frames) * 64 &&
            last.chunk_embs.size() == static_cast<size_t>(chunk_frames) * 512 && last.native_total_frames >= offset + 39,
            "final direct reference shape differs");
    event(report, label + "-final", "completed");
    std::vector<float> result(first.native_preds.begin(), first.native_preds.begin() + 72 * 8);
    result.insert(result.end(), last.native_preds.begin() + offset * 8, last.native_preds.begin() + (offset + 39) * 8);
    return result;
}

static std::vector<float> finish_case(DiarStream& stream, const std::vector<float>& audio,
                                      const std::vector<size_t>& pushes, bool preview, bool& saw_preview,
                                      const std::filesystem::path& output, const std::string& name, std::ofstream& report) {
    event(report, name, "started");
    size_t offset = 0, iteration = 0;
    std::vector<float> stable_prefix;
    while (offset < audio.size()) {
        const size_t length = std::min(pushes[iteration++ % pushes.size()], audio.size() - offset);
        stream.feed_audio(audio.data() + offset, length);
        offset += length;
        if (preview) {
            stream.flush_available(static_cast<int64_t>(offset / 160));
            saw_preview = saw_preview || stream.stable_frames() < stream.n_frames();
        }
        snapshot(stream, output / (name + "-push-" + std::to_string(iteration)));
        validate(stream.frame_probs(), true);
        require(stream.frame_probs_base() == 0, "unexpected probability compaction in bounded calibration");
        const auto& current = stream.frame_probs();
        require(current.size() >= stable_prefix.size(), "stable prefix disappeared");
        require(std::equal(stable_prefix.begin(), stable_prefix.end(), current.begin()), "previous stable rows changed");
        const auto stable = stream.stable_frames();
        require(stable >= 0 && stable <= stream.n_frames() && static_cast<size_t>(stable) * 8 <= current.size(), "invalid stable frontier bounds");
        stable_prefix.assign(current.begin(), current.begin() + stable * 8);
    }
    event(report, name, "finish-started");
    stream.finish();
    snapshot(stream, output / (name + "-finish"));
    event(report, name, "finish-returned");
    validate(stream.frame_probs(), true);
    require(stream.n_frames() == static_cast<int64_t>(audio.size() / 160), "candidate final valid-frame count differs");
    require(stream.stable_frames() == stream.n_frames(), "candidate final stable frontier differs");
    require(stream.frame_probs_base() == 0, "unexpected final retained base");
    auto result = stream.frame_probs();
    require(result.size() == static_cast<size_t>(stream.n_frames()) * 8, "candidate retained shape differs");
    require(result.size() >= stable_prefix.size() && std::equal(stable_prefix.begin(), stable_prefix.end(), result.begin()), "finish revised stable rows");
    stream.finish();
    const float ignored = .125f;
    stream.feed_audio(&ignored, 1);
    snapshot(stream, output / (name + "-repeated-late"));
    require(stream.frame_probs() == result && stream.n_frames() == static_cast<int64_t>(audio.size() / 160), "repeat finish or late push changed result");
    event(report, name, "completed");
    return result;
}

int main(int argc, char** argv) {
    try {
        require(argc == 5, "usage: valid-mel-finish-probe MODEL PCM17872.f32 NATIVE112x128.f32 OUTPUT-DIR");
        static_assert(sizeof(float) == 4);
        const uint16_t endian = 1;
        require(*reinterpret_cast<const uint8_t*>(&endian) == 1, "little-endian host required");
        const std::filesystem::path output = argv[4];
        require(!std::filesystem::exists(output), "refusing candidate output overwrite");
        auto audio = read_f32(argv[2], 17872);
        auto mel = read_f32(argv[3], 112 * 128);
        mel.resize(111 * 128);
        std::filesystem::create_directory(output);
        std::ofstream report(output / "cases.jsonl");
        require(bool(report), "cannot open candidate case report");
        report << std::setprecision(17);
        event(report, "model", "load-started");
        ggml_runtime::Params params;
        params.use_gpu = false;
        params.n_threads = 4;
        ggml_runtime::BackendManager backend(params);
        nemo_speech::asr::BatchingConfig batching;
        batching.enabled = false;
        DiarModel model(backend, argv[1], batching);
        require(model.cfg().is_v3() && model.cfg().n_mels == 128 && model.cfg().num_speakers == 8 && model.cfg().output_subsampling_factor == 1, "candidate model contract differs");
        event(report, "model", "load-completed");
        DiarGeometry geometry;
        geometry.chunk_len = 9;
        geometry.chunk_right_context = 4;
        geometry.chunk_left_context = 0;
        geometry.fifo_len = 264;
        geometry.spkcache_len = 264;
        geometry.spkcache_update_period = 222;
        auto direct = reference(model, mel, 0, output, report);
        write_f32(output / "direct-valid111.f32", direct);
        const std::vector<std::vector<size_t>> schedules{{17872}, {16640}, {317, 3, 1021, 160, 997}, {8000, 4000, 5872}};
        const std::vector<std::string> names{"whole", "transport16640", "uneven", "provisional"};
        // A fixed 1e-4 same-runtime policy is declared before execution; it is
        // separate from all original HF/DER/release gates.
        constexpr float tolerance = 1.e-4f;
        for (size_t i = 0; i < schedules.size(); ++i) {
            DiarStream stream(model, geometry);
            bool saw_preview = false;
            auto actual = finish_case(stream, audio, schedules[i], i == 3, saw_preview, output, names[i], report);
            write_f32(output / (names[i] + ".f32"), actual);
            const float error = max_difference(actual, direct);
            report << "{\"case\":\"" << names[i] << "\",\"frames\":111,\"maximum_probability_error\":" << error
                   << ",\"saw_provisional\":" << (saw_preview ? "true" : "false") << "}\n";
            report.flush();
            require(error <= tolerance, "candidate pipeline differs from valid-Mel direct graph reference");
            if (i == 3) require(saw_preview, "provisional regression never produced a preview");
            if (i == 0) {
                stream.reset();
                require(stream.n_frames() == 0 && stream.frame_probs().empty(), "reset did not clear candidate stream");
                bool ignored = false;
                auto replay = finish_case(stream, audio, schedules[i], false, ignored, output, "reset-replay", report);
                write_f32(output / "reset-replay.f32", replay);
                require(replay == actual, "reset/replay changed candidate output");
            }
        }
        const std::vector<size_t> lengths{0, 1, 159, 160, 161, 319, 320, 321, 11519, 11520, 11521,
                                         16639, 16640, 16679, 16680, 16735, 16736, 16799, 16800, 17872, 17920};
        for (size_t samples : lengths) {
            std::vector<float> silence(samples, 0.f);
            DiarStream stream(model, geometry);
            bool ignored = false;
            auto actual = finish_case(stream, silence, {samples ? samples : 1}, false, ignored, output, "boundary-" + std::to_string(samples), report);
            write_f32(output / ("boundary-" + std::to_string(samples) + ".f32"), actual);
            report << "{\"case\":\"boundary\",\"samples\":" << samples << ",\"frames\":" << stream.n_frames() << "}\n";
            report.flush();
        }
        geometry.chunk_left_context = 1;
        auto left_reference = reference(model, mel, 1, output, report);
        write_f32(output / "left1-direct.f32", left_reference);
        for (size_t i = 0; i < 2; ++i) {
            DiarStream stream(model, geometry);
            bool ignored = false;
            auto actual = finish_case(stream, audio, i ? schedules[2] : schedules[0], false, ignored, output, i ? "left1-uneven" : "left1-whole", report);
            write_f32(output / (i ? "left1-uneven.f32" : "left1-whole.f32"), actual);
            const float error = max_difference(actual, left_reference);
            report << "{\"case\":\"left1-" << (i ? "uneven" : "whole") << "\",\"frames\":111,\"maximum_probability_error\":" << error << "}\n";
            report.flush();
            require(error <= tolerance, "left-context candidate differs from direct graph reference");
        }
        report.close();
        require(bool(report), "cannot close candidate case report");
        std::cout << "{\"fixed_candidate_finish_regression_completed\":true,\"stream_sessions\":28,\"direct_graph_calls\":4,\"boundary_cases\":21,\"tolerance\":0.0001}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
