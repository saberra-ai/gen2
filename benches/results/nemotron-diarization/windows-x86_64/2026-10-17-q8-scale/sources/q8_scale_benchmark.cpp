// Model-free comparison of two exact-source SGEMM translation units.
#include "ggml-quants.h"
#include "ggml-cpu-impl.h"
#include "sgemm.h"
#include <algorithm>
#include <array>
#include <windows.h>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

extern "C" bool gen2_q8_tile_control_sgemm(const ggml_compute_params *, int64_t, int64_t, int64_t,
    const void *, int64_t, const void *, int64_t, void *, int64_t, int, int, int);
extern "C" bool gen2_q8_tile_candidate_sgemm(const ggml_compute_params *, int64_t, int64_t, int64_t,
    const void *, int64_t, const void *, int64_t, void *, int64_t, int, int, int);
using Kernel = decltype(&llamafile_sgemm);
static_assert(std::is_same<Kernel, decltype(&gen2_q8_tile_control_sgemm)>::value, "exact original ABI");
static_assert(std::is_same<Kernel, decltype(&gen2_q8_tile_candidate_sgemm)>::value, "exact original ABI");
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "binary32 control");
static_assert(sizeof(block_q8_0) == 34, "original Q8 block layout");

extern "C" void gen2_q8_scale_broadcast_probe(uint16_t, float *);
static std::array<float, 65536 * 4> scale_half_outputs;

// Original CPU lookup symbol and initialization formula; retained unchanged.
extern "C" { float ggml_table_f32_f16[1 << 16]; }

[[noreturn]] static void reject_support(const char * name, unsigned code) {
    std::fprintf(stderr, "GEN2_Q8_UNSUPPORTED_%s\n", name);
    std::fflush(stderr);
    ExitProcess(code);
}
// Complete SGEMM retains unused F32 references. Reject use without C exceptions.
extern "C" void ggml_barrier(ggml_threadpool *) { reject_support("barrier", 73); }
extern "C" void ggml_threadpool_chunk_set(ggml_threadpool *, int) { reject_support("set", 74); }
extern "C" int ggml_threadpool_chunk_add(ggml_threadpool *, int) { reject_support("add", 75); }

static void save_output(const std::string & name, const void * data, DWORD bytes) {
    HANDLE file = CreateFileA(name.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("fresh kernel output required");
    DWORD written = 0;
    const BOOL wrote = WriteFile(file, data, bytes, &written, nullptr);
    const BOOL closed = CloseHandle(file);
    if (!wrote || !closed || written != bytes) throw std::runtime_error("complete kernel output write required");
}

static int initialize_half_table() {
    int finite_checked = 0;
    for (uint32_t i = 0; i < (1u << 16); ++i) {
        const float value = GGML_COMPUTE_FP16_TO_FP32(static_cast<ggml_fp16_t>(i));
        ggml_table_f32_f16[i] = value;
        float * converted = scale_half_outputs.data() + i * 4;
        gen2_q8_scale_broadcast_probe(static_cast<uint16_t>(i), converted);
        for (int lane = 0; lane < 4; ++lane) {
            if (std::isnan(value) ? !std::isnan(converted[lane]) :
                    std::memcmp(converted + lane, &value, sizeof(float)) != 0)
                throw std::runtime_error("F16C A-scale broadcast differs from original half table");
        }
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t sign = (i & 0x8000u) << 16;
        uint32_t mantissa = i & 0x3ffu, exponent = (i >> 10) & 31u;
        if (exponent == 31) {
            if ((mantissa == 0 && bits != (sign | 0x7f800000u)) ||
                (mantissa != 0 && !std::isnan(value))) throw std::runtime_error("invalid special half conversion");
            continue;
        }
        uint32_t expected = sign;
        if (exponent != 0) expected |= ((exponent + 112u) << 23) | (mantissa << 13);
        else if (mantissa != 0) {
            int shift = 0;
            while ((mantissa & 0x400u) == 0) { mantissa <<= 1; ++shift; }
            expected |= static_cast<uint32_t>(113 - shift) << 23;
            expected |= (mantissa & 0x3ffu) << 13;
        }
        if (bits != expected) throw std::runtime_error("finite half table differs from IEEE binary16");
        ++finite_checked;
    }
    if (finite_checked != 63488) throw std::runtime_error("complete finite half coverage required");
    return finite_checked;
}

struct Shape { int m, n, k; };
constexpr std::array<Shape, 22> shapes = {{{1,2,32},{2,3,64},{5,7,96},{7,5,160},{512,13,2048},{512,292,2048},{512,535,2048},{2048,13,512},{2048,292,512},{2048,535,512},{1536,13,512},{1536,292,512},{1536,535,512},{512,13,512},{512,292,512},{512,535,512},{512,3,1024},{512,12,1024},{512,13,1024},{192,13,512},{192,292,512},{192,535,512}}};
constexpr int repeats = 8;
constexpr float guard = 0x1p80f;

using Clock = std::chrono::steady_clock;
struct alignas(64) WorkerInterval { int64_t start_ns = 0, end_ns = 0; };
struct Observation {
    bool enabled;
    Clock::time_point epoch;
    std::array<WorkerInterval, 4> workers{};
};

struct Work {
    Kernel kernel;
    const Shape * shape;
    const block_q8_0 * a;
    const block_q8_0 * b;
    float * c;
    int lda, ldb, ldc;
    Observation * observation;
    void operator()(int ith, int nth) const {
        ggml_compute_params params{};
        params.ith = ith;
        params.nth = nth;
        Clock::time_point start{};
        if (observation->enabled) start = Clock::now();
        const bool accepted = kernel(&params, shape->m, shape->n, shape->k / 32, a, lda, b, ldb, c, ldc,
                    GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F32);
        if (observation->enabled) {
            const auto end = Clock::now();
            observation->workers[ith] = {
                std::chrono::duration_cast<std::chrono::nanoseconds>(start - observation->epoch).count(),
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - observation->epoch).count()};
        }
        if (!accepted) {
            throw std::runtime_error("admitted Q8 kernel rejected dimensions");
        }
    }
};

class Pool {
    std::mutex mutex;
    std::condition_variable ready, complete;
    std::vector<std::thread> workers;
    const Work * task = nullptr;
    uint64_t generation = 0;
    int remaining = 0, started = 0;
    bool stopping = false, failed = false;
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        ready.notify_all();
        for (auto & worker : workers) { if (worker.joinable()) worker.join(); }
    }
public:
    explicit Pool(int count) {
        try {
            for (int ith = 0; ith < count; ++ith) {
                workers.emplace_back([this, ith, count] {
                    std::unique_lock<std::mutex> lock(mutex);
                    uint64_t observed = 0;
                    ++started;
                    complete.notify_one();
                    for (;;) {
                        ready.wait(lock, [&] { return stopping || generation != observed; });
                        if (stopping) return;
                        observed = generation;
                        const Work * current = task;
                        lock.unlock();
                        bool bad = false;
                        try { (*current)(ith, count); } catch (...) { bad = true; }
                        lock.lock();
                        failed = failed || bad;
                        if (--remaining == 0) complete.notify_one();
                    }
                });
            }
            std::unique_lock<std::mutex> lock(mutex);
            complete.wait(lock, [&] { return started == count; });
        } catch (...) { stop(); throw; }
    }
    ~Pool() { stop(); }
    Pool(const Pool &) = delete;
    Pool & operator=(const Pool &) = delete;
    void run(const Work & work) {
        std::unique_lock<std::mutex> lock(mutex);
        task = &work;
        remaining = static_cast<int>(workers.size());
        failed = false;
        ++generation;
        ready.notify_all();
        complete.wait(lock, [&] { return remaining == 0; });
        if (failed) throw std::runtime_error("worker kernel failure");
    }
    int size() const { return static_cast<int>(workers.size()); }
};

static uint32_t next(uint32_t & state) { state = state * 1664525u + 1013904223u; return state; }
static void fill(std::vector<block_q8_0> & blocks, uint32_t & state) {
    for (auto & block : blocks) {
        block.d = static_cast<ggml_fp16_t>(0x2800 + ((next(state) >> 8) % 0x1800));
        for (auto & value : block.qs) value = static_cast<int8_t>(static_cast<int>(next(state) % 255) - 127);
    }
}

static std::vector<float> output(const Shape & s, const std::vector<float> & c, int ldc) {
    if (c.size() != 32 + static_cast<size_t>(ldc) * s.n) throw std::runtime_error("C size changed");
    for (size_t i = 0; i < c.size(); ++i) {
        const bool logical = i >= 16 && i < c.size() - 16 && (i - 16) % ldc < static_cast<size_t>(s.m);
        if (!logical && c[i] != guard) throw std::runtime_error("output guard overwritten");
    }
    std::vector<float> values;
    bool nonzero = false;
    for (int col = 0; col < s.n; ++col) for (int row = 0; row < s.m; ++row) {
        const float value = c[16 + col * ldc + row];
        if (!std::isfinite(value) || value == guard) throw std::runtime_error("invalid logical output");
        nonzero = nonzero || value != 0;
        values.push_back(value);
    }
    if (!nonzero) throw std::runtime_error("zero-only output");
    return values;
}

int main(int argc, char ** argv) {
    try {
        if (argc == 3 && std::strcmp(argv[1], "--support-control") == 0) {
            if (std::strcmp(argv[2], "barrier") == 0) ggml_barrier(nullptr);
            else if (std::strcmp(argv[2], "set") == 0) ggml_threadpool_chunk_set(nullptr, 1);
            else if (std::strcmp(argv[2], "add") == 0) (void)ggml_threadpool_chunk_add(nullptr, 1);
            throw std::runtime_error("unsupported control name or rejected call returned");
        }
        if (argc != 4) throw std::runtime_error("telemetry mode, workers and fresh output prefix required");
        const std::string mode = argv[1];
        const bool enabled = mode == "on";
        const int threads = std::strcmp(argv[2], "1") == 0 ? 1 : std::strcmp(argv[2], "2") == 0 ? 2 :
                            std::strcmp(argv[2], "4") == 0 ? 4 : 0;
        const uint32_t endian = 1;
        if ((mode != "off" && mode != "on") || !threads || *reinterpret_cast<const unsigned char *>(&endian) != 1)
            throw std::runtime_error("fixed telemetry/workers/little-endian control required");
        const int finite_half_checked = initialize_half_table();
        save_output(std::string(argv[3]) + ".half.f32", ggml_table_f32_f16, sizeof(ggml_table_f32_f16));
        save_output(std::string(argv[3]) + ".scale-half.f32", scale_half_outputs.data(), static_cast<DWORD>(sizeof(scale_half_outputs)));
        Pool pool(threads);
        for (size_t index = 0; index < shapes.size(); ++index) {
            const auto & s = shapes[index];
            const int blocks = s.k / 32, lda = blocks + 2, ldb = blocks + 3, ldc = s.m + 3;
            std::vector<block_q8_0> a(static_cast<size_t>(lda) * s.m), b(static_cast<size_t>(ldb) * s.n);
            uint32_t state = 0xc0ffee11u ^ (static_cast<uint32_t>(index) * 0x9e3779b9u);
            fill(a, state); fill(b, state);
            const auto original_a = a, original_b = b;
            const auto check_inputs = [&] {
                if (std::memcmp(a.data(), original_a.data(), a.size() * sizeof(block_q8_0)) ||
                    std::memcmp(b.data(), original_b.data(), b.size() * sizeof(block_q8_0)))
                    throw std::runtime_error("kernel input changed");
            };
            std::vector<float> c(32 + static_cast<size_t>(ldc) * s.n, guard);
            std::vector<float> reference;
            for (int phase = 0; phase < 4; ++phase) {
                const bool candidate = phase == 1 || phase == 2;
                const char * variant = candidate ? "scale" : "control";
                Kernel kernel = candidate ? gen2_q8_tile_candidate_sgemm : gen2_q8_tile_control_sgemm;
                Observation observation{enabled, Clock::now(), {}};
                const Work work{kernel, &s, a.data(), b.data(), c.data() + 16, lda, ldb, ldc, &observation};
                std::fill(c.begin(), c.end(), guard);
                pool.run(work);
                check_inputs();
                auto actual = output(s, c, ldc);
                if (phase == 0) reference = actual;
                if (actual.size() != reference.size() ||
                    std::memcmp(actual.data(), reference.data(), reference.size() * sizeof(float)) != 0)
                    throw std::runtime_error("kernel warmup output changed");
                std::array<int64_t, repeats> times{};
                std::array<std::array<WorkerInterval, 4>, repeats> intervals{};
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    std::fill(c.begin(), c.end(), guard);
                    observation.workers = {};
                    const auto start = Clock::now();
                    observation.epoch = start;
                    pool.run(work);
                    times[repeat] = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
                    intervals[repeat] = observation.workers;
                    check_inputs();
                    actual = output(s, c, ldc);
                    if (times[repeat] <= 0 || actual.size() != reference.size() ||
                        std::memcmp(actual.data(), reference.data(), reference.size() * sizeof(float)) != 0)
                        throw std::runtime_error("repeated kernel output changed");
                    if (enabled) for (int ith = 0; ith < threads; ++ith) {
                        const auto & interval = intervals[repeat][ith];
                        if (interval.start_ns < 0 || interval.end_ns < interval.start_ns || interval.end_ns > times[repeat])
                            throw std::runtime_error("worker interval outside whole dispatch");
                    }
                }
                const std::string name = std::string(argv[3]) + "." + std::to_string(index) + "." + std::to_string(phase) + ".f32";
                save_output(name, actual.data(), static_cast<DWORD>(actual.size() * sizeof(float)));
                std::printf("{\"shape\":%zu,\"m\":%d,\"n\":%d,\"k\":%d,\"workers\":%d,\"workers_started\":%d,\"half_table_entries\":65536,\"finite_half_entries_checked\":%d,\"variant\":\"%s\",\"phase\":%d,\"telemetry_mode\":\"%s\",\"bytes\":%zu,\"warmups\":1,\"repetitions\":%d,\"timed_ns\":[",
                    index, s.m, s.n, s.k, threads, pool.size(), finite_half_checked, variant, phase, mode.c_str(), reference.size() * sizeof(float), repeats);
                for (int i = 0; i < repeats; ++i) std::printf("%s%lld", i ? "," : "", static_cast<long long>(times[i]));
                std::printf("],\"worker_intervals_ns\":[");
                if (enabled) for (int repeat = 0; repeat < repeats; ++repeat) {
                    std::printf("%s[", repeat ? "," : "");
                    for (int ith = 0; ith < threads; ++ith)
                        std::printf("%s{\"worker\":%d,\"start_ns\":%lld,\"end_ns\":%lld}", ith ? "," : "", ith,
                            static_cast<long long>(intervals[repeat][ith].start_ns), static_cast<long long>(intervals[repeat][ith].end_ns));
                    std::printf("]");
                }
                std::printf("]}\n");
            }
        }
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 2; }
}
