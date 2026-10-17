// Model-free control for the diagnostic CPU loop: persistent workers and fusion.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#include <share.h>
#else
#include <unistd.h>
#endif

int main(int argc, char ** argv) {
    if (argc != 3 || (std::string(argv[1]) != "1" && std::string(argv[1]) != "2" && std::string(argv[1]) != "4")) {
        return 2;
    }
    const int threads = std::atoi(argv[1]);
    auto pool_params = ggml_threadpool_params_default(threads);
    std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(
        ggml_threadpool_new(&pool_params), ggml_threadpool_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_cpu_init(), ggml_backend_free);
    if (!pool || !backend) { return 3; }
    ggml_backend_cpu_set_n_threads(backend.get(), threads);
    ggml_backend_cpu_set_threadpool(backend.get(), pool.get());

    for (int shape = 0; shape < 4; ++shape) {
        const bool quantized = shape % 2 != 0;
        const int columns = shape < 2 ? 32 : 1;
        const ggml_init_params params = {1024 * 1024, nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        if (!ctx) { return 4; }
        auto * weights = ggml_new_tensor_2d(ctx.get(), quantized ? GGML_TYPE_Q8_0 : GGML_TYPE_F32, 64, 64);
        auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 64, columns);
        auto * scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
        auto * product = ggml_mul_mat(ctx.get(), weights, input);
        auto * norm = ggml_rms_norm(ctx.get(), product, 1e-5f);
        auto * output = ggml_mul(ctx.get(), norm, scale);
        ggml_set_output(output);
        auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
        ggml_build_forward_expand(graph, output);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
        if (!buffer) { return 5; }
        for (auto * tensor : {weights, input, scale}) {
            std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 16.0f;
            }
            if (tensor->type == GGML_TYPE_Q8_0) {
                std::vector<unsigned char> bytes(ggml_nbytes(tensor));
                if (ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), bytes.data(), 0, 64, 64, nullptr) != bytes.size()) return 9;
                ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
            } else ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
        }
        std::vector<float> values(static_cast<size_t>(ggml_nelements(output)));
        for (int repeat = 0; repeat < 3; ++repeat) {
            if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) { return 6; }
            ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
            const std::string path = std::string(argv[2]) + "." + std::to_string(shape) + "." + std::to_string(repeat) + ".f32";
            int fd = -1;
#ifdef _WIN32
            if (_sopen_s(&fd, path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _SH_DENYRW, _S_IREAD | _S_IWRITE)) return 7;
            FILE * file = _fdopen(fd, "wb");
            if (!file) _close(fd);
#else
            fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
            FILE * file = fd < 0 ? nullptr : fdopen(fd, "wb");
            if (!file && fd >= 0) close(fd);
#endif
            if (!file) { return 7; }
            const bool written = std::fwrite(values.data(), sizeof(float), values.size(), file) == values.size();
            const int closed = std::fclose(file);
            if (!written || closed != 0) { return 8; }
            std::printf("matrix-control threads=%d shape=%d quantized=%d columns=%d repeat=%d bytes=%zu\n", threads, shape, quantized, columns, repeat, values.size() * sizeof(float));
        }
    }
    return 0;
}
