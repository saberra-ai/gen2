#include "stable-diffusion.h"
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

struct image_output {
    uint32_t width, height, channels;
    const uint8_t *pixels;
    sd_image_t *owner;
    int count;
};
using progress_fn = bool (*)(int, int, float, void *);
struct progress_state { sd_ctx_t *ctx; progress_fn callback; void *user; bool cancelled = false; };
static void progress(int step, int steps, float time, void *user) {
    auto &state = *static_cast<progress_state *>(user);
    if (!state.callback(step, steps, time, state.user)) {
        state.cancelled = true;
        sd_cancel_generation(state.ctx, SD_CANCEL_ALL);
    }
}
static void fail(char *error, size_t capacity, const char *message) {
    if (error && capacity) std::snprintf(error, capacity, "%s", message);
}

#define EXPORT extern "C" __attribute__((visibility("default")))
EXPORT uint32_t gen2_image_abi() { return 2; }
EXPORT size_t gen2_image_devices(char *out, size_t capacity) { return sd_list_devices(out, capacity); }

EXPORT sd_ctx_t *gen2_image_load(const char *denoiser, const char *encoder, const char *vae,
                               uint64_t budget_mb, char *error, size_t capacity) {
    try {
        sd_ctx_params_t params;
        sd_ctx_params_init(&params);
        params.diffusion_model_path = denoiser;
        params.llm_path = encoder;
        params.vae_path = vae;
        params.backend = "metal";
        params.params_backend = "metal";
        params.auto_fit = false;
        params.eager_load = true;
        params.n_threads = 8;
        params.flash_attn = true;
        params.diffusion_flash_attn = true;
        const std::string budget = std::to_string(static_cast<double>(budget_mb) / 1024.0);
        params.max_vram = budget.c_str();
        auto *ctx = new_sd_ctx(&params);
        if (!ctx) fail(error, capacity, "could not load image components on Metal");
        if (ctx && std::strcmp(sd_get_model_version_name(ctx), "Qwen Image 2.1") != 0) {
            free_sd_ctx(ctx);
            fail(error, capacity, "this image profile requires Qwen Image 2.1 components");
            return nullptr;
        }
        return ctx;
    } catch (const std::exception &e) { fail(error, capacity, e.what()); }
    catch (...) { fail(error, capacity, "unknown image load exception"); }
    return nullptr;
}

// All calls are serialized by gen2's process-wide native gate, because SD's
// callbacks are global. The callback and its state never outlive this call.
EXPORT int gen2_image_generate(sd_ctx_t *ctx, const char *prompt, uint32_t width, uint32_t height,
                              int64_t seed, progress_fn callback, void *user,
                              image_output *out, char *error, size_t capacity) {
    *out = {};
    progress_state state{ctx, callback, user};
    sd_set_progress_callback(progress, &state);
    int status = 1;
    try {
        sd_img_gen_params_t params;
        sd_img_gen_params_init(&params);
        params.prompt = prompt;
        params.width = static_cast<int>(width);
        params.height = static_cast<int>(height);
        params.seed = seed;
        params.batch_count = 1;
        params.sample_params.sample_steps = 8;
        params.sample_params.sample_method = EULER_SAMPLE_METHOD;
        params.sample_params.guidance.txt_cfg = 1.0f;
        float sigmas[] = {1.f, .978453f, .95418f, .926626f, .89508f, .845148f, .704534f, .414568f, 0.f};
        params.sample_params.custom_sigmas = sigmas;
        params.sample_params.custom_sigmas_count = 9;
        const bool ok = generate_image(ctx, &params, &out->owner, &out->count);
        if (state.cancelled) status = 2;
        else if (ok && out->owner && out->count == 1 && out->owner[0].data) {
            const auto &image = out->owner[0];
            out->width = image.width;
            out->height = image.height;
            out->channels = image.channel;
            out->pixels = image.data;
            status = 0;
        } else fail(error, capacity, "native generation did not produce one image");
    } catch (const std::exception &e) { fail(error, capacity, e.what()); }
    catch (...) { fail(error, capacity, "unknown image generation exception"); }
    sd_set_progress_callback(nullptr, nullptr);
    if (status && out->owner) { free_sd_images(out->owner, out->count); *out = {}; }
    return status;
}

EXPORT void gen2_image_free_image(image_output *image) {
    if (image->owner) free_sd_images(image->owner, image->count);
    *image = {};
}
EXPORT void gen2_image_unload(sd_ctx_t *ctx) { free_sd_ctx(ctx); }
