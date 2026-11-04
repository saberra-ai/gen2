#include "stable-diffusion.h"

#include <cstdio>
#include <exception>

// ABI 1: only fixed-width numbers, C strings and opaque owned pointers cross
// this boundary. Upstream's changing struct layouts stay on the C++ side.
struct probe_image {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    const uint8_t *pixels;
    sd_image_t *owner;
    int count;
};

static void fail(char *error, size_t capacity, const char *message) {
    if (error && capacity) {
        std::snprintf(error, capacity, "%s", message);
    }
}

extern "C" {
__attribute__((visibility("default"))) uint32_t gen2_qwen_probe_abi() { return 1; }

__attribute__((visibility("default"))) size_t gen2_qwen_probe_devices(char *out, size_t capacity) {
    return sd_list_devices(out, capacity);
}

__attribute__((visibility("default"))) sd_ctx_t *gen2_qwen_probe_load(
    const char *diffusion, const char *vae, const char *encoder, char *error, size_t capacity) {
    try {
        sd_ctx_params_t params;
        sd_ctx_params_init(&params);
        params.diffusion_model_path = diffusion;
        params.vae_path = vae;
        params.llm_path = encoder;
        params.backend = "metal";
        params.params_backend = "metal";
        params.auto_fit = false;
        params.eager_load = true;
        params.n_threads = 8;
        params.flash_attn = true;
        params.diffusion_flash_attn = true;
        sd_ctx_t *ctx = new_sd_ctx(&params);
        if (!ctx) fail(error, capacity, "stable-diffusion.cpp failed to load the image bundle");
        return ctx;
    } catch (const std::exception &e) {
        fail(error, capacity, e.what());
    } catch (...) {
        fail(error, capacity, "unknown native exception while loading");
    }
    return nullptr;
}

__attribute__((visibility("default"))) bool gen2_qwen_probe_generate(
    sd_ctx_t *ctx, const char *prompt, int64_t seed,
    sd_progress_cb_t progress, void *user, probe_image *out, char *error, size_t capacity) {
    *out = {};
    sd_set_progress_callback(progress, user);
    bool ok = false;
    try {
        sd_img_gen_params_t params;
        sd_img_gen_params_init(&params);
        params.prompt = prompt;
        params.width = 1024;
        params.height = 1024;
        params.seed = seed;
        params.batch_count = 1;
        params.sample_params.sample_steps = 8;
        params.sample_params.sample_method = EULER_SAMPLE_METHOD;
        params.sample_params.guidance.txt_cfg = 1.0f;
        float sigmas[] = {1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f,
                          0.845148f, 0.704534f, 0.414568f, 0.0f};
        params.sample_params.custom_sigmas = sigmas;
        params.sample_params.custom_sigmas_count = 9;
        ok = generate_image(ctx, &params, &out->owner, &out->count);
        if (ok && out->owner && out->count == 1 && out->owner[0].data) {
            const auto &image = out->owner[0];
            out->width = image.width;
            out->height = image.height;
            out->channels = image.channel;
            out->pixels = image.data;
        } else {
            ok = false;
            fail(error, capacity, "stable-diffusion.cpp did not produce exactly one image");
        }
    } catch (const std::exception &e) {
        fail(error, capacity, e.what());
    } catch (...) {
        fail(error, capacity, "unknown native exception during generation");
    }
    sd_set_progress_callback(nullptr, nullptr);
    if (!ok && out->owner) {
        free_sd_images(out->owner, out->count);
        *out = {};
    }
    return ok;
}

__attribute__((visibility("default"))) void gen2_qwen_probe_free_image(probe_image *image) {
    if (image->owner) free_sd_images(image->owner, image->count);
    *image = {};
}

__attribute__((visibility("default"))) void gen2_qwen_probe_unload(sd_ctx_t *ctx) {
    free_sd_ctx(ctx);
}
}
