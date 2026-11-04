# Local image generation

`Runtime::load_image_model` loads Qwen-Image-2.1-Turbo as a typed image model.
The handle supports blocking generation, a progress iterator, cancellation,
explicit eviction, and automatic restoration after eviction. Chat and image
models share the same runtime memory budget and least-recently-used eviction
order. There is no conversation or token stream to manufacture for an image.

## Setup

Enable `backend-stable-diffusion`. It adds a dynamic library loader; Cargo
does not compile or download a native runtime. The currently qualified backend
is Apple silicon macOS with Metal. Other platforms report a typed backend
error. Chat-only builds do not enable this feature.

Build the ABI-2 library with Xcode's Metal toolchain, CMake and Ninja:

```sh
bash tools/build-image-backend.sh
```

This checks out stable-diffusion.cpp at
`68285df319457e2d79310b48513e4ba1bf8ebe9a` and builds
`target/stable-diffusion/build/bin/libgen2_image.dylib`. The dylib contains its
own static GGML and exports only six gen2 functions. The script verifies that
allowlist to keep its GGML private alongside llama.cpp. ABI-1 qualification
libraries from `tools/qwen-image-metal` are deliberately rejected by this API.
`GEN2_SD_SOURCE_DIR` overrides the checkout location;
`GEN2_IMAGE_BUILD_JOBS` controls native build concurrency (default 8).

The host supplies that library's installed path to `ImageLoadOptions::new`.
It must stay available for the runtime's lifetime, including later restoration
of evicted models. A host can package and sign the dylib with its application.

Download the components using the pinned helper (requires the `hf` CLI):

```sh
bash tools/qwen-image-metal/download.sh
```

The default location is `$HOME/models/qwen-image-2.1-turbo`; set
`GEN2_QWEN_MODELS_DIR` to choose another. All three files are required:

- `qwen-image-2.1-turbo-Q4_K_M.gguf` — denoiser.
- `Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf` — text encoder.
- `vae/qwen_image_2.1_vae_bf16.safetensors` — matching 2.1 VAE.

## Generate

```rust,no_run
use gen2::{Runtime, image::{ImageBundle, ImageLoadOptions}};

let runtime = Runtime::new()?;
let bundle = ImageBundle::qwen_image_2_1_turbo(
    "/models/qwen-image-2.1-turbo-Q4_K_M.gguf",
    "/models/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
    "/models/vae/qwen_image_2.1_vae_bf16.safetensors",
);
let model = runtime.load_image_model(
    bundle,
    ImageLoadOptions::new("/app/lib/libgen2_image.dylib"),
)?;
let response = model.generate("A red teapot beside a green plant")
    .size(1024, 1024)
    .seed(42)
    .run()?;
response.image.save("teapot.png")?;
# Ok::<(), gen2::Error>(())
```

The request defaults to 1024 × 1024 and seed 42. Turbo sampling is fixed to
eight Euler steps, CFG 1 and the checkpoint's sigma schedule. Dimensions must
be multiples of 32, each at most 2048, with at most 1,048,576 total pixels.
Prompts must be nonempty, contain no NUL, and fit the load profile's UTF-8 byte
limit (default 4096). Request validation happens before restoration or eviction.
Seeds must be nonnegative, so the reported seed always identifies the request.

`GeneratedImage` owns RGB/RGBA pixels in shared immutable storage. Cloning a
result is cheap. Image bytes remain valid after model eviction or runtime drop.
`width()`, `height()`, `channels()` and `bytes()` expose the raster directly;
`save` uses its destination file extension.

A runnable example is included:

```sh
cargo run --no-default-features --features backend-stable-diffusion \
  --example image_generation -- \
  target/stable-diffusion/build/bin/libgen2_image.dylib \
  "$HOME/models/qwen-image-2.1-turbo" \
  "A red teapot beside a green plant" teapot.png
```

## Progress and cancellation

```rust,no_run
# let model: gen2::image::ImageModel = todo!();
use gen2::image::ImageEvent;
let mut stream = model.generate("A lighthouse on a green hill").stream()?;
let cancel = stream.canceller(); // Clone + Send: give it to the UI.
for event in stream.by_ref() {
    match event? {
        ImageEvent::Progress(p) => println!("{}/{}", p.step, p.total_steps),
        ImageEvent::Completed(_) => {},
        _ => {},
    }
}
let response = stream.finish()?;
# Ok::<(), gen2::Error>(())
```

`ImageCanceller::cancel()` requests cooperative cancellation. An already
submitted GPU graph finishes before cancellation is observed at a progress
boundary; conditioning and VAE decoding can also take time. A cancelled job
returns error code `image_cancelled` and no partial image. A later request can
reuse the same context. Dropping a stream requests cancellation without joining
the native call on the caller's thread. `finish()` waits for actual completion
and releases the model's busy state before returning.

One request per image model is allowed. A second request or an explicit
eviction while busy returns `image_busy`. The native library's callbacks are
global, so image contexts share a process-wide execution gate. Queued requests
remain charged and pinned; cancellation while waiting for that gate is checked
without running another native job. Progress buffering is bounded, and a slow
event consumer does not block native progress callbacks. The iterator is
blocking; use it from a host worker thread or `spawn_blocking` in an async app.
Chat and image requests may overlap, but long image GPU graphs can delay chat
inference on the same GPU. Coexistence does not guarantee interactive latency.

## Memory admission and residency

`model.memory_plan()` exposes `weights_mb`, `working_mb` and `total_mb` in MiB.
The weights estimate includes the denoiser, encoder, VAE, and conversion/alignment
headroom. Working memory scales with the permitted pixel count and prompt-byte
limit and includes native caches and conditioning buffers. These are conservative
estimates; they do not measure RSS or impose a hard bound on driver allocations.
The native runner also receives the admitted envelope as its managed-memory budget.

The **entire peak envelope remains reserved while the context is resident**.
Native buffers may survive between calls, so completion and cancellation do not
silently remove them from the ledger. Lower `ImageLoadOptions::max_pixels` or
`max_prompt_bytes` before loading to reserve for a smaller workload. Increasing
either beyond this backend's supported limits is rejected.

`Runtime::residency()` and `models()` include image model IDs alongside chat
models. `Runtime::stats().image_reserved_mb` is the image subset of total
estimated residency; `active_image_jobs` counts running and waiting image jobs.
Admission uses the shared inference allocation, current system pressure and
process headroom. Explicit host budgets are configured with
`Runtime::builder().resident_memory_budget_mb(...)` and do not bypass the OS
pressure checks. If a request cannot fit, it fails with `image_resource_limit`.

Idle image contexts participate in the same LRU order as language models.
An active image job is pinned until the native function has returned. Memory is
credited back only after native context destruction completes:

```rust,no_run
# let runtime: gen2::Runtime = todo!();
# let model: gen2::image::ImageModel = todo!();
runtime.evict_image_model(&model)?;
assert!(!model.is_resident());
// The same handle restores the bundle under admission before it generates.
let result = model.generate("A teapot").run()?;
// Or restore explicitly without generating:
runtime.preload_image_model(&model)?;
# Ok::<(), gen2::Error>(())
```

The runtime owns registered contexts until eviction or runtime destruction, as
it does for language models. Component sizes and modification times are checked
again before restoration; changed files require loading a new model handle.

## Verification

Deterministic driver tests exercise shared LRU order, refusal before native
allocation, lazy reload, busy models, cancellation with in-flight resources,
stream abandonment, failure recovery, and output ownership. They run without
model downloads or native libraries:

```sh
cargo test --no-default-features --lib image::tests
```

The ignored integration test runs the actual public API, a 1024 × 1024 image,
concurrent llama.cpp session recall, native cancellation and retry, explicit
eviction, and generation through the same handle after lazy restoration:

```sh
export GEN2_IMAGE_LIBRARY="$PWD/target/stable-diffusion/build/bin/libgen2_image.dylib"
export GEN2_IMAGE_MODELS="$HOME/models/qwen-image-2.1-turbo"
export GEN2_IMAGE_LLAMA="$HOME/models/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf"
export GEN2_IMAGE_EVIDENCE_DIR="$PWD/target/stable-diffusion/live-$(date +%s)"
cargo test --no-default-features \
  --features backend-stable-diffusion,backend-llamacpp,metal \
  --test image_generation -- --ignored --nocapture --test-threads=1
```

This backend currently covers text-to-image on Apple silicon Metal. Image
editing, reference-image encoders, mobile and other accelerators need their own
backend implementation and qualification. The earlier low-level milestone
remains documented in [the original probe](../tools/qwen-image-metal/README.md).

## Verified integration — October 16, 2026

On an M4 Pro with 64 GiB RAM and macOS 26.3, the public API produced a
1024 × 1024 RGBA image in 194.81 s. Its SHA-256 matched the original probe
exactly: `f986fc53ca0ab558884d05893b3e66de08738d4ac4e06f3f0d1691de2fd519a2`.
Peak process RSS was 11.28 GiB. The admitted profile reserved 18,606 MiB:
12,462 MiB for components/headroom and 6,144 MiB for workspace. This deliberately
exceeds the measured usage in this single run.

Qwen3-0.6B stayed resident with all 29 layers on Metal and recalled the same
session word before, during and after generation. Its concurrent turn took
45.20 s in the 1024 run, versus 23 ms after generation. Long image graphs can
delay chat on a shared GPU; this release does not promise latency isolation.
Native cancellation, reuse after cancellation, explicit eviction, and lazy
restoration through the same image handle all passed. The final image
reservation was zero after eviction. No automatic eviction was needed in this
live run; deterministic tests separately exercise eviction between model types.

The final native bridge, including its model-family check, also passed a
256 × 256 run with cancellation/retry and lazy reload. That generation took
11.30 s and its concurrent chat turn took 133 ms. These single-run timings are
qualification evidence, not a benchmark or a performance guarantee.

Local evidence directories beneath `target/stable-diffusion/` are
`public-api-20261016/` and `public-api-final-smoke-20261016/`, each with images
and `report.json`. Logs, including `/usr/bin/time -l` output, are under
`target/qwen-image-metal/image-api-live.log` and `image-api-final-smoke.log`.
The combined backend unit/integration regression run passed 1,113 tests with
18 ignored; the ignored Metal test above was then executed explicitly. There
were 67 passing doctests (4 ignored). Clippy passed for the library, new example
and new integration test. Existing vendored llama binding warnings remain.
