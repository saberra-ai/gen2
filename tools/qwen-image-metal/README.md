# Qwen-Image-2.1-Turbo Metal qualification

The supported Runtime API now lives in [the image generation guide](../../docs/image-generation.md).
This ABI-1 harness remains a record of the first low-level milestone; new
applications should use `Runtime::load_image_model` and the ABI-2 bridge.

Standalone first-milestone harness for gen2. It loads the Unsloth Q4_K_M
denoiser, Qwen3-VL-8B UD-Q4_K_XL text encoder and matching BF16 VAE through a
pinned stable-diffusion.cpp build, then generates a 1024 × 1024 PNG with seed
42, Euler, CFG 1 and the checkpoint's fixed eight-step sigma schedule.

Both inference runtimes run **inside the same process**. The image runtime is
a dylib containing its own statically linked GGML. A linker export allowlist
keeps its GGML and SD symbols private; the build verifies the six exported
bridge symbols with `nm`. The bridge hides upstream struct layouts behind a
small versioned C ABI and frees images through their allocating library.

The live probe requires the llama.cpp model to have layers on Metal. It keeps
the same model and conversation alive while it:

1. Establishes a secret word before loading the image runtime.
2. Eagerly loads the image components onto Metal, with auto-fit disabled.
3. Asks llama.cpp to recall the word after the first denoising step, on a Rust
   worker thread. The response must finish before the last denoising step.
4. Checks recall after generation while the image context is still loaded.
5. Frees the image context and closes the dylib, then checks recall again.

Every recall and residency assertion must pass. The probe also checks eight
steps, pixel dimensions, RGB/RGBA output, nonconstant pixels, and a decoded PNG.
Only then does it write `report.json` with timings, placement, overlap steps,
the process ID, and the PNG SHA-256. Use a fresh output directory for each run.

## Run

After building and downloading once, generate from your own prompt:

```sh
target/qwen-image-metal/cargo/debug/gen2-qwen-image-metal-probe \
  --prompt "A tiny lighthouse on a floating island above the clouds, watercolor"
```

This uses the current checkout's bridge and the default model locations below.
`--output DIR` selects a fresh directory for `image.png` and `report.json`.
The full positional form remains available for custom component locations.

Needs Apple silicon macOS, Xcode (including its Metal toolchain), CMake, Ninja,
Rust, and the Hugging Face `hf` CLI. Model downloads occupy about 10 GB; build
artifacts need additional disk space. Run from the gen2 repository root:

```sh
git submodule update --init --depth 1 vendor/llama-cpp-sys-2/llama.cpp
bash tools/qwen-image-metal/download.sh
bash tools/qwen-image-metal/build.sh
cargo build --locked --manifest-path tools/qwen-image-metal/Cargo.toml \
  --target-dir target/qwen-image-metal/cargo

target/qwen-image-metal/cargo/debug/gen2-qwen-image-metal-probe \
  target/qwen-image-metal/build/bin/libgen2_qwen_probe.dylib \
  "$HOME/models/qwen-image-2.1-turbo" \
  "$HOME/models/qwen3-0.6b-gguf/Qwen3-0.6B-Q4_K_M.gguf"
```

An optional fourth argument selects the output directory. The default is a
new timestamped directory beneath `target/qwen-image-metal/`. Capture native
placement logs and peak process RSS with macOS `/usr/bin/time -l` if desired.
`GEN2_QWEN_MODELS_DIR` changes the download directory;
`GEN2_QWEN_BUILD_JOBS` changes native build concurrency (default 8).

The upstream runtime and three model revisions are pinned in `build.sh` and
`download.sh`. `Cargo.lock` pins the harness's Rust dependencies. The gen2
dependency is the local repository, with only llama.cpp and Metal enabled.

## Scope

This is a qualification tool, not a public image-generation API. Image
allocations are outside gen2's runtime memory admission and eviction ledger.
There is one image job at a time; the native progress callback is global to
the image dylib. It does not implement editing, runtime downloads, image
eviction, cancellation, non-Apple platforms, or a long-running memory soak.

Sources: [Unsloth model instructions](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF/blob/5d1f9f759ea877ae556cda88dbec958b62ad3809/README.md),
[stable-diffusion.cpp model guide](https://github.com/leejet/stable-diffusion.cpp/blob/68285df319457e2d79310b48513e4ba1bf8ebe9a/docs/qwen_image_2.1.md).

## Verified result — October 16, 2026

Passed on an Apple M4 Pro (14 CPU cores, 64 GiB unified memory), macOS 26.3,
with gen2 core at `1bc9f5bb9ffb715aa144ff23bf360e77e4102a32` and the pinned
image runtime above. The llama.cpp submodule is
`847f447c310a3784d623a953e6f0bb07f04c0f27`; the image runtime's separate GGML
is `dc998771e700d49bbdf9a39cede22bb5b3311767`.

| Check | Observed result |
| --- | --- |
| Image | 1024 × 1024 RGBA PNG; red teapot and plant; visually inspected |
| Sampling | 8 Euler steps, CFG 1, fixed Turbo sigmas, seed 42 |
| Image bundle load | 9.40 s, eager Metal placement |
| Generation, including text conditioning and VAE decode | 210.41 s |
| Eight denoising steps | 192.22 s |
| Whole process wall time | 222.29 s |
| Peak process RSS (`/usr/bin/time -l`) | 12,025,888,768 bytes / 11.20 GiB |
| Peak process memory footprint (same tool) | 11,640,169,832 bytes / 10.84 GiB |
| Llama placement | Qwen3-0.6B Q4_K_M, 29/29 layers on MTL / Apple M4 Pro |
| Same session recall before image load | `MAPLE`, 82 ms |
| Recall during denoising | `MAPLE`, 30 ms, started and finished between steps 1 and 2 |
| Recall with image context still loaded | `MAPLE`, 57 ms |
| Recall after image context freed and dylib handle closed | `MAPLE`, 30 ms |
| Runtime evictions | 0 |

Local evidence is under `target/qwen-image-metal/`: the `milestone-20261016/`
directory contains `image.png` and `report.json`; `milestone-20261016.log`
contains native logs, step timings, session checks and process resource totals.
The PNG SHA-256 is
`f986fc53ca0ab558884d05893b3e66de08738d4ac4e06f3f0d1691de2fd519a2`.
PNG dimensions were independently checked with `sips`, and the saved hash was
checked against the report. The native export allowlist, Rust build, Clippy
(`--no-deps -- -D warnings`), rustfmt and shell syntax checks passed. Existing
vendored llama.cpp/binding compiler warnings remain; no gen2 core changes or
full gen2 test-suite run were needed for this standalone harness.

This qualifies one image and one small resident language model on this Mac.
It does not establish memory requirements for larger language models or
production concurrency. Process RSS is not a separate measurement of total
GPU or system memory. Native callbacks plus a second Rust thread establish
overlapping requests, rather than a claim of simultaneous GPU kernels.
