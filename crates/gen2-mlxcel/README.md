# gen2-mlxcel

[mlxcel](https://github.com/saberra-ai/mlxcel) as a `gen2` backend: fast MLX
inference on Apple silicon, registered through `gen2::advanced::plugin`.

```rust
let engine = gen2::Engine::builder()
    .model("/models/qwen3-0.6b-4bit")
    .backend(gen2_mlxcel::plugin())
    .build()?;
```

It claims any directory holding a `*.safetensors` file, the same rule `gen2`
applies for its own MLX backend.

## Why it is a separate crate

Neither `mlxcel` nor `mlxcel-core` has a crates.io release, and the registry
refuses a git dependency even when optional. So the published `gen2` carries
no reference to them; this crate is a workspace member with `publish = false`,
buildable by git and path consumers:

```sh
cargo build -p gen2-mlxcel                 # compiles MLX C++ through cxx; minutes, once
PIO_TEST_MLX_MODEL=/models/qwen3-0.6b-4bit cargo test -p gen2-mlxcel
```

## Do not link `backend-mlx` in the same binary

`gen2`'s `backend-mlx` feature (mlx-rs) and this crate (mlxcel-core's cxx
bindings) both link MLX C++. Enabling both in one binary duplicates the MLX
symbol surface. This crate replaces `backend-mlx` on the Mac path; nothing
enforces the rule at compile time now that the two no longer share a manifest.

## Bundled metallib

A packaged macOS app that ships its own `mlx.metallib` points the worker at it
with `PIO_MLX_METALLIB=/path/to/mlx.metallib` before the engine starts. Unset,
MLX uses the path baked in at build time.
