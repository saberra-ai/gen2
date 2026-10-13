# Pinned optional resampler source

Unmodified source from <https://github.com/dofuuz/soxr>, commit
`a66f3eeeeb62a32403ff143b756eed92b1ec6b62`, the libsoxr submodule of
Python-SoXR `v1.1.0` (release commit `9147978`). Only root build/notice files,
`src/`, and `cmake/` are included; examples, tests, and manuals are omitted.

The build matches Python-SoXR's static configuration: OpenMP and LSR bindings
disabled, constant-rate SIMD/PFFFT enabled when supported by the host compiler.
Only the HQ float32 mono oneshot operation is exposed by the private Rust ABI
adapter. No source or binaries are downloaded by the build.

libsoxr is LGPL-2.1-or-later; see `LICENCE` and per-file notices. PFFFT's
BSD-like license is retained in `src/pffft.c`. FFT4G and SIMD components also
retain their original source notices. The entire optional feature is covered
by the distribution cautions in `tools/bark/README.md`; it is off by default.
