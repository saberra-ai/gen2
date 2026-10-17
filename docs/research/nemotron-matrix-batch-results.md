# Batched matrix observer: local native controls pass

The separately versioned [batched-emission control](nemotron-matrix-batch-control.md)
now passes on macOS arm64. All twelve fresh graph processes completed: workers
1/2/4, each off/on/on/off, four original shapes evaluated three times per process.
All 144 finite, nonzero vectors are byte-exact per shape, including comparison
with the original per-row observer's saved control outputs. All 168 enabled
worker/matrix records have the required direct-SGEMM, converted-SGEMM and two
fallback routes, stage attempts/rejections, conversion flags and RMS/multiply
fusion. One persistent pool per process and complete graph/worker coverage pass.

The [saved receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-matrix-batch-control/collection-qualification.json)
independently replays every trace and raw vector. All 24 configure/build,
dependency and graph commands exited zero with their owned process groups
reaped. The collector rechecks all 3,425 native source descriptors, the exact
single-file transform, cache and compile database, 28 compiler source/object
inputs, actual compilers, binary/library identities and aliases, frozen local
subjects, logs and every prior output. Native binaries stay in the cache;
retained source/build descriptors and raw outputs are in the receipt.

The resulting CPU source is 143,331 bytes, SHA-256
`2d0ea19dcc51f53ffc376f51ebd53aa5c575adeead0b29cf4bf1993a9ac58000`.
Only the original post-final-barrier matrix emission block changes. Arithmetic,
capture lifetime, graph scheduling, timers and barriers remain unchanged. The
actual C-emitter seam also passes identical-wire checks for 256 extreme-value
records, one-write counting, disabled/empty cases and injected allocation,
formatting, truncation and short-write failures. The full local evidence-tool
suite passes 463 tests with seven Windows-only skips (18.757 seconds).

These small native graphs establish correctness of the batching candidate.
They do not measure full-model observer overhead. Next, freeze a new Windows
SDK/model protocol with its own runtime identity and patch, preserving the exact
compiler/source/ABI/numerical checks, full coverage and original timing bounds.
The original Windows observer's failed overhead result remains unchanged. No
production performance, quality, attribution, desktop, package or release gate
is closed; human review remains documented and deferred, and main unchanged.
