# Matrix observer: first native control fails its route prediction

The matrix-path observer compiles and emits a complete fallback trace on macOS
arm64, but its planned native control **does not pass**. The standalone build
left `GGML_LLAMAFILE=OFF`; neither SGEMM stage was compiled into the CPU routine.
The first enabled process therefore reports vector-dot fallback for all four
shapes, contradicting the preregistered `[2,4,5,5]` route prediction. This result
does not validate SGEMM instrumentation or qualify Windows, model performance,
production integration or a release.

The [82-file receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-matrix-path-control-failure/collection-qualification.json)
preserves both failed controllers, the original build and execution admissions,
all logs, compiled subject snapshots, compiler database/cache, source/object/
compiler/runtime identities and all 24 actual output arrays. Native binaries and
the full source tree remain in the local development cache. The
[original control specification](nemotron-matrix-path-profile.md) is retained
byte-for-byte as the frozen pre-execution plan; its unexecuted status describes
that earlier point. This document records the subsequent actual result.

## Actual sequence

The original controller configured and built successfully, then failed while
resolving a compile-database object path. It incorrectly joined the database's
build-relative `output` field to the entry's working directory. No graph ran in
that attempt. Its qualification and traceback remain unchanged.

A separately reviewed execution controller admitted the same preserved build.
It resolved each actual compiler command's `-o` argument against that command's
working directory and required equality with the build-relative database output.
All 27 unique objects, compiler identities, dylib aliases, build logs, original
subjects and the complete 3,425-file source map were checked. No rebuild occurred.
The only canonical source change is the declared CPU observer file: 142,477 bytes,
SHA-256 `93551fb37e0ec9ced9c4f3f0adfe87c4d04a873aec8fb23338417296bbb3f8f2`.

Ten dependency inspections and two fresh one-worker graph processes completed
with exit zero and their owned process groups reaped. The graph processes ran
off, then on. Each evaluated four shapes three times, using one persistent pool:
F32/F32 with 32 columns, Q8/F32 with 32 columns, and each with one column. The
execution controller stopped at the first enabled route assertion; it did not
run the remaining enabled/control repetitions or the two/four-worker cases.

Saved-only collection independently admits all 12 enabled graph/worker matrix
records and the surrounding operation profile. Each graph has one matrix root
and the expected RMS normalization/multiply fusion. Every matrix route is 5
(vector-dot); both SGEMM attempt counters are zero. Q8 inputs alone require the
observed F32-to-Q8 conversion. The 32-column shapes have eight chunks and the
one-column shapes have one chunk. Worker wall intervals retain their original
meaning, including scheduling, internal waits and observer effects.

All 24 output arrays are finite, nonzero and byte-identical per shape across
the two modes and three repetitions. Arrays contain 8,192 bytes for each
32-column shape and 256 bytes for each one-column shape. This saved-only check
does not turn the aborted 144-output control into a pass.

## Diagnosis and next experiment

The actual CMake cache records `GGML_LLAMAFILE:BOOL=OFF`; none of the 27 compiler
commands defines `GGML_USE_LLAMAFILE`. The canonical CMake default is off, and
the control project omitted an explicit setting. These construction records
explain why the SGEMM call sites were absent. There is no evidence here about
the model's dispatch choices or whether any optimization would improve its RTF.

A separate control must explicitly enable LLAMAFILE, verify that setting and
its actual compiler inputs before graph execution, and retain the original
four-shape route predictions and off/on/on/off sequence at 1/2/4 workers. Give it
a new build/controller identity and fresh output directory. Preserve these
failures. A passing local control must precede the separately pinned Windows
SDK/model experiment; original numerical, observer-overhead and performance
gates remain unchanged.

The full Python suite passes 444 tests with seven Windows-only skips. The native
build retains two unreachable-break warnings in the CPU source; no warning-free
build claim is made. Human word annotation and independent review remain
documented and deferred at the user's request. N4/N5 and main promotion remain
open.
