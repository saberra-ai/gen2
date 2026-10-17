# Explicit-LLAMAFILE matrix observer control

Status: implemented, native execution pending. This is a separate experiment
following the preserved [failed control](nemotron-matrix-path-control-failure.md).
The old control project and all earlier sources, receipts and failures remain
unchanged. This control changes the standalone CMake project to explicitly force
`GGML_LLAMAFILE=ON`; the graph harness, matrix observer, arithmetic and source map
remain identical. No production optimization or release qualification is claimed.

Before any graph process, require the actual cache to record LLAMAFILE on, native
tuning off, Metal/Accelerate/BLAS/OpenMP/backend loading off, and the known host's
ARM dot-product feature available with SVE and integer-matrix features absent.
Require the actual compile database to contain the CPU C source and LLAMAFILE
SGEMM source exactly once, each defining `GGML_USE_LLAMAFILE`, and retain their
actual source/object/compiler identities. Resolve objects from the actual `-o`
argument plus command working directory and require agreement with CMake's
build-relative `output` field. Record all compiler sources, dependency reports,
binary/dylib identities and aliases, build logs and complete source map.

The original observer derives the same 142,477-byte CPU source, SHA-256
`93551fb37e0ec9ced9c4f3f0adfe87c4d04a873aec8fb23338417296bbb3f8f2`,
from the admitted canonical nineteen-patch 3,425-file source tree. Only that
file changes in the copied native tree. Freeze the local tool import closure,
fixture, harness, new CMake project, controller and this specification before
compilation, and recheck actual source/build/subject/library inputs between
processes and afterward. Use a new build root and fresh exclusive outputs;
failed attempts remain preserved, without changing their expected outcomes.

Retain the original four graphs and predictions: F32/F32 64×64 by 64×32 uses
direct SGEMM (route2); Q8/F32 with the same dimensions rejects the first stage,
converts the input and uses second-stage SGEMM (route4). The corresponding
one-column graphs reject SGEMM and use vector-dot fallback (route5), with input
conversion only for Q8. Each process uses one persistent pool, evaluates every
graph three times and retains the known RMS normalization/multiply fusion.

Run twelve fresh graph processes: workers1/2/4, each off/on/on/off. Require all
144 finite, nonzero outputs to match byte-for-byte per shape across repeats,
observer modes and worker counts. Each enabled trace must cover exactly twelve
graphs, every worker and one matrix root per graph, with expected actual type,
shape, stage attempts/rejections, conversion flags and fusion. Keep per-worker
walls and scheduling counters separate. This local control does not establish
full-model dispatch, Windows support, observer overhead bounds or model RTF.

A passing local result is a prerequisite for a separately versioned Windows
SDK/model protocol retaining original compiler/source/ABI/numerical checks,
Windows controls before model acquisition, full output histories and original
drift/observer-overhead bounds. Quality, attribution, desktop, packaging and
signing gates remain open. Human annotation/review stays documented and deferred
at the user's request; main remains unqualified.
