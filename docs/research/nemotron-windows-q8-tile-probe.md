# Windows Q8 tile kernel control

Status: preregistered, not executed. The [operation profile](nemotron-windows-cpu-ops-results.md)
places matrix multiplication first in per-worker kernel wall observations. The
[math errno comparison](nemotron-windows-math-errno-results.md) preserves outputs
but fails its speedup prediction and real-time requirement. This next experiment
builds a model-free kernel control before any new full SDK/model experiment.

## Source findings and hypotheses

The canonical nineteen-patch SGEMM file is 162829 bytes, SHA-256
`e226380b05393302ba91b38d22718d9521700ee2cb7943f83ad316a7a9dcd597`.
Its Q8×Q8 AVX2 path already uses llamafile SGEMM when the column count is at least
two. The 16-register dispatch normally selects a 4×2 tile. NeMo already requests
extra CPU buffer types, and this pinned repack implementation has no Q8 AVX2
selection; simply enabling the existing repack option cannot add that kernel.
These are source findings, not observations of each actual model tensor's path.

Ranked hypotheses before invocation:

1. A 2×2 Q8 tile reduces register pressure enough to improve the selected large
   synthetic matrices by at least 5% in both candidates versus both controls.
2. The existing 4×2 tile benefits more from input reuse; reducing the tile does
   not meet that prediction or makes the kernel slower.
3. Q8 tiling is insufficient to explain overall model time. Even a positive
   kernel result will need observed model paths/shapes and a full fixed-clip
   experiment before it can support a production optimization.

Retain both object disassemblies and symbols. Timing alone cannot establish a
register-spill explanation, and this experiment does not presume a hot tensor
shape or measure pure arithmetic cost.

## One candidate change

Construct the exact canonical 3425-file GGML tree from the pinned NeMo/llama
revisions and all nineteen original patches, checking the preceding complete
source audit. Preserve that tree. A separate candidate copy changes only the
AVX quantized class's `matmul` dispatch, gated to Windows AVX2 without AVX512 and
Q8×Q8→F32. It calls the existing `gemm<2,2>` on the even interior, then the original
`mnpack` on the remaining rows and columns. The arithmetic routines, per-output
block order, conversion/accumulation functions and remaining types stay intact.
The candidate is 163351 bytes, SHA-256
`304359cacc2bf8dd2fcdd5e07812ebb4df3930d69101f7366bc6b2664b71f195`.
A byte-identical upstream fixture preserves the original MIT notice and permits
portable transformation tests; actual compilation uses independently materialized
source and requires the fixture to match it exactly.

Build both translation units into one benchmark with separate C entry aliases.
Their internal implementations remain in the original anonymous namespace.
Use the exact admitted Clang/linker binaries, original platform/Release flags,
explicit contraction off and portable AVX2/FMA/F16C. Compile a static GGML base
with CPU backends and dynamic backend loading disabled for support functions.
No NeMo SDK or model is built/loaded. This standalone static configuration and
harness are distinct from production construction. Require all actual compiler
sources/objects, selected compiler aliases, cache/compiler files, tool hashes,
source/candidate bytes and final executable to be bound and unchanged.

The complete frozen SGEMM units retain F32 dispatch references to three CPU-pool
support functions. The harness supplies exact-signature definitions that throw
instead of performing that unsupported work. Each fresh process must demonstrate
all three rejection controls before Q8 compute and report that count. Any use of
one of those paths during the benchmark fails the worker. This keeps the frozen
SGEMM source intact without linking a production CPU backend into this control.

## Actual control and interpretation

Run twelve fresh processes: one/two/four persistent workers, each using
`control / tile2 / tile2 / control`. Require actual Windows x64 AVX2/FMA and at
least four logical processors before compute. Each process waits for all workers
to start, then evaluates these fixed `(m,n,k)` shapes:

`(1,2,32), (2,3,64), (5,7,96), (7,5,160), (512,33,512), (2048,33,512), (512,264,512)`.

These are synthetic coverage/performance shapes, not established model tensor
shapes. Inputs use deterministic legal Q8 integer values and exact half-scale
bits, with padded A/B strides. Output uses a padded leading dimension and guard
regions. Check input immutability, complete finite nonzero logical outputs and
all output guards after every call. Require one declared warmup and eight timed
calls per shape, all repeated outputs byte-identical within a process. Require
all 84 saved vectors byte-identical across variants and worker counts. The public
SGEMM ABI is checked against its original header at compile time. Output creation
is exclusive; any malformed result, write/build/worker/Job failure stops admission.

Save every positive integer nanosecond timing, including harness wake/dispatch
cost. Do not discard outliers. On the four-worker surrounding sequence, sum the
eight timed calls for each shape. Accepted control drift requires
`10*abs(last-first) <= first`; 5% speedup requires
`20*max(candidates) <= 19*min(controls)`. The kernel prediction passes only if
both predicates hold for every primary shape, indices 4/5/6. The smaller shapes
exercise correctness and tails; all their timings are also retained. A failed
prediction is a useful diagnostic result, not permission to relax criteria.

Run the existing fifteen compiler controls, seven new source/vector/timing
controls and ten actual Windows Job controls before construction. Preserve all
command statuses, owned-tree cleanup and successful/failed artifacts.
Retain the pristine archive, complete canonical source tree and individual patch
reads in those artifacts so a receiver can reconstruct and compare every file.
Freeze producer/protocol/workflow inputs at first invocation; corrections need new
versions. Recheck all fixed source/compiler/build inputs between processes and
afterward, including earlier logs and vectors.

This control establishes neither a model RTF nor numerical acceptance for a
changed model runtime. A useful kernel candidate still needs a separately
versioned full SDK/model experiment, the original native/public numerical,
quality/performance/lifecycle/package/offline/coexistence/desktop/signing gates,
and the deferred human word-reference review. Main remains unqualified.
