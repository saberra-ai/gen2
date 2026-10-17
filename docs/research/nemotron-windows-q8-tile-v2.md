# Windows Q8 tile harness correction

Status: implemented, not executed. This separately versioned experiment corrects
three standalone-harness defects from the [actual failed link](nemotron-windows-q8-tile-build-failure.md).
The original producer, protocol, source, failed artifact and collection remain
unchanged. This is a model-free experiment, not a production SDK change.

## Predictions and unchanged scope

1. Supplying the original CPU half-conversion table and initialization formula
   removes the observed `ggml_table_f32_f16` link failure. Independent checks must
   cover every finite binary16 value, signed zero, infinities and NaN classes.
2. Replacing throwing C-linkage support definitions with explicit process exits
   removes the `/EHsc` contract warnings. Each of the three negative modes must
   produce its exact nonzero exit and diagnostic, without reaching Q8 compute.
3. Explicitly disabling OpenMP removes the inherited `libomp.lib` link dependency.
   Cache, actual compiler/build commands and final PE dependency evidence must
   agree before any benchmark invocation.

The [original kernel hypotheses and preregistration](nemotron-windows-q8-tile-probe.md)
remain intact: the candidate changes only the Q8 Windows AVX2 dispatch to the
existing 2×2 arithmetic helper. Original and candidate SGEMM hashes remain
`e226380b05393302ba91b38d22718d9521700ee2cb7943f83ad316a7a9dcd597`
and `304359cacc2bf8dd2fcdd5e07812ebb4df3930d69101f7366bc6b2664b71f195`.
Neither kernel source nor its arithmetic, source materialization, input seed,
shapes, padded strides, guards, worker counts, warmup/repetition count or timing
predicate changes. Both variants use the same corrected harness and support.

## Construction and actual controls

Bind the full original protocol and 214-file failed-build receipt before work,
plus the new complete import closure, sources, workflow and this document. Rebuild
from the exact independently checked nineteen-patch/3,425-file canonical tree.
Use the original admitted Clang/linker, contraction off, original global and
Release flags, AVX2/FMA/F16C and `/EHsc`. Preserve the compiler cache and database
immediately after configuration, including when construction subsequently fails.
All compiler inputs and objects must be present and hash-bound. Preserve separate
kernel objects/disassemblies, executable and dependency evidence. CPU backends,
backend loading, shared libraries and OpenMP remain off; this standalone static
configuration does not establish a production build.

The harness defines the original `float ggml_table_f32_f16[65536]` symbol and uses
canonical `GGML_COMPUTE_FP16_TO_FP32` for initialization. Before creating workers,
it checks all 63,488 finite values against an independent integer conversion,
including signed zero and subnormals; it checks infinity signs and NaN classes.
It exclusively writes the complete 262,144-byte table for each fresh kernel
process. The receiver independently verifies every finite binary32 result using
Python binary16 decoding, checks special classes and requires all twelve saved
tables to be byte-identical. Table initialization/validation is outside timing.

Unsupported CPU-pool paths print only `GEN2_Q8_UNSUPPORTED_<name>` to stderr and
terminate with `ExitProcess`: barrier 73, chunk set 74, chunk add 75. Before each
kernel process, run three separate owned negative-control processes using the
same bound executable. Require empty stdout, the exact stderr line and exact
expected exit, with normal observed/reaped status and successful Job cleanup.
A zero exit, wrong code, extra/missing diagnostic or cleanup failure stops the
experiment. Accidental unsupported-path use by a benchmark process also fails.
These 36 fresh negative processes replace the invalid in-process exception
controls; output rows no longer claim those controls occurred inside that process.
Their startup is outside the measured kernel calls. Retain actual command and
case associations, including partial negative-control evidence on failure.

Run the original twelve kernel processes: workers 1/2/4, each in
control/tile2/tile2/control order, with three rejection controls immediately
before each. The seven original shapes and all 84 compact outputs must be
complete, finite, nonzero and byte-identical across workers/variants. Check inputs
and output guards after every call. Retain one warmup and every one of eight
positive integer nanosecond observations per shape, including harness dispatch.
On four workers, every primary shape (indices 4/5/6) must satisfy both original
predicates: `10*abs(last-first) <= first` and
`20*max(candidates) <= 19*min(controls)`. No outlier removal, averaging away drift,
model-shape claim or model RTF follows from this control.

Before native construction run all 15 compiler controls, seven original tile
controls, seven corrected table/support/vector controls and ten actual Windows
Job controls. Preserve successful and failed source trees, archive, patch reads,
logs and command statuses. Recheck frozen inputs, compiler aliases/binaries,
built objects and all earlier case/guard outputs between processes and finally.
Freeze this producer at first invocation; another correction requires another
version. A useful kernel still needs observed model paths and a separate full
SDK/model/native/public comparison. Original quality, performance, lifecycle,
package, offline, coexistence, desktop and signing gates remain; human word-time
annotation/review stays documented and deferred at the user's request.
