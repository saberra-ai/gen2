# Windows Q8 2×4 control on captured model geometries

Status: implemented; actual Windows execution pending. The
[bounded full-model matrix observer](nemotron-windows-matrix-batch-results.md)
passes its original timing bounds and shows converted Q8 SGEMM occupying
98.61–98.63% of each worker's matrix kernel wall. This is a separately versioned
model-free kernel experiment. It preserves the earlier slower
[2×2 candidate](nemotron-windows-q8-tile-v2-results.md) and every original release
failure. It changes no production SDK or public interface.

## Hypotheses and exact candidate

Ranked hypotheses declared before execution: the existing 2×4 helper's different
operand reuse/register layout is beneficial at these geometries; the current
4×2 path is already preferable and the new helper regresses; or dispatch/wake
overhead and scheduling dominate small observations. The first predicts both
candidate processes at least 5% faster than both surrounding controls on every
primary shape. Timing alone does not identify a register-spill or reuse cause.

Require the exact canonical 162,829-byte SGEMM fixture, SHA-256
`e226380b05393302ba91b38d22718d9521700ee2cb7943f83ad316a7a9dcd597`.
Change only `tinyBLAS_Q0_AVX::matmul`: for Windows AVX2/F16C without AVX512,
Q8 A/Q8 B/F32 C and m≥2/n≥4, call the existing `gemmMx4<2>` on the even-row,
four-column interior. Use the original `mnpack` for both non-overlapping tail
regions. Leave other types/architectures/small cases unchanged. No arithmetic
helper, block layout, reduction order, padding or thread-count policy changes.
The full candidate is 163,565 bytes, SHA-256
`6b669285cd9772400aa22beb377f6944d90b428b67372606e8792321174d0ea4`.

The standalone candidate target defines `GEN2_Q8_MX4_EXPECTED=1`. Its source
compile guard fails if the actual Windows/AVX2/F16C/non-AVX512 condition is absent;
the producer requires this definition in the actual candidate compiler command.
The original control target and all common compiler options remain unchanged.

## Fixed shapes and harness

Before construction, re-admit both enabled model profiles and all four workers
from the exact saved matrix receipt. Require the same six converted geometries,
root counts, types and complete distinct-column sets across all eight worker
observations. Primary A shapes each have 2,604 roots per worker and 63 distinct
columns. Select the minimum, median and maximum of those distinct values:
13, 292 and 535, fixed before any kernel timing. Kernel `(m,n,k)` means A
`[k,m,1,1]` and B `[k,n,1,1]`; input values are deterministic synthetic Q8 blocks,
not exported model tensor payloads.

| Primary m | Primary k | n values |
| ---: | ---: | --- |
| 512 | 2048 | 13, 292, 535 |
| 2048 | 512 | 13, 292, 535 |
| 1536 | 512 | 13, 292, 535 |
| 512 | 512 | 13, 292, 535 |

Retain the original four tiny boundary shapes first: (1,2,32), (2,3,64),
(5,7,96), (7,5,160). Then the twelve primaries above, then the two secondary
captured geometries: m512/k1024 with n3/12/13, and m192/k512 with n13/292/535.
All 22 ordered shapes are mandatory. The eight-call timing predicates remain
visible for every shape; primary acceptance requires every index 4–15.

Derive the C++ harness by exact shape-array and variant-name substitutions from
the corrected v2 harness. Preserve its full half-table initialization, IEEE
checks, padded strides, deterministic seeds, immutable-input checks, output
guards after every call, exclusive output creation, persistent worker pool,
one warmup and all eight positive integer nanosecond measurements. Derive the
CMake project by changing its project name and adding the candidate compile
guard definition only. Arithmetic and both actual SGEMM translation units remain
separate with the original ABI aliases. Preserve the three fail-closed support
exits, exact codes/diagnostics and independently checked 65,536-entry half table.

## Construction, execution and decisions

Freeze the full preceding v2 protocol and all its inputs, the actual v2 result,
the complete actual bounded-model receipt and its exact producer protocol,
this specification/workflow, both fixtures, new import/test closure and exact
harness/project derivations. Reconstruct all nineteen patches from the pinned
pristine tree and admit all 3,425 source descriptors before compilation. Keep
the exact original Clang/linker binaries, contraction off, original global and
Release flags, AVX2/FMA/F16C, OpenMP/backend/CPU-backend/shared-library exclusions,
compiler input/object/alias/disassembly and final dependency evidence. Preserve
configuration evidence even if building fails. Recheck actual frozen source,
toolchain, binary/object and all earlier guard/case outputs between processes
and finally, with scoped Job cleanup on every command.

Run the original compiler, source/vector/timing, half-table/support and actual
owned-Job tests plus the new geometry/candidate controls before construction.
Run twelve fresh kernel processes: workers 1/2/4, each
control/mx4/mx4/control. Immediately before each, run the three separately
supervised support rejections. Require all 36 exact nonzero exits and clean
teardown; benchmark processes must exit zero. All 264 full, finite/nonzero output
vectors must be byte-identical per shape across modes/workers. All twelve full
half tables must match and independently validate every finite value, signed
zero, infinity signs and NaN classes. Keep partial evidence on any failure.

For each of the twelve four-worker primary shapes retain both original integer
predicates: `10*abs(last-first) <= first` and
`20*max(candidates) <= 19*min(controls)`. Every primary must pass; no outlier
removal, averaging across failed shapes or lowered bound is permitted. A failing
prediction is retained separately from construction/numerical completion.

A passing kernel control still needs a new full SDK/model comparison with all
histories/turns and original numerical/RTF gates, then fresh production/public/
lifecycle/package/offline/coexistence qualification. CPU names across CI jobs
do not establish a matched host. Original quality/attribution, human-word,
desktop/fresh-machine and signing gates remain open; human review stays
documented and deferred, and main remains unqualified.
