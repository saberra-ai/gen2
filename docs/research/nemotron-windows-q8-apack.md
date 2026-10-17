# Q8 aligned A packing: fixed comparison

The [completed packed-B-scale2×2 comparison](nemotron-windows-q8-mx2-results.md)
has exact outputs and zero selected-loop vector stack operands, but fails all48
required speed comparisons. The earlier row and A-scale experiments also fail.
Preserve those results. A smaller static stack count is not a performance fix.

This distinct candidate retains production2×4 geometry and worker tile ownership.
Before its tile loop, each worker copies only the A rows touched by its assigned
tiles into separate contiguous quantized-payload and decoded-scale arrays. Each
32-byte payload is aligned; each scale uses the original `unhalf` operation.
The original packed B-scale conversion, ordered per-output block loop, scale
multiplication, integer dot product, FMA and horizontal reduction remain exact.
All original helper definitions, full-region bounds and both tails remain intact.

The code shows repeated A data loads and scale decoding across column tiles.
Packing can amortize that work and avoid unaligned packed-block payload loads.
This is an implementation hypothesis, not evidence of a measured bottleneck.
Extra copy/allocation/free work, cache footprint and duplicated boundary rows
between workers can outweigh the benefit, especially for small column counts.

## Bounded scratch and complete timing

Scratch uses36bytes per packed block:32 payload bytes and four scale bytes.
Overflow-safe bounds cap each worker allocation at64MiB. Empty worker ranges
return before allocation; unsupported scratch sizes, zero blocks and allocation
failure fall back to the unchanged original helper. Allocation uses32-byte
alignment and matching destruction via `_aligned_free`, following the
[Microsoft CRT contract](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/aligned-malloc?view=msvc-170).
Packing, allocation, fallback and destruction are all inside each measured
kernel dispatch, including warmup and every repetition. Nothing is cached
between calls or lifted outside timing. This bound is not a process RSS or
Gen2 reservation pass; full-model memory admission remains a later requirement.

## Actual transport controls required before timing

Every fresh benchmark process runs the same packing implementation on65,536
blocks: all binary16 scale inputs and all256 byte patterns at every payload
position. Two rows, a nonzero starting row and padded row strides exercise
indexing. Original source/padding and destination guards must remain exact.
Packed scale bits must match the original lookup; both complete output arrays
are saved. Receiver replay independently checks every payload byte and the
finite/infinity/NaN-class binary16 reference, plus byte equality with the original
half-table file. NaN payload equality is claimed only between the actual saved
lookup and packed transport, not as a hardware-conversion equivalence.

Ten actual bound controls include negative/zero inputs, the exact cap, the next
block, two-row boundary cases and `INT64_MAX` multiplication rejection. A local
portable C++ control compiles these exact packing/bound/control bodies with
scalar memcpy load/store shims; this tests indexing and bounds, not Windows AVX
generation. The frozen Windows benchmark executes the real AVX transport.

## Unchanged comparison and decisions

The checked benchmark/producer/metrics derivations retain all22 original shapes,
twelve fresh processes (1/2/4workers, off/on/on/off), one persistent pool per
process, control/apack/apack/control phases, one warmup and eight timed calls.
Compiler identity, AVX2/F16C ISA, contraction-off flags, clocks and observer
boundaries remain unchanged. The additional transport controls run before the
pool/timings and do not move actual per-dispatch packing outside the clocks.

Require all1,056 complete byte-exact phase vectors, twelve complete original
half tables, twelve complete packing output pairs,36 support rejections and95
owned/reaped commands. Bind canonical source, source derivation, dependencies,
compiler/linker, objects, executable, command argv/cwd and every saved output.

The prediction is that this packing amortization yields the original at-least5%
speed improvement, with at-most10% surrounding-control drift, for every one of
the twelve primary shapes in all four four-worker processes. The independent
observer gate retains at-most5% overhead and10% drift for all48 primary/phase
comparisons. Candidate repeat drift remains reported. A failure remains failed;
one/two-worker, non-primary, static or workflow results cannot replace it.

This preregistration selects one packing design, not a parameter sweep. No actual
Windows construction/numerical/timing result, full-model RTF, SDK/public/lifecycle,
package/offline/coexistence, GUI/microphone, fresh-machine, platform or signing
qualification is established. Production/main remain unchanged. Original DER
failures and user-deferred human annotation/review remain open.
