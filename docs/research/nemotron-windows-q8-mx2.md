# Fixed packed-B-scale Q8 2×2 comparison

Status: preregistration and implementation only. No Windows build, compiled
candidate vector, timing, model/SDK call or production change has occurred.

The [one-row inspection](nemotron-windows-q8-row-loop-inspection.md) finds zero
explicit vector stack operands and90 normalized instructions per eight outputs,
versus production192bytes/95instructions, despite its failed speed prediction.
The [A-scale conversion experiment](nemotron-windows-q8-scale-results.md) is
exact but also fails its speed and observer gates. Static counts alone do not
establish a performance cause. All previous failures remain retained.

## One distinct helper and its tradeoffs

The current production control is the canonical fixture transformed by
`q8_mx4_source.transform`. A private clone narrows its packed-B-scale Mx4
template to two columns, instantiated with two A rows. This differs from the
older [generic scalar-scale2×2 trial](nemotron-windows-q8-tile-v2-results.md),
which failed on its own original synthetic shapes. No old trial is repeated or
retroactively requalified.

Only the private helper's name, two-column extent/indexing/accumulator count,
B half-scale packing and omitted columns2/3 loads/contributions change.
The two surviving packed half values retain exactly their original low32bits;
the unused upper32bits become zero. The same `_mm_cvtph_ps` call converts them.
Each output retains its original A table lookup, scale multiplication, increasing
block order, signed dot product, explicit `madd` FMA and horizontal sum. The
original helpers remain byte-identical for every other dispatch and both tails.

The full region still requires `m>=2 && n>=4`, with `me=m/2*2` and `ne=n/4*4`.
Only its call changes to the private `gemmMx2<2>`. Both original `mnpack` tails
are untouched. The candidate has four source accumulators and two B vectors,
retains B reuse across two A rows, repeats A work across column tiles, doubles
tile count and changes worker ownership under the original contiguous duty
formula. Every output must still have exactly one owner.

Ranked falsifiable predictions:

1. Fewer live vectors with retained two-row B reuse may remove explicit vector
   stack operands and improve every required paired dispatch by at least5%.
   The separate static prediction is zero such operands in the candidate's
   unique four-FMA reduction loop; normalize two iterations for eight outputs.
2. Repeated A loads/conversion and tile work may outweigh that saving, producing
   an exact candidate that fails the original speed predicate.
3. Worker/pool scheduling may prevent a whole-dispatch improvement. Unchanged
   worker wall intervals and whole-dispatch observations retain these effects;
   neither wall timing nor static counts can assign a CPU/memory cause.

The static prediction is reported separately. It cannot replace a numerical,
speed, drift or observer requirement or establish a measured bottleneck.

## Frozen construction and decisions

Benchmark, metrics, CMake and producer are checked derivations of the original
same-process one-row harness. Only source/module/protocol/project/role selection
and compile guard change. There is no clock/support/observer/ABI/compiler edit.
Keep native083861c241061562b8ec29eb06d7f6cdc3af8775, all3,425 canonical GGML
files, nineteen patches, original Clang/linker/AVX2/FMA/F16C identities and
contraction-off flags. Retain complete source/command/object associations,
objects, disassemblies, dependency evidence and the actual executable.

Run twelve fresh processes: workers1/2/4, each off/on/on/off, one persistent
pool per process. All22 captured/boundary shapes execute
control/mx2/mx2/control, one warmup and eight timed calls per phase. Require
all1,056 full phase vectors byte-exact within/across processes, twelve complete
original half tables,36 support rejections and95 owned/reaped commands. Preserve
all padding/tail/input-immutability/repeat checks and every individual timing.

For each of twelve primary shapes in every four-worker process, both candidate
totals must be at least5% faster than both surrounding controls, with drift at
most10%. Separately, all48 primary/position observer comparisons must meet
the original5% overhead and10% drift bounds. Inclusive integer predicates stay
exact. No failed/unstable observation may be discarded or replaced by a smaller
worker count or non-primary result. Numerical/observer failure withholds the
corresponding interpretation, regardless of workflow success.

The protocol binds the actual row/A-scale result identities and complete small
static receipts; this producer does not replay their old vectors. A model-free
pass supports only a separately frozen full-model comparison. Native/public
SDK RTF, lifecycle, package/offline/coexistence, desktop/microphone/fresh-machine,
platform and signing/release qualification remain required. Original AMI DER
failures and user-deferred human word review remain open; main is unchanged.
