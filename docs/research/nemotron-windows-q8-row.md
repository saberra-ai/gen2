# Fixed Q8 one-row dispatch comparison

Status: preregistration and implementation only. No Windows candidate build,
numerical run, timing observation, model/SDK call or production change has
occurred for this experiment. The original release gates remain unchanged.

The [actual production candidate](nemotron-windows-q8-candidate-results.md)
preserves native/public numerical parity but all four RTFs exceed one. The
[subsequent two-row consumption helper](nemotron-windows-q8-pair-results.md)
preserves all1,056 phase vectors, yet none of48 four-worker primary comparisons
meets the original speedup gate. Its explicit vector stack operand widths
increase192→256bytes. These are static operand counts, not measured traffic or
proof of a spill bottleneck. Every failed prediction stays retained.

## One fixed change and its tradeoffs

The exact current production control is `q8_mx4_source.transform` applied to
the canonical162,829-byte fixture. The candidate changes only its full-region
call from `gemmMx4<2>` to the already-existing `gemmMx4<1>` and the private
compilation-guard name. Removing those two substitutions reconstructs the
control byte-for-byte. No helper body or arithmetic is edited.

The source has four accumulator vectors rather than eight per tile. This may
reduce generated register pressure, but generated code and performance must
be measured. It also reloads B across A rows, repeats the packed B half-scale
conversion per row and doubles the number of full-region tiles. Those costs
may outweigh any benefit. This is a single fixed choice, not a sweep over
tile shapes or permission to retry unchanged timing until it passes.

The even-row/four-column full region remains unchanged, including the original
`m>=2 && n>=4` dispatch condition. Both tail calls remain original; an odd last
row still uses the original tail path. Full-region worker ownership changes
because each tile now covers one row, using the unchanged contiguous duty
formula. Every output still has exactly one owner. The original padded-output,
input-immutability, complete-vector and repeat checks must demonstrate actual
compiled correctness, including boundary shapes. Each output retains original
increasing block order, signed `updot`, ordered scale multiplication, `madd`
and `hsum`. Types, quantized layout, ISA and contraction policy stay unchanged.

## Frozen comparison and decisions

The producer, metrics, benchmark and CMake project are checked once-only
derivations of the paired-kernel experiment. Only source/module/protocol/project
selection, role `row1` and compilation guard differ. All clock placement,
support checks, compiler/linker identities, dependency retention, original
AVX2/FMA/F16C and contraction-off flags remain mandatory. The exact native
source remains commit083861c241061562b8ec29eb06d7f6cdc3af8775.

Run twelve fresh processes: workers1/2/4, each off/on/on/off. Each persistent
pool executes all22 captured/boundary shapes, control/row1/row1/control, one
warmup and eight timed calls per phase. Retain all1,056 full phase vectors,
twelve complete65,536-entry half tables,36 support rejections,95 owned/reaped
command records and every whole/worker timing. All vectors must be byte-exact
within and across processes. No unstable or failed observation is discarded.

For every one of12 primary shapes in each four-worker process, both candidates
must be at least5% faster than both surrounding controls and control drift
must be at most10%. Separately, all48 primary/position observer comparisons
must meet the original5% overhead and10% drift bounds. All inclusive integer
predicates remain unchanged. Non-primary and one/two-worker observations
cannot replace failed primaries. Numerical or observer failure withholds the
corresponding interpretation; workflow success alone is not a timing pass.

Retain both actual source/command/object associations, objects/disassemblies,
full source/compiler/dependency evidence and complete outputs. Any subsequent
static loop analysis must retain its unique function/loop identification and
avoid causal or measured-traffic claims. The protocol pins the prior failed
pair result/manifest identities and complete eight-member static receipt;
this producer does not replay those prior vectors or replace their result.

A model-free pass supports only a separately frozen full-model comparison.
Actual SDK, native/public parity, lifecycle, live RTF, package/offline/mixed,
desktop and platform qualification remain required. Original DER, deferred
human word boundaries/review, fresh-machine complete-app operation and public
signing/release/main gates remain open.
