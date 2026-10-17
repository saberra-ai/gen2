# Fixed Q8 A-scale conversion comparison

Status: preregistration and implementation only. No Windows build or candidate
execution has occurred. The production kernel and all prior failures remain.

The [one-row result](nemotron-windows-q8-row-results.md) is exact but fails all48
primary speed comparisons. Changing tile dimensions did not supply a qualifying
improvement. This experiment keeps production2x4 geometry and ownership fixed.

## Hypotheses and one change

The pinned x86 `simd-mappings.h` defines `GGML_CPU_FP16_TO_FP32` through the
65,536-entry `ggml_table_f32_f16` lookup. The existing Mx4 helper uses this path
for each A scale, while its four packed B scales already use `_mm_cvtph_ps`.
This source observation establishes a dependent lookup, not its runtime cost.

Ranked falsifiable hypotheses:

1. Removing the A-scale dependent table access may reduce dispatch time. Replace
   it with F16C conversion of the same unsigned16bits, followed by a lane-zero
   broadcast. Predict byte-exact finite arithmetic and the original5% paired
   dispatch improvement on every primary shape; a failure rejects that prediction.
2. Register allocation may outweigh the lookup saving. Inspect the actual
   generated reduction loop if needed; an increase in explicit stack operands
   contradicts a static stack-access reduction prediction. Counts alone do not
   prove CPU cost or measured memory traffic.
3. Worker/pool scheduling may dominate. The unchanged whole-dispatch and enabled
   worker intervals will show whether any observed worker-wall reduction reaches
   whole dispatches. They include preemption/stalls and cannot assign CPU causes.

`q8_scale_source.transform` starts from the exact current production
`q8_mx4_source.transform` control. It derives a private `gemmScaleMx4` clone of
the canonical Mx4 template; only its name and A-scale expression differ. The
full region calls that clone with RM2. Existing helpers remain byte-identical.
The shared candidate inline broadcast converts with `_mm_cvtph_ps`, then
`_mm_shuffle_ps(...,0)`. A candidate-only exported probe calls the same helper
and saves every lane for independent conversion checks.

Production2x4 tile geometry, contiguous job ownership, even-row/four-column
bounds and both original tail calls stay exact. B loads/conversion, per-output
increasing block order, signed integer dot product, scale multiplication,
explicit FMA and horizontal sum stay unchanged. No fast-math, reassociation,
ISA widening, compiler-policy change or production source promotion is included.

## Numerical and timing controls

The benchmark/metrics/CMake/producer are checked derivations of the preceding
one-row experiment. Changes are source/module/protocol/project/role selection
and the exhaustive A-scale helper control; clocks and timing decisions are
unchanged. Exact compiler, linker, dependencies, source/object/command bindings,
native083861c241061562b8ec29eb06d7f6cdc3af8775 and contraction-off flags remain.

Before timed work, each of twelve fresh processes invokes the actual candidate
helper on all65,536 half inputs, saves all four lanes (1,048,576bytes), and
checks the original CPU table. Independent saved-output replay must validate
all63,488 finite values bit-for-bit, signed zeros, subnormals, both infinities,
NaN classification and identical broadcast lanes. NaN payload preservation
between lookup and hardware is not claimed. This does not broaden the existing
finite output admission or replace complete matrix parity.

Run workers1/2/4, each off/on/on/off, with one persistent pool per process.
All22 shapes execute control/scale/scale/control, one warmup and eight timed
calls per phase. Retain all1,056 full vectors, twelve original half tables,
twelve four-lane helper tables,36 support rejections,95 owned/reaped commands,
both objects/disassemblies and every timing. All full matrix vectors must
match within/across processes; all original guards and input checks remain.

For every one of twelve primary shapes in each four-worker process, both
candidate totals must be at least5% faster than both surrounding controls,
with control drift at most10%. All48 primary/position observer comparisons
must independently meet the original5% overhead and10% drift bounds. Inclusive
integer decisions remain exact; no failed observation may be filtered or
replaced by a one/two-worker or non-primary result. Numerical/observer failure
withholds the corresponding interpretation even if the workflow succeeds.

The protocol binds original row result/manifest identities without replaying
those prior vectors. Prior failures remain retained. A model-free pass can
justify only a separate full-model comparison. Native/public SDK RTF, lifecycle,
package/offline/coexistence, GUI/microphone, fresh-machine, platform and public
signing/release qualification remain required. Original AMI DER failures and
the user-deferred human word-boundary review stay open; main is unchanged.
