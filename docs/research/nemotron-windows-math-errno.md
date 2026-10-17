# Windows math errno experiment

Status: preregistered, not executed. The preceding
[CPU operation profile](nemotron-windows-cpu-ops-results.md) admits complete
measurement and exact outputs but still fails the original real-time gate.
This experiment changes one compiler policy on unchanged native inference source.
It cannot close quality, public-runtime, lifecycle, package or desktop gates.

## Hypotheses and prediction

Ranked hypotheses before invocation:

1. Requiring library math calls to set `errno` inhibits useful optimization.
   Removing only this requirement will preserve every probability byte and make
   both candidates at least 5% faster than both surrounding controls.
2. Matrix kernels, conversion or work partitioning dominate independently of
   this policy. The candidate will not meet that speedup prediction; inspect
   those paths separately after retaining the result.
3. Runner variation is larger than the change. More than 10% drift between
   controls prevents a positive speedup interpretation, even if a candidate
   happens to be faster.

The actual saved Clang19.1.5 driver trace already contains `-O3` and loop/SLP
vectorization for `/O2 /Ob2`; merely requesting O3 is not a new optimization.
See the [exact driver observation](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-fp-contract-probe/clang-no-contract-driver.stderr.log).
It also contains `-fmath-errno`. Clang documents
[`-fno-math-errno`](https://clang.llvm.org/docs/UsersManual.html#cmdoption-f-no-math-errno)
as allowing transformations that need not set math `errno`. This is a narrower
policy than fast math. It is not evidence of any speedup or numerical acceptance
for this model; the full output comparison remains mandatory.

## Exact construction

Derive the controller mechanically from frozen
`windows_compiler_fp_diagnostic.py`, retaining original Job ownership/timeouts,
source reconstruction, compiler/probe/ABI/ISA/private-name/state/version audits,
asset locks and between-case/final rechecks. Use two fresh SDKs and only the
original exact Clang compiler/linker binaries; the identical original MSVC
compiler builds both native probes. The two SDK labels are `control` and
`no-errno`; these labels are not compiler names. Both use Clang and retain the
existing diagnostic four-worker identity, never the production identity.

Both SDKs use explicit `/clang:-ffp-contract=off`. Only the candidate adds
`/clang:-fno-math-errno` to the original C and CXX initializers. Do not override
platform or Release flags, ISA definitions, dependency revisions, source patches,
thread policy, graph/fusion behavior or runtime inputs. Keep the original
nineteen-patch GGML tree and all three Gen2 runtime patches; the CPU profiler's
extra patch is absent. Require independent 3425-file reconstruction and all97
actual compiler inputs. Every one of the97 commands must match its corresponding
saved baseline after normalizing only selected compiler alias, isolated source
root, the one contraction flag and, for the candidate, the one errno flag.
Require original compiler/linker hashes. Retain full flags and commands.

Before each SDK build, capture the exact Clang `-###` driver expansion with the
same optimization/ISA/contraction/errno flags. This invokes no compilation or
model. Require effective O3, Haswell AVX2 target, loop/SLP vectorization, contraction
off, control math-errno on and candidate math-errno off; reject unrelated unsafe
math settings. Bind both logs with the compiler evidence and recheck them during
all subsequent cases. Driver observation alone does not prove the model's
arithmetic behavior or which kernel is affected.

## Full model and decisions

On one actual Windows x64 AVX2/FMA host, run
`control / no-errno / no-errno / control` in four fresh processes. Every case uses
the original Q8 model, 60-second licensed meeting, exact960000-sample PCM, original
geometry, all58 pushes and exactly four attached/effective workers. The complete
6000x8 arrays and all27 turns must match the frozen reference. Require every
probability byte identical in addition to the original1e-6 comparisons. Save
results, raw vectors and worker observations before comparison; preserve any
failure without retries or weakened tolerances.

Report all integer microsecond times and RTFs. Control drift is accepted only
when `10 * abs(last_control - first_control) <= first_control`. The speedup
prediction requires `20 * max(candidate_times) <= 19 * min(control_times)` and
accepted drift. Record both candidates' strict `elapsed_micros < 60000000`
decision separately. A failed prediction is a valid diagnostic result, not a
release pass. Do not combine CPUs or instances from different workflow runs or
infer errno cost from the earlier profile's worker intervals.

The workflow first runs the existing15 compiler controls, eight new derivation/
flag/driver/full-vector/timing controls and ten actual Windows Job controls. Freeze
all producer/protocol/workflow inputs at first invocation. Preserve the original
producer and every successful/failed receipt. A positive result would still
require a separately admitted production build without diagnostic telemetry,
full native/public/lifecycle performance, packaging/offline/coexistence/receiver
checks and all original quality, deferred human-reference, desktop and signing
requirements before main promotion.
