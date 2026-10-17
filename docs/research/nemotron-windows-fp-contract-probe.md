# Scalar compiler contraction observation

Status: preregistered, not executed. The preceding
[compiler experiment](nemotron-windows-compiler-results.md) fails full numerical
parity. Saved arrays first exceed1e-6 at zero-based frame8/slot0:
0.9970491528511047 versus0.9970501661300659. Maximum error in the first72 frames
is9.775161743164062e-6; later error reaches0.2715801000595093. This is compatible
with early arithmetic differences growing through state, but does not prove it.

Ranked falsifiable hypotheses:

1. Implicit floating-point contraction introduces the initial difference. A
   minimal ordinary multiply-add will differ between exact MSVC/Clang binaries,
   and explicitly disabling Clang contraction will match separate rounding.
2. Compiler macros select different kernel implementations. Preprocessed relevant
   GGML branches will differ; matching those branches would remove the discrepancy.
3. Thread scheduling affects results. Repeated unchanged SDK runs will yield
   different raw arrays. This scalar experiment cannot test SDK repeatability.
4. An input/build setting escaped admission. Rechecking the saved source, compiler
   flags, assets and PCM would reveal a mismatch. Existing exact audits lower
   this hypothesis without asserting all compiler behavior is identical.

The next experiment tests only one low-level prediction of hypothesis1.
It cannot show that this difference causes the speech-model failure. No SDK is
constructed or executed, and no model is acquired or loaded. It changes no
runtime/lock/geometry/ISA/worker/output/performance acceptance requirement.

Use Windows2022, the exact two compiler binary hashes recorded in the failed
diagnostic, common `/O2 /Ob2 /DNDEBUG /std:c++17 /MD /arch:AVX2 /EHsc` options,
and one frozen C++ source. These are relevant scalar optimization/ISA options,
not a claim to reproduce the whole SDK build. Observe three fresh binaries:
MSVC-default, Clang-default, and Clang with only `/clang:-ffp-contract=off` added.
Retain driver logs, actual object disassembly, executable/object hashes and three
fresh-process observations per binary. CPU AVX2/FMA checks precede any execution.

The source uses runtime bit-pattern inputs `(1+2^-23),(1-2^-23),-1`. Three noinline
functions evaluate ordinary `a*b+c`, a volatile rounded product followed by the
addition, and explicit scalar FMA. The separate result must be zero; explicit
FMA must be exactly-2^-46 (`0xa8800000`). Preserve raw result bits and compiler
macros. Retain MXCSR; compare rounding/masking/FTZ/DAZ controls while excluding
sticky exception flags from that equality. The disabled-contraction variant
must produce zero for the ordinary expression. Unknown results/control failures
stop the observation rather than becoming a supported prediction.

Support the scalar prediction only if MSVC-default ordinary is zero,
Clang-default ordinary is-2^-46, and Clang-no-contract ordinary is zero, with
all explicit controls/repetitions passing. An unsupported prediction is a valid
diagnostic outcome. Neither outcome establishes model parity, speedup, a fix or
release qualification. A later full SDK/model test remains necessary before
claiming any model cause or candidate improvement.

The rationale comes from primary documentation: [MSVC floating-point options](https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior?view=msvc-170)
describe its default precise behavior and contraction changes in VS2022;
[Clang19 floating-point options](https://releases.llvm.org/19.1.0/tools/clang/docs/UsersManual.html#controlling-floating-point-behavior)
describe contraction controls, and its [clang-cl forwarding option](https://releases.llvm.org/19.1.0/tools/clang/docs/UsersManual.html#the-clang-option)
allows the explicit Clang flag. Generic driver documentation does not establish
the actual installed clang-cl default; that is what this observation measures.

All logged commands use the reviewed owned-Job helper with60-second timeouts.
Compiler-environment discovery runs in a bounded child; its full environment
output remains in temporary private handles and is never copied to the artifact.
Child mappings strip ambient compiler flags, `GH_TOKEN`/`GITHUB_TOKEN` and timing configuration.
No host settings/installers/credentials/signing state change. Freeze the complete
local import closure, source, reference compiler receipt, workflow and this plan
before invocation. Preserve failures and keep every original release gate open.
