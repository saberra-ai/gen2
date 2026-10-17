# Bounded CPU Windows qualification plan

Status on 2026-10-17: actual bounded Windows MSVC construction, complete native/public
parity and all seven lifecycle tests pass; native/public RTF1.03335105/1.0005604167
fails the unchanged below-one performance gate. The SDK remains inadmissible for
bounded packaging. A separate [worker-budget diagnostic](nemotron-windows-worker-diagnostic.md)
preregisters the fixed4/2/1/4 same-host experiment. The first diagnostic fails
before compilation/inference on unrelated TTS Git-LFS smudging. The separately
versioned [checkout correction](nemotron-windows-worker-diagnostic-v2.md) passes
the actual Git-LFS regression; all359 model-free tests and all three source-only/LF
compositions pass. The [actual corrected matrix](nemotron-windows-worker-results.md)
now preserves a368-member result: four/two/one/four native RTF is
1.04734/1.29618/2.26025/1.05379 on one EPYC7763 host, with all four complete
histories byte-exact and27turns identical. All121 commands and all three compiler/SDK
audits pass. Fewer workers do not solve the performance failure; retain the
production policy and investigate computation/compiler behavior. No diagnostic
SDK is admitted as a production/public/offline package.

The separate [session-time diagnostic](nemotron-windows-session-profile.md) reuses
the exact reviewed four-worker SDK and upstream `NEMO_SPEECH_TIMING` switch,
without rebuilding native code. Its fixed off/on/on/off receiver sequence must
preserve all outputs and complete frontend/session timing coverage before phase
interpretation. The [actual result](nemotron-windows-session-results.md) now
passes those checks, retaining four byte-exact histories and both complete143-row
timing logs. About99.6% of profiled elapsed time lies in graph compute/output;
surrounding costs are small. All four native RTFs exceed1, and the final control
is34.15% slower than the first, limiting instrumentation/causal conclusions.
Investigate graph execution next without presuming a hot operator or compiler
cause. The34-member collection supplies diagnostic evidence; no new runtime,
public/lifecycle or package gate is qualified.

The separate [compiler comparison](nemotron-windows-compiler-diagnostic.md) now
preregisters fresh MSVC/clang-cl SDKs with identical admitted inference source,
the same MSVC probe source/compiler and an MSVC/Clang/Clang/MSVC process sequence.
Actual compiler/ABI/ISA/source admission precedes any model use; original full
output and performance criteria remain unchanged. The
[first actual job](nemotron-windows-compiler-job-failure.md) now passes all eight
compiler and five Windows Job controls, then fails at the first MSVC build's
live-descendant check. Its114-member collection retains all13 verified reaped
command trees,12 recorded zero codes and the unknown failed-command code/process
identities. No SDK installation, Clang build or model case is reached. A versioned
owned-background-process cleanup correction is required before the unchanged
experiment can complete; no compiler speed improvement is established.

The separate [versioned ownership correction](nemotron-windows-compiler-diagnostic-v2.md)
adds retained wrapped status and membership-scoped PID/image observations, drains
owned background descendants on normal exit and retains verified-empty cleanup
before returning status. Its Windows regression gates precede the original
compiler experiment. Every original inference,
portable ISA, numerical and production acceptance requirement stays unchanged.

The [actual corrected compiler result](nemotron-windows-compiler-results.md) now
passes both SDK admissions and all82 supervised zero-exit/reaped commands. It
stops at the first Clang probability comparison: max error0.27158,13975 values
above1e-6 and26 versus27 turns. Native RTF1.036746/0.990393 describes only the first
MSVC/Clang pair; the second candidate and closing control never execute. Preserve
the failure and investigate numerical divergence; no tolerance, performance or
production/package requirement is relaxed and no candidate is qualified.

The [scalar contraction probe](nemotron-windows-fp-contract-probe.md) next isolates
ordinary multiply-add, forced separate rounding and explicit FMA using the exact
two compiler binaries, before any new SDK/model execution. The
[actual scalar result](nemotron-windows-fp-contract-results.md) now confirms the
MSVC/Clang default contraction difference and restores separate rounding with
Clang contraction-off, with all three repetitions/explicit controls passing.
No SDK/model ran; model causality and every runtime/package gate remain open.

The separately versioned [full contraction diagnostic](nemotron-windows-compiler-fp.md)
now adds only the Clang contraction-off initializer, preserving platform defaults
and requiring every compiler command otherwise match its saved baseline. Both
SDK admissions precede the original full-output/fixed-order experiment. The
[actual full result](nemotron-windows-compiler-fp-results.md) now retains all four
byte-exact histories and 27 matching turns, both complete SDK/compiler admissions
and 84 zero-exit/reaped commands. Native RTF is 1.183428/1.067797/1.086572/1.190287;
control drift passes at 0.5796085%, but the 10% speedup prediction fails. This
isolates a useful numerical setting for this configuration while leaving the
below-one performance and all production/public/package requirements open.

The next [CPU graph profile experiment](nemotron-windows-cpu-ops-experiment.md)
now implements timing around existing fused-kernel and barrier boundaries in a
separately identified diagnostic SDK. It requires exact twenty-patch/3425-file
source admission, the original compiler/flags plus diagnostic metadata, actual
Windows model-free controls before asset acquisition, and four full-output
off/on/on/off cases. Local source composition and 415 Python tests pass. The
[actual measurement](nemotron-windows-cpu-ops-results.md) now passes all 59 commands,
twelve synthetic processes/36 identical vectors, four byte-exact model outputs
and both complete 84-graph/four-worker traces. Observer bounds pass with 0.2902212%
control drift; native RTF 1.013520/1.020798/1.015328/1.016462 still fails below one.
MUL_MAT, UNARY and FLASH_ATTN_EXT lead each separate worker's kernel wall totals.
Internal waits, scheduling, observer effects and overlapping workers limit
interpretation. The 292-file receipt qualifies measurement only; original
production/public/package and release gates remain open.

The next [math errno comparison](nemotron-windows-math-errno.md) implements a
single candidate compiler-policy change on the unchanged nineteen-patch tree.
Both fresh SDKs use the original Clang/linker and identical MSVC probe compiler,
explicit contraction off, exact97-command checks and original four-worker/58-push
geometry. Full outputs must be byte-identical. The driver expansion must show
the intended errno policy and existing effective O3. All 423 local tests pass
with seven Windows-only skips. The
[actual result](nemotron-windows-math-errno-results.md) passes 87 command cleanups,
both source/compiler/SDK audits, intended driver policies and four full byte-exact
histories. Native RTF 1.083064/1.023811/1.022393/1.022017 still fails below one.
Control drift 5.6365119% passes; the 5% speedup prediction fails, with both candidates
slower than the final control. The 326-file receipt supports no production fix or
release promotion. Matrix and detailed unary execution remain investigation targets.

The [Q8 tile kernel control](nemotron-windows-q8-tile-probe.md) now implements a
Q8-only Windows AVX2 2×2 dispatch candidate and guarded synthetic comparisons with
one/two/four workers. All 430 local tests pass with seven Windows-only skips.
The [first actual Windows attempt](nemotron-windows-q8-tile-build-failure.md) passes
all 32 Python/Job controls, exact source admission and 44 command cleanups, then
fails linking the standalone harness: the original half-conversion table is
missing. No benchmark case, SDK or model runs. Preserve that failure and version
the harness correction, including valid unsupported-path controls and explicit
OpenMP exclusion. This model-free experiment cannot establish model RTF, qualify
a production SDK or close the remaining package, quality, human-review or desktop
gates.

The new [producer](../../tools/nemotron-diarization/platform_bounded_windows_native.py),
[composed-source verifier](../../tools/nemotron-diarization/platform_bounded_sources.py)
and [workflow](../../.github/workflows/diarization-bounded-windows-native.yml) use a
separate [frozen protocol](../../tools/nemotron-diarization/bounded-windows-native.protocol.json)
binding42 tool files and264 repository files. They preserve the historical
producer and profile. The new producer applies all three ordered patches, checks
their original/post-patch hashes and exact four-file inference diff, prepares
private DLL names before the exact LF overlay, and rechecks the composed source
after construction and neural qualification. Independent19-patch GGML replay,
actual compiler-source auditing, state test, version query, PE/private imports,
complete6000-row native/public histories and all seven exact lifecycle tests
remain mandatory. Native and public observed RTF must each be strictly below1;
functional/lifecycle observations are retained separately if performance fails.
The workflow retains failed evidence and never automatically retries.
It derives the native probe by replacing exactly one quoted historical version
literal, retains original/derived source hashes and checks the derived bytes
before model admission and after qualification. The frozen probe stays unchanged.
The existing strict CTest validator admits both observed summary formats while
requiring exactly one actually passed `diar_state` case.

All333 model-free tools tests pass, including seven new tests for source composition,
missing edits, foreign/staged/submodule changes, exact identity, complete history
finite positive below-one performance and exact real-helper version derivation. The
[actual local source receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-windows-source/composition/qualification.json)
also admits a fresh local clone with the real three patches and exact five-file
composition after naming/LF preparation. That run performs no compiler, native
helper, version query, model or Windows API call; it cannot close Windows gates.
Both review axes cleared the producer before commit `dfdede8e74709a0c89be01d53aad92d346b50b99`.
Its first [Windows run](https://github.com/saberra-ai/gen2/actions/runs/39022551571)
has completed with a performance failure. The
[171-member receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-bounded-native-failure/collection-qualification.json)
verifies the exact artifact digest and66 SDK files. All47 child commands return0;
the enclosing qualifier exits1 at the final unchanged RTF assertion. MSVC build,
state test, exact version, private PE/imports, all6000 rows/48000 probabilities,
all27 turns and all seven lifecycle tests pass. Probabilities and turn values are
exactly equal. Public/native observed RTF is **1.0005604167 / 1.03335105**;
both fail the strict below-one requirement. Functional success does not qualify
this SDK, and bounded package admission rejects it.

Saved-data replay independently verifies42 tool/264 repository pins at `dfdede8`,
all three inference patches, derived helper bytes, all19 GGML patches against the
saved pristine archive, materialized tree and recorded actual compiler inputs.
It executes no delivered code or neural call. Downloaded binaries/ZIP and the
57MB pristine archive stay local. The recorded CPU exposes2 cores/4 logical
processors; the effective GGML worker count was not instrumented. Measuring a
smaller worker budget is a prospective diagnostic, not a proven scheduling cause
or fix. No timing allowance or automatic retry changes this result.
The first unadmitted collection pass is preserved locally. The replacement
admits producer and two additional evaluator sources using stdlib before local
imports, binds every unpacked input back to the verified ZIP, retains both192000B
raw vectors and their descriptors, then repeats saved-data replay. No inference
or Windows execution is repeated.

## Bounded package admission

The separate [SDK admission module](../../tools/nemotron-diarization/bounded_platform_sdk_receipt.py)
and [packager](../../tools/nemotron-diarization/package_bounded_windows.py) now
implement the bounded profile's development package contract. They reject older
profile/format identities, unsuccessful or slow producers, missing/reordered
patch operations, incomplete lifecycle commands, a changed native probe and
producer source/protocol mismatches. All original producer pins are verified at
the recorded commit; the feature test workflow fetches that exact source before
offline historical admission tests.

The new v3 bounded package retains all three inference patches, compiler audit,
original producer protocol/receipt, derived probe and complete native/public
6000-row outputs. Admission replays the numerical comparison, exact PCM identity
and strict below-one RTF requirement from retained values; pass flags alone do
not suffice. Package copying checks the added evidence before and after copying.
The central variant preserves the five-private-DLL-only contract and exact
external Microsoft prerequisite; existing PE/import, ISA/cache, CRT provenance,
source-notice, archive and relocation checks remain. Historical v2 tools and
receipts stay unchanged. No package is constructed in this step.

All343 model-free tests passed for that slice, including ten new SDK/package admission tests.
Their SDK/result records are explicitly synthetic controls, not actual Windows
runtime evidence. Actual bounded package construction and receiver execution,
own-process negative-load thread cleanup, central setup/offline System32 checks,
chat load orders and realtime results remain open. The old receiver orchestration
still binds the v1 producer and cannot qualify the new bounded package.

The [Windows own-process cleanup and versioned receivers](nemotron-windows-thread-cleanup.md)
are now implemented with350 passing model-free tests. The separate observer
control must demonstrate a known live and retired worker before the101-failure
probe can be used. The receiver removes model/reference inputs for that first
stage, retains source/binary/module provenance, then performs the original
offline/central/lifecycle/chat controls with the bounded comparator. None has
executed on Windows yet. A new orchestration protocol and a producer meeting
the unchanged performance gate are still required; this failed SDK is not admitted.

The [bounded lock](../../tools/nemotron-diarization/runtime-valid-mel-pool-cpu.lock.json) pins upstream `083861c241061562b8ec29eb06d7f6cdc3af8775`, llama `bd4f514db14d87fded667787a7a963bfbaa98e89`, and exact runtime `nemo-speech-asr 0.2.0+gen2-valid-mel-pool-cpu-v2`. Its ordered inference patches are valid-Mel finish, CPU thread cap, then persistent pool. [The cap](../../tools/nemotron-diarization/patches/cpu-thread-cap.patch) uses standard C++ concurrency: `min(existing 4, hardware_concurrency())`, falling back to one. [The pool](../../tools/nemotron-diarization/patches/persistent-cpu-pool.patch) uses GGML registry procedures and manager-owned cleanup. Neither patch contains Apple-specific calls; private DLL filenames do not change the registry procedure strings.

The [current Windows receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-current-valid-mel-offline-chat/qualification.json) passes functional/offline/chat controls with **older `gen2-valid-mel-v1`**, but records public/native RTF **1.06112135 / 1.05515815**: the below-one performance requirement remains unsatisfied by those observations. Its [producer binding](../../tools/nemotron-diarization/current-valid-mel-windows-runtime.protocol.json) identifies run **38940008466**, artifact **12065431951**, source **8fbef3e8a46a034b8e0ffa8f096409db683a923e**. These are historical references, not bounded artifacts. Bounded M1 [native](../../benches/results/nemotron-diarization/macos-arm64/2026-10-17-bounded-native/qualification.json) (`05c5ac1535a38e76b258ca57749bea820a7dc14f`) and [public](../../benches/results/nemotron-diarization/macos-arm64/2026-10-17-bounded-public/qualification.json) (`9ddefaf9c17806730cb2e12f1106c08102331b43`) passes support the candidate, not Windows parity or speed.

## Ordered implementation slices and gates

1. **Fresh construction and source admission.** Add versioned Windows bounded modules/protocol/workflow, adapting [the platform builder](../../tools/nemotron-diarization/platform_valid_mel_native.py) and [existing workflow](../../.github/workflows/diarization-valid-mel-native.yml). Apply all three locked inference patches; call `prepare_private_names` before the hash-guarded Windows LF materializer overlay. Preserve portable x64 flags, OpenMP/backend-DL/GPU exclusions, actual state test, version check, MSVC provenance and private PE/import checks. Do not relabel the existing SDK.

2. **Exact composed-source audit.** [pooled_runtime.verify_completed_source](../../tools/nemotron-diarization/pooled_runtime.py) rejects Windows; [the platform verifier](../../tools/nemotron-diarization/platform_valid_mel_sources.py) allows only the old pipeline patch plus LF overlay. A new verifier must bind exactly four inference files plus `cmake/llama_cpp.cmake`, locked before/after hashes, exact patches and pristine submodule. Retain `audit_cpu_tree` independent 19-patch GGML replay and actual compiler-input checks.

3. **Native/public lifecycle and performance.** Freeze bounded Windows probes, complete histories/turn parity, seven lifecycle cases and unchanged RTF acceptance before execution. [The failure-path probe](../../tools/nemotron-diarization/native_failed_load_probe.cpp) uses Mach APIs: implement a separately pinned Windows own-process thread/cleanup observation. Rust [version admission](../../src/diarization/ffi.rs) already accepts the exact bounded tag without a platform condition. No public thread knob is needed. On a host reporting at least four CPUs the cap remains four; pool reuse must earn its measured benefit.

4. **Fresh package and receiver qualification.** Version the original-profile bindings in [SDK receipt admission](../../tools/nemotron-diarization/platform_sdk_receipt.py), [packager](../../tools/nemotron-diarization/package_windows.py), [package qualifier](../../tools/nemotron-diarization/qualify_windows_package.py), [chat qualifier](../../tools/nemotron-diarization/qualify_windows_chat.py) and [consumer orchestration](../../tools/nemotron-diarization/qualify_current_valid_mel_windows_runtime.py). Retain all three patches in provenance and bind the fresh producer/inventory. Preserve five private DLLs, dependency/notices closure, publisher central-VC setup, offline System32 CRT controls and both chat load orders.

Original quality, human annotation, physical desktop, other platform and release gates remain open; this plan changes none of their criteria or historical evidence.
