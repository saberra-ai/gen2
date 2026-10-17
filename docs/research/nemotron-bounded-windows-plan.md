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

The next [session-time diagnostic](nemotron-windows-session-profile.md) reuses
the exact reviewed four-worker SDK and upstream `NEMO_SPEECH_TIMING` switch,
without rebuilding native code. Its fixed off/on/on/off receiver sequence must
preserve all outputs and complete frontend/session timing coverage before phase
interpretation. Actual profiling remains pending; no new runtime is qualified.

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
