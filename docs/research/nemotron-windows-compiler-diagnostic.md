# Windows native compiler comparison

Status: preregistered diagnostic, not executed. The prior
[session-time result](nemotron-windows-session-results.md) places about99.6% of
profiled time in graph compute/output, with material control drift. Fewer workers
already proved slower on their measured host. This next experiment tests whether
the supported upstream `clang-cl` build route offers a useful compiler/toolchain
improvement without widening the portable CPU requirement.

Ranked hypotheses and predictions:

1. MSVC code generation limits graph performance. A fresh clang-cl SDK from
   identical source runs faster than surrounding fresh MSVC controls while
   preserving every probability/turn at the original1e-6 bounds.
2. A few operators dominate. Operator timing would identify a concentrated cost;
   this compiler experiment cannot establish that attribution.
3. Worker synchronization/host contention dominates. Thread-time or scheduling
   observations would reveal substantial waiting; control drift also limits
   compiler causality. Native graph wall time alone does not distinguish it.
4. Costs are diffuse. Neither a compiler substitution nor one operator explains
   a substantial gain; more detailed investigation remains necessary.

This experiment directly tests only the first prediction. Support that prediction
only if both Clang observations are at least10% faster than both MSVC controls,
and absolute control drift relative to the first control is at most10%. Use
integer elapsed microseconds for these comparisons. Otherwise retain descriptive
times without claiming an isolated speedup. These are investigation thresholds,
not new acceptance gates. A useful candidate must
retain the original strict below-one performance, original quality, complete
numerical parity and all public/package/release requirements in later independent
qualification. No averaging, RTF allowance or selected faster observation can
qualify a release. Retain every individual observation and both controls.

Use the pinned NeMo and GGML revisions and all three bounded inference patches,
followed by the existing exact two-site four-worker diagnostic telemetry and
Windows LF materializer overlay. The inference source hashes are exactly the
previous admitted four-worker diagnostic variant. Both SDKs intentionally report
the existing experimental tag `nemo-speech-asr 0.2.0+gen2-diagnostic-cpu-threads-4`;
their separate compiler identities, paths, hashes and artifacts identify the
toolchains. This tag remains outside production Rust/packager admission.

Construct fresh isolated MSVC and clang-cl SDKs on one Windows2022 job. Reuse
the frozen worker builder, adapting only its guarded SDK compiler selections
and provenance compiler argument. The upstream build script already supports
`-Compiler msvc` and `-Compiler clang-cl`; no source patch implements compiler
selection. Both diagnostic probes use the same unchanged source and MSVC compiler
recipe, so this experiment changes the SDK compiler/toolchain rather than the
probe compiler. The two probes are separate builds, not claimed byte-identical.
Dependency construction continues through the same pinned vcpkg route. Default
compiler/linker behavior can differ; describe this as a compiler/toolchain
comparison, not a proven individual optimization or exact instruction comparison.

Before any model acquisition, both builds must pass exact source and independent
canonical19-patch GGML/compiler-input replay, `diar_state` CTest, private PE/import
checks and actual runtime version checks. Retain full CMake caches, compiler
databases and generated C/CXX compiler files. Require actual MSVC/Clang identity,
Clang's MSVC ABI, exact selected compiler paths and every compiler command bound
to that binary. CMake's full/8.3 path spellings must identify the same actual file
and hash; basename matching is insufficient. Record compiler identities/versions
and compiler/linker binary hashes. Require the original Release/Ninja/shared CPU configuration, actual
AVX2/FMA/F16C CPU definitions and portable ISA options, with native/AVX512/VNNI,
OpenMP, backend-DL and GPU paths disabled. Reject conflicting ISA observations.
Capture receiver CPU/OS and require actual AVX2/FMA support before model use.
Static PE observations do not establish runtime module closure or portability.

Remove ambient `CL`, `_CL_`, `CC`, `CXX`, `CFLAGS`, `CXXFLAGS`, `CPPFLAGS` and
`LDFLAGS` from compiler child mappings, case-insensitively. `NEMO_SPEECH_TIMING`,
`GH_TOKEN` and `GITHUB_TOKEN` are absent as well. These are copied process mappings;
saved host settings remain unchanged. Git LF and unrelated TTS LFS-skip policies
are also process-scoped. No host/tool installer or settings change is performed.

After both complete admissions, acquire the original locked Q8 model and60-second
AMI fixture. Run fresh processes in fixed **MSVC,clang-cl,clang-cl,MSVC** order,
with one repeat, CPU index-1, four configured/effective/attached workers and the
original9/4/0/264/264/222 geometry/58 pushes. Preserve all6000x8 probabilities and
27-turn numerical-reference behavior, exact960000-sample PCM identity, original
load/push/finish/elapsed metrics, commands and complete logs. Compare every case
to the retained four-worker numerical reference at unchanged1e-6 limits; its
cross-host RTF is never a matched timing control. Timing rows must remain absent.
Persist each case's result and full arrays before numerical comparison, so a
failed comparison retains actual outputs. Stop and retain a failed diagnostic
when admission or parity fails; do not silently substitute flags or thresholds.

Recheck frozen project inputs, SDKs, materialized compiler trees, selected
compiler/linker binaries/path aliases, compiler evidence and model assets before every
case and after the sequence. SDK builds have1800-second child timeouts; other
children600seconds. Preserve failure evidence as well as success. SDK/delivery
payloads remain diagnostic artifacts; they do not constitute a bounded production
package. No public Gen2 lifecycle, failed-load cleanup, offline/coexistence,
GUI/microphone, fresh-machine, human annotation, quality or signing gate closes.

Every timed command runs inside a Windows kill-on-close Job Object. An internal
Python worker waits for the command payload until its parent assigns it to the
job, so assignment failure cannot release a command outside supervision. All
descendants inherit job membership without breakaway permission. On timeout or
error, terminate the job, reap the worker and verify zero active job processes;
normal completion also requires zero active processes. Cleanup failures remain
explicitly unverified, never a successful terminal command. The workflow first
runs actual Windows controls for normal/nonzero exit, assignment failure and a
timeout after a real grandchild has spawned; the grandchild must never perform
its delayed write. These controls invoke no speech model or compiler.
