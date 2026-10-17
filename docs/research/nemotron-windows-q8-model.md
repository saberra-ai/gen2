# Full-model Windows control for the existing Q8 2×4 dispatch

Status: producer implemented; actual Windows execution pending. The
[standalone kernel](nemotron-windows-q8-mx4-results.md) and
[worker observer](nemotron-windows-q8-worker-timing-results.md) failed their
declared predictions. This distinct experiment measures the original full-model
native real-time requirement directly. It changes no production runtime lock,
SDK admission, public interface or earlier result.

## Ranked hypotheses and decisions

1. The existing dispatch reduces enough full-model work for both candidate runs
   to achieve native RTF strictly below one while preserving every output byte.
2. Its small/mixed kernel improvements are insufficient at model scale; at least
   one candidate still fails real time or the surrounding controls drift.
3. The integrated source/dispatch or numerical behavior differs from the
   standalone control; source admission, model-free graphs or complete model
   histories reject it before any performance interpretation.

Retain all four observations. Compute the unchanged integer predicates:
`10*abs(last-first) <= first` and
`20*max(candidates) <= 19*min(controls)`. The 5% speedup prediction and strict
real-time result are separate fields; neither replaces a failed earlier
microkernel/observer predicate. Both candidate native elapsed times must be
strictly below60,000,000 microseconds for the original60-second real-time check.
No timing retry, averaging across failures or selection of favorable runs.

## Exact sources and construction

Build two separate SDKs from the same pinned NeMo/llama revisions, bounded
valid-Mel/native-pool source and original nineteen GGML patches. Both use the
original admitted Clang/MSVC-ABI compiler, linker, explicit contraction off,
platform/Release options, portable AVX2/FMA/F16C, CPU-only backend exclusions
and four-worker policy. Preserve the actual MSVC probe compiler. Give control
and candidate distinct diagnostic version tags; neither is a production identity.

Control retains the original canonical SGEMM. Candidate appends one exact patch
which changes only that file: the already tested 2×4 dispatch and one leading
`GEN2_Q8_MX4_EXPECTED=1` definition. The definition activates the existing
compile guard; the candidate body following that line must be byte-identical to
the preceding standalone candidate. Original arithmetic helpers, reduction
order, tail partitioning, types and nonselected paths remain unchanged. No CPU,
matrix or worker-wall timing observer is added to GGML.

Independently replay nineteen/twenty patches and admit all3,425 materialized
source descriptors. Candidate differs at exactly SGEMM; control matches the
original complete map. Admit all97 compiler commands and every source/object
binding, actual cache/ISA flags, original compiler/linker fingerprints and
private DLL closure. Normalize only isolated source roots/compiler aliases,
one contraction flag and the exact diagnostic version tokens relative to the
original baseline. Retain actual SGEMM objects/disassemblies for each SDK and
require the source-level candidate compile guard to be active.

Preparation re-admits complete preceding receipts once. Execution freezes the
original full-SDK protocol/input closure, new source/import closure, prior
protocols/manifests and qualification records. Old duplicate phase vectors are
not new compiler/model inputs; their complete manifests remain pinned and the
old receipts remain unchanged. Every actual input consumed by this producer
must be frozen, and source/SDK/compiler/assets and earlier outputs are rechecked
between model processes and finally.

## Actual graph and model controls

Before model acquisition, build the same admitted model-free GGML graph control
against each installed SDK using the unchanged MSVC compiler. Run each SDK with
one/two/four workers, all four original direct/conversion/fallback graph shapes
and three repetitions: six fresh processes,72 complete finite/nonzero vectors.
Require every per-shape vector to match across SDKs/workers/repetitions, with
bound binary, source, libraries and raw logs retained.

Then run four fresh full60-second/58-push native processes in
control/candidate/candidate/control order on the same admitted AVX2/FMA host.
Acquire only the original pinned model/WAV assets. Require four actually
configured/attached workers, correct distinct SDK version and actual PCM hash,
all6,000 rows/48,000 probabilities/192,000 bytes, unchanged retained geometry,
finalization and turns. Compare each complete output to the frozen native
reference using original validators and additionally require byte identity.
The two probes may differ only in their expected diagnostic version literal.
Retain all original ABI, model-free state, source/provenance and owned Windows
Job cleanup checks, including partial evidence if any stage fails.

This is a native diagnostic. A positive result still needs production identity
and full native/public, lifecycle, package/offline/coexistence and supported
desktop qualification. It cannot close the failed meeting-quality target,
deferred human word-time review, Pnola GUI/microphone/playback/IPC, fresh-machine
or signing requirements. Main remains unchanged.

## Local preparation evidence

The eleven new model-free admission tests pass. They exercise the actual saved
97-command compiler baseline for both diagnostic identities, exact Git patch
application, original-series preservation, complete 3,425-file maps and compiler
input descriptors, full-model probability identity, graph-vector rejection and
integer timing boundaries. They supply no new native inference measurement.

The separate source-only reconstruction at
`/Users/victor/.cache/gen2-nemotron3-n0/windows-q8-model-source-admission-v2/qualification.json`
passes all22 owned commands with zero exit and no surviving process group. Two
fresh native/llama clones admit bounded inference and the Windows LF overlay,
then independently materialize all3,425 files for each variant. Control differs
at zero files; candidate differs only at
`ggml/src/ggml-cpu/llamafile/sgemm.cpp`. No compiler, SDK or model runs in this
preparation. Actual Windows compiler, graph, numerical and timing evidence is
still required before interpreting the candidate.

Independent review identified and corrected two input-binding gaps before any
Windows dispatch: all97 actual compiler objects are now copied with complete
source/command/object descriptors and rechecked between histories, and the
consumed compiler-FP collection/cache/database/compiler/admission/source-audit
payloads are explicitly frozen. Negative controls reject missing, duplicated,
outside-build or mutated compiler objects and changed retained copies. The
source-only reconstruction was repeated as v2 after these changes; its22
commands and both complete maps pass. The original v1 preparation is preserved.
