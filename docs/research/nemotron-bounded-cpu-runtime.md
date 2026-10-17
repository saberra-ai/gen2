# Available-CPU-bounded pooled runtime candidate

The [fixed M1 thread contrast](nemotron-cpu-thread-contrast.md) preserves all
outputs and observes RTF 0.181545/0.202791 at three threads, versus
1.98405/1.77253 at four threads on the same three-CPU virtual receiver. Implement
a separately identified native candidate before making a public performance
claim: `gen2-valid-mel-pool-cpu-v2`.

The candidate changes only CPU `Diarizer::backend_params`: request the smaller
of its existing default and `std::thread::hardware_concurrency()`. An unknown
zero count has a defined one-worker fallback. GPU requests retain their current
settings. Larger hosts retain the existing four-thread request. The owned-pool
implementation, private object layout, frame contract, weights, segmentation,
geometry and audio transport remain the same. Compile-time policy assertions
cover 0/1/2/3/4/14 reported CPUs and a requested count below available capacity.

The [new lock](../../tools/nemotron-diarization/runtime-valid-mel-pool-cpu.lock.json)
binds the complete original pooled parent and the exact additional patch/source
hashes. Its three patches are in canonical source-diff order. Profile admission
must reject changed parents, reordered/changed patches, missing source hashes,
and unrelated CMake/frame/model changes. Existing profile IDs and default
selection remain unchanged. Rust's exact version allowlist still rejects this
candidate; no public loader or release admission is claimed here.

The [construction protocol](../../tools/nemotron-diarization/macos-bounded-construction.protocol.json)
binds current local modules before import and the committed source/protocol
before preparation. Construct a new portable armv8.2-a+dotprod+fp16 SDK and
relocated package with all source/compiler/GGML/dependency/notice provenance,
native model-free state test and exact new C ABI version. Then require the
existing 101 missing-model constructor cleanup attempts, their immediate and
settled thread counts, bounded waits, strict runtime image audit, and opaque
SDK/package/control archives. Retain the established storage reserve, free-space
floors, process-group cleanup, failure logs and timeout protections. The new
controller keeps this frozen construction separate from historical producers.

No weights are acquired or admitted by construction. At preregistration the
new SDK has not been built, the new source has not been prepared by its
controller, and no new native/helper/model execution has occurred. Model-free
Python profile checks are separate from actual native behavior. Successful
construction would still require an explicitly frozen native/public numerical
and lifecycle qualification on the three-CPU M1 receiver, followed by long,
mixed-model, quality and desktop package/platform qualification under this
identity. Original 20% DER, human word gold, GUI/microphone, fresh-machine,
signing and release gates remain unchanged; main remains unchanged.

The [actual model-denied M1 construction](https://github.com/saberra-ai/gen2/actions/runs/39003880853)
now passes at `aa066573`. The [112-member collection](../../benches/results/nemotron-diarization/macos-arm64/2026-10-17-bounded-cpu-construction/collection-qualification.json)
independently verifies all 200 delivered files, 98 compiler units, the same
3,425-file GGML inventory/19 patches/pristine archive, and original versus
relocated library sections. SDK/runtime/control archives restore with exactly
84/123/2 inventory entries. The actual package reports the new exact version,
passes the model-free native state test, and loads precisely 92 private libraries
in the saved failed-load control. All 101 settled thread counts return to one;
immediate counts range from one to three. Storage floors and the physical reserve
remain satisfied and released. No model/frontend inference or native execution
occurs during independent collection. Native/public numerical parity, real-time
behavior, full lifecycle, long/mixed/quality/app/platform and release
qualification under this identity remain pending.

The [fixed C ABI qualification protocol](../../tools/nemotron-diarization/macos-bounded-native.protocol.json)
now binds construction artifact `12089814731` and pooled reference artifact
`12083481818`, including every ZIP member and the reference's original producer.
Fifty tool/data inputs and the workflow are frozen before local imports;
committed source/protocol admission precedes compilation. Derive the helper by
changing one exact version literal in the existing public C ABI probe. Compile
only public SDK headers and four pinned WAV/hash source inputs, retaining and
checking all non-system compiler dependencies. No private C++ manager or explicit
thread override enters this helper; parameter selection belongs to the SDK.

Eight model-denied preparation commands include compile/runtime isolation,
101 relocated missing-model failures and a strict 92-library audit. Only then
acquire the pinned weights/audio and run exactly two fresh processes: the first
twelve seconds, then the complete sixty-second fixture. Compare all 1,200/6,000
rows, eight channels and every final turn with the saved independent pooled C ABI
reference, preserving 1e-6 bounds and reporting byte equality. Require warm
compute/audio below one for both on the fixed three-CPU M1 VM. Preserve complete
process timings/RSS, all outcomes and arrays, storage/reserve/group cleanup,
source/reference/prior-result isolation and failure-stop behavior. No retries
or additional neural calls are planned. At this preregistration no new parity
helper has been compiled or executed and no bounded-runtime inference has run.
The Rust version allowlist and the remaining roadmap gates stay unchanged.

The [first qualification attempt](../../benches/results/nemotron-diarization/macos-arm64/2026-10-17-bounded-native-admission-failure/collection-qualification.json)
remains failed: its producer declaration omitted the artifact `name` and
`run_id` required by the delivery validator. It stopped with `KeyError('name')`
before publishing either delivery, compiling a helper or starting any command
or inference. Its eight-member receipt verifies both delivered files, failed
job metadata/log and released storage reserve. No numerical result exists.

The [V2 protocol](../../tools/nemotron-diarization/macos-bounded-native-v2.protocol.json)
adds those exact identity fields, retaining all SDK/model/helper/case/parity and
performance settings. A regression using the actual frozen declarations fails
on the original missing name and passes after correction. A new `--admit-only`
mode must verify both complete actual archives and their qualification/origin
records on the local host before the corrected M1 run. That mode returns before
SDK restoration, helper compilation, asset acquisition or neural admission.
Commit the V2 protocol before that admission check and retain its evidence;
the original failed protocol/receipt remains unchanged. This is a versioned
correction of a pre-inference manifest error, with no repeated neural calls or
relaxed numerical/performance gates.
