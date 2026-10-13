# Bark CPU qualification evidence

The local [M4 Pro receipts](m4-pro/provenance.json) qualify implementation
`8277263fe2054e79f1ffb00bdc545e1d80cb7320` with ONNX Runtime 1.24.4,
CPUExecutionProvider, four intra-op threads and one inter-op thread.
Weights and audio are not committed; source/runtime hashes, pinned corpus IDs,
raw text and token decisions, human labels, quality slices and measured timings
are retained. The reproducible commands and limitations are in
[the speech guide](../../../tools/bark/README.md).

Local release gates pass: 250/250 raw transcripts and token sequences;
112/112 supplementary raw transcripts; numeric frontend/graph probes;
corrupt/offline provisioning; cancellation, bounded queues, shutdown/reload,
admission and both Laya initialization orders. The original Python ORT 1.23.2
comparison is limited to 25 clips, not full-corpus old-runtime qualification.
Malformed and old native libraries return errors without leaking reservations.

The matched warm CPU sweep passed at 1, 5, 15, 30 and 60 seconds. Five seconds
had a 0.182-second median; 60 seconds had a 2.29-second median. Sampled standalone
peak RSS was 3.09 GB. These are measured runs, not latency guarantees; occasional
host-load outliers are retained in the raw samples and p95 figures. The separate
mixed run kept the real Qwen3-0.6B GGUF chat and synthetic Laya smoke model resident
alongside speech using the merged native chat pin, completed a 100-request soak, preserved chat output and ended
with zero reservations. It does not qualify a real Laya checkpoint or GenAI
API-31/API-24 coexistence. [Workspace checks](m4-pro/workspace-checks.json)
also record default/no-backend/native test suites, lint, Intel Mac cross-check
and package-shape checks. Registry publication of the merged native-chat path
dependencies is not qualified; see [vendor notes](../../../vendor/README.md).

Resampling remains experimental. The pinned SoXR/PFFFT build matched all 54
non-16-kHz normalized sample fixtures and all 112 transcripts on this Mac.
Other compilers/CPUs/recordings can change greedy decisions. Human-label probes
retain the helper's known proper-name errors; implementation parity is not a
universal dictation-quality claim. Streaming/microphone capture, diarization,
word timestamps, accelerators, mobile and MLX remain unqualified follow-ups.

The [Linux CI receipts](linux-x86_64/provenance.json) qualify merged implementation
`8277263` in [run 38345540390](https://github.com/saberra-ai/gen2/actions/runs/38345540390):
250/250 raw texts/token sequences, 112/112 supplementary raw texts, numeric and
ABI/lifecycle/Laya gates pass. Linux CI does not qualify performance on that host.
The merged macOS and Windows jobs remain pending at publication; the user
requested pushing without waiting. Their completion is not claimed here.

The corrected pre-merge implementation passed all three operating systems in
[run 38343559265](https://github.com/saberra-ai/gen2/actions/runs/38343559265).
The retained [historical macOS arm64 receipts](macos-arm64-ci/provenance.json)
identify `d69eb52`, not the merged release, and must not be used as its proof.
