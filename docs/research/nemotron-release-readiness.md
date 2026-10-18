# Nemotron release readiness audit

The full roadmap remains incomplete. N0–N3 have verified Mac ARM64 CPU
implementation evidence; N4 attribution and N5 release qualification remain
open. Source `856ffffa04c3f5234d49f7f93a636fe716a38550` was audited after the
Windows CPU-counter result was independently replayed and pushed. This audit
preserves every original quality/performance requirement and the user's
deferral of human word review.

The [four-member audit receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-release-readiness/collection.json)
retains its stdlib-only source, pre-execution protocol and
[complete observations](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-release-readiness/audit.json).
All14 inputs were bound before parsing and rechecked afterward. This checks
retained qualification reports and observes current filesystem capacity. It
does not rerun their lower-level numerical/command replays or establish a new
runtime, model, GUI, VM or release pass.

| Remaining requirement | Verified present state | Next dependency |
| --- | --- | --- |
| Original ≤20% DER on every required slice | CPU AMI production33.2146%/43.8407%; Metal33.2235%/43.8229%. Original-NeMo shared41.5071%/50.0045% also fails. | Product-appropriate acoustic-boundary review, separate development and fresh validation; then a justified quality change. Completed context/segmentation contrasts selected no passing replacement. |
| Human word-time and attribution validation | Prepared HCRC reference has0 annotators/0 reviewed words; no human gold. | Actual annotation, independent review and adjudication. **Deferred by the user: “just document it for now.”** No repeat request, synthetic gold or acceptance waiver. |
| Physical Pnola GUI, microphone, playback and IPC timing | Same-host complete-app CLI recording/live checks pass; physical observations remain unqualified. | Accessible interactive test host and actual capture/display/playback evidence. |
| Fresh-machine complete app | Private transfer/authentication skipped. Local VM Data access failed before writes; operational guest and complete-app test remain unqualified. | Operational fresh receiver. The disposable VM route additionally needs original storage admission and qualified guest access/provisioning. |
| Windows production real time | Candidate native/public RTF1.063041/1.046525/1.062326/1.055805; functional/lifecycle checks pass, strict real-time/package prerequisites fail. | A new justified candidate and complete original native/public/platform/package checks. The later counter observer also fails both timing bounds and supplies no resource-share selection. |
| Independent unseen-meeting validation | Current corpus has documented training-exposure limits. No new representative unseen human-labeled set is established. | Separately frozen licensed recordings, recording-level training-exclusion evidence and independent labels. |
| Advertised platform release and public signing | Development packaging/offline/coexistence receipts exist; complete release and public signing/notarization remain unqualified. | Actual platform/backend delivery and public signing/notarization receipts. Same-host ad-hoc construction is separate. |

The fixed audit observed **5.51845GiB available**, below the existing VM start
floor120GiB and running floor80GiB. This is a point-in-time observation, not a
permanent claim. No reserve, clone, guest, disk read/hash or cleanup was attempted;
the floors remain unchanged. Do not delete retained model/evidence/VM data or
other projects to manufacture admission.

The100-model-reload soak is already complete. The bounded Mac public receipt
also reports a passing warm RTF0.605346, and all16 valid-Mel corpus native/public
pairs are byte-exact with Q8 degradation within2absolute DER points. Those are
useful completed checks; they do not close the failed absolute accuracy or
missing human/physical/deployment requirements. Repeating them would add no
evidence for those gaps.

The [complete corpus results](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-frozen-corpus/qualification.json)
identify the earlier valid-Mel runtime. They do not by
itself qualify the subsequent `gen2-valid-mel-pool-cpu-v2` profile across all
eight slices. The [separately frozen bounded-profile corpus check](nemotron-bounded-corpus.md)
now completes static input/package/oracle admission and all eight fresh CPU
native/public pairs. The independently replayed outputs are byte-exact and stay
within the original Q8 degradation bound. Six slices pass every quality gate;
both AMI channels still fail at production DER 33.214572%/43.840741%. The run
retains the original references, FP32 arrays and failed accuracy gate. This
closes the bounded-profile corpus identity gap for these CPU slices and leaves
every unmet requirement in the table above open.

The [quality investigation](nemotron-quality-next-diagnostics.md) already
contains original-NeMo, frontend, teacher-forced tail, prefix, larger-context
and segmentation results. Further policy selection needs acoustic review and
separate validation rather than selection from failed targets. The
[Windows counter result](nemotron-windows-thread-cpu-results.md) preserves the
negative timing evidence. No unchanged failed experiment should be rerun in
search of a pass.

Human review remains deferred, and skipped app transfer remains skipped. No
credential, Keychain, signing, host-settings or physical-access action was taken.
The goal stays incomplete and main stays unchanged until actual evidence closes
all original gates.

The subsequent [Windows checkout repair](nemotron-windows-checkout.md) now
passes all 13 feature-check jobs at `f421e960`. The four Windows jobs each verify
22 long tracked paths before compiling and testing. This repairs a repository
checkout regression caused by retained evidence paths; it supplies no new
model-quality, Windows real-time or release qualification. The unmet gates in
the table remain unchanged.

A distinct [graph-only CPU observer prototype](nemotron-windows-graph-cpu-observer.md)
now removes the prior per-region cycle instrumentation while preserving the
batched wall observer. Its source/adapter/parser controls pass in the local
648-test model-free suite (eight Windows-only skips). The real Windows API
control now executes successfully at `8db9c0e5`, and all thirteen feature CI
jobs pass in run39091355966. The raw row preserves a zero user CPU delta and
zero API errors. Actual GGML parity, observer timing bounds and production
performance remain unqualified. This
does not select an optimization or close the Windows real-time gate.

The subsequent [native graph-only diagnostic preparation](nemotron-windows-graph-cpu-profile.md)
gives the observer a separate runtime identity and exact patch/source/compiler
admission. Its 658-test local model-free suite passes (650 executed, eight
Windows-only skips). The manual-only workflow has not been dispatched; actual graph/model
parity, both observer bounds and production performance remain unqualified.

The [new source-only receiver preparation](nemotron-graph-cpu-source-admission.md)
preserves the original 8 GiB Linux admission and outer-owned child supervision
for the full 3,425-file composition check. Local controls pass in a 661-test
suite (653 executed, eight Windows-only skips). That preparation checkpoint
precedes the completed source result below. Native execution and its later
artifact/return route remain unqualified. This supplies no additional runtime,
performance or release qualification.

The [actual graph source-only result](nemotron-graph-cpu-source-results.md)
now passes complete 3,425-file composition and nineteen-command ownership
replay. All 3,049 bindings match, the outer group is gone, and every command
starts above the original 8 GiB floor. The `ce65591d` feature CI is also 13/13
successful. This closes source composition only; native execution, artifact
replay/return and all unmet release requirements remain open. Workflow
registration was still pending at that historical checkpoint.

The registration-only context `71a27e26` adds a one-message push job while
guarding native compute for explicit manual dispatch. Native tools, inference
and observer code remain unchanged. The source receiver now pins the updated
protocol. The [actual registration result](nemotron-graph-cpu-registration.md)
now confirms successful push run39094365683, active workflow388152309 and the
native job skipped with zero steps. The matching source-only run39094365629
reports success; its updated artifact still needs separate independent replay.
The native diagnostic remains undispatched and its artifact-return route
unqualified. Registration does not execute the SDK/model diagnostic or qualify
a release.

The [graph-only saved-data receiver preparation](nemotron-graph-cpu-receiver-preparation.md)
now supplies the matching source-derived admission, supervision and semantic
replay programs. Eight focused controls pass, including actual graph-only CPU
aggregation and closed publication checks. Native identities remain unset and
no artifact is consumed. Owned acquisition, bounded result return and
independent local replay remain required before native dispatch; the unmet
requirements in the table remain unchanged.

The [transport preparation](nemotron-graph-cpu-transport.md) now supplies owned
Linux acquisition/bundling and bounded local quarantine wiring, with fourteen
focused controls passing. Actual identities remain unset and no artifact is
consumed; independent local semantic replay remains required before native
dispatch. This supplies no actual model, timing or release result.

The [independent local graph replay preparation](nemotron-graph-cpu-local-replay.md)
now reuses the fixed reviewed raw validation blocks with graph-only CPU fields,
complete source/command proofs and seven passing focused controls. Actual native
and return identities remain unset, and no actual new artifact is replayed.
End-to-end execution and review remain required; the unmet gates are unchanged.

The local replay preparation at `65cd82cd540fee5ba99da725f2f89b51f16d5e32`
now passes all thirteen feature jobs in
[CI run 39097218996](https://github.com/saberra-ai/gen2/actions/runs/39097218996),
and its source reviews are complete. On 2026-10-18 the subsequent
`gh workflow run diarization-windows-graph-cpu-profile.yml --ref codex/nemotron3-n0`
request failed with HTTP 403: "Resource not accessible by personal access token."
A follow-up run listing still shows only registration run39094365683; no manual
native diagnostic was created. An authorized GitHub session must start the
prepared workflow on that branch before actual artifact acquisition/replay can
proceed. The current token cannot perform that action. Credentials and workflow
guards remain unchanged. This is separate from the skipped private-app transfer;
all original quality, human, desktop, fresh-machine and release gates stay open.
