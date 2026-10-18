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
