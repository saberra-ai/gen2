# Graph CPU workflow registration result

The native diagnostic workflow is now registered and active. Push run
[39094365683](https://github.com/saberra-ai/gen2/actions/runs/39094365683)
at `0b2b7a7e9e81c83772d8059cc675cdb00a75e452` completed successfully:
registration job117577611570 emitted its single registration message, and
native job117577612370 was skipped with zero steps. The workflow API returns
active workflow388152309 at the expected path. No manual diagnostic was
dispatched and no compiler, SDK, graph or model ran in this registration run.

The [seven-file receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-graph-cpu-registration/collection.json)
retains the raw run/workflow records, registration log, local test log,
stdlib-only collector and [checked result](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-graph-cpu-registration/registration.json).
The collector binds six current files to the pushed source before fetching
metadata. Native protocol674377 bytes, SHA256
`581a25fee8e4cf5191bc9b5d22884cc9429b4d6e8ceada46e954a531f10484fa`,
matches native context `71a27e26a6285adf393f4b7d5862b21a5302efc4`.
Compared with the earlier `ce65591d` protocol, only workflow/document
descriptors changed; every other protocol field is equal. The native job is
byte-identical after removing its explicit manual-dispatch guard.

The registration-context local suite reports661 tests, eight Windows-only
skips, in50.100 seconds. The separate updated source-only run
[39094365629](https://github.com/saberra-ai/gen2/actions/runs/39094365629)
also reports success at the same source. Its artifact still needs a separate
capture and independent replay; the retained source-composition proof remains
the earlier context documented in [source results](nemotron-graph-cpu-source-results.md).
Feature run39094365701 was still in progress when this receipt was collected.
Its [terminal API status](https://github.com/saberra-ai/gen2/actions/runs/39094365701)
subsequently reports success for all13 jobs at the same source. This later
observation is separate from the registration-only receipt.

Registration closes the workflow-discovery prerequisite only. The native
diagnostic still needs its separately qualified artifact replay and bounded
return route before execution. Actual graph/model parity, both observer timing
bounds, production real time and all original N4/N5 release gates remain open.
Human word annotation and independent review remain deferred at the user's
request, and private app transfer remains skipped. Main remains unchanged.
