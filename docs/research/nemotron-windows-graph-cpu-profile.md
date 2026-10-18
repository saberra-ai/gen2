# Windows graph-only CPU diagnostic protocol

The [graph-only adapter and actual Windows API control](nemotron-windows-graph-cpu-observer.md)
pass. This separately identified native diagnostic prepares the next test of
that observer inside real GGML. The previous detailed thread/cycle observer
failed both timing bounds; that result remains unchanged. The new design
removes its per-region cycle calls and ledgers while retaining the exact
batched wall observer and two graph CPU reads per enabled worker graph.
This contrast changes instrumentation coverage and its emission together;
it cannot isolate API-call cost or establish a production optimization.

The runtime reports `nemo-speech-asr 0.2.0+gen2-diagnostic-graph-cpu-v1`.
Its appended `gen2-graph-cpu-profile.patch` must materialize the exact output
of the previously tested graph-only source transform. The original nineteen
patches, arithmetic, fusion, evaluation, planning and barriers remain unchanged.
The worker, verifier and compiler adapters differ from the frozen CPU producer
only at their declared import sites. The original complete 3,425-file source
audit admits one changed CPU file; all 97 compiler commands must retain the
original compiler/linker, contraction-off flag and exact version-token delta.
Private PE naming/import/ABI/state controls remain mandatory.
The unchanged batched matrix emitter retains its checked 2,048-byte row budget;
the detailed observer's larger per-region CPU ledger is absent.

The native compute job is manual-only. A separate push-only registration job
prints one message and has no checkout, source acquisition, SDK or model step;
the native job is skipped on that event. This resolves initial branch-only
registration without promoting main. [GitHub's event documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_dispatch)
states that, after a workflow has run once, the API/CLI can dispatch it against
any branch or tag. Actual registration still requires a completed registration
run and a matching active workflow API record before native dispatch.
The native job's checkout retains LF and long-path child
Git configuration. Before model acquisition, twelve fresh actual Windows graph
processes run one/two/four workers in off/on/on/off order, preserving 144 complete
finite nonzero vectors and all original SGEMM/conversion/fallback/fusion checks.
Input and output identities, including retained build-only graph libraries,
are rechecked throughout. Every command retains the existing kill-on-close Job
supervision, timeout and process-tree cleanup evidence.

Only after those controls pass may four fresh native 60-second model processes
run off/on/on/off. Each retains the original 6000×8 complete probability array,
58 pushes and 27 turns. Both enabled traces must join all 84 graphs, four distinct
worker thread IDs per graph and 129 matrix roots, with exact repeated execution
signatures. API failures or counter regression reject interpretation; genuine
zero kernel/user 100 ns deltas remain valid. CPU durations and wall capture differ
in boundaries and granularity. They supply no per-operation CPU share, cycles,
exact descheduled time or pure arithmetic measurement.

Both original integer observer bounds remain required before interpretation:

- `10 * abs(final_control - first_control) <= first_control`.
- `10 * max(enabled1, enabled2) <= 11 * min(control0, control3)`.

The stdlib freezer binds the prior source/receipt closure, the actual Windows
API receipt, all new local imports, exact patch and workflow before execution.
The 658-test local model-free suite passes in 50.824 seconds (650 executed, eight
Windows-only skips). Ten focused producer controls include whole-file derivation
checks for all five adapters, original compiler/source mutation controls and
exact equality between the applied patch and the existing graph source transform.
This source preparation admits no local SDK build, model acquisition or model
run. The new diagnostic has not been dispatched. It needs separate execution,
retained raw artifacts and independent replay. Existing full native artifact
return requires the original 8 GiB floor; any separately bounded return still
requires its original 4 GiB admission. No floor is lowered, retained data deleted,
or VM disk read/hashed. The failed production Windows RTFs, AMI accuracy,
deferred human review, desktop/deployment and release gates remain open.
