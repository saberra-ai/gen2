# Bounded CPU profile: original corpus requalification

The existing full-corpus results identify `gen2-valid-mel-v1`. They cannot by
themselves qualify the subsequent `gen2-valid-mel-pool-cpu-v2` runtime. This
separate [protocol](../../tools/nemotron-diarization/bounded-corpus.protocol.json)
keeps the original eight slices, exact cached official CPU FP32 arrays, audio,
references, UEM, scorers, thresholds and per-slice quality gates. No new recording,
threshold, postprocessing or oracle is selected. Every original failure remains.

The new CPU-only check uses eight fresh native/public pairs on the physical M4
Pro,16 neural processes in original slice order. All valid probability rows and
final turns must match the qualified bounded C ABI output, with additional exact
byte equality. Shared Q8 degradation must remain within2absolute DER points of
the original FP32 outputs; absolute shared and production DER must still be at
most20% on each slice. A quality failure is retained and does not truncate the
set. A runtime, identity, input, loader, cleanup or resource failure stops the
trial without automatic retry.

## Reuse and admission

Preparation copies the exact qualified bounded native/public/load helpers and
CPU package, the original corpus/oracle files and the unchanged Q8 model into a
separate input root. Five bounded file-copy commands complete and are reaped;
all177 staged filesystem entries match their source bytes, modes and aliases.
`cp -c` requests copy-on-write; this supplies no claim about physical allocation
or protection against privileged writers. No compiler, SDK/helper/version
function, model, download or conversion runs during preparation.

The [22-member preparation receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-corpus-preparation/collection.json)
retains all five command sidecars/logs, source/bootstrap, exact staged inventory
and the passing model-free suite:640 reported tests,7 skipped,48.940seconds.
The [qualification](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-corpus-preparation/collection-qualification.json)
explicitly leaves controller admission and all model calls unperformed.

Public helper reuse binds the original actual build record, all312 historical
source pins and its qualified binary. Native/public/load helper payloads also
match the completed bounded30m/1h helper archive manifest and qualification.
The staged package must pass static provenance/signature/dependency checks and
identify the exact bounded CPU profile. Source and input identities are checked
before local imports and between commands; helper compilation is not repeated.

First run `qualify_bounded_corpus.py --protocol ... --admit-only`, retain its
fresh `-admission` output, and independently inspect it before full execution.
Admission verifies the package, inputs, source/build/oracle provenance and
storage; it executes no native/helper/model function. Full execution uses a
different fresh output. All106 tools and312 repository source pins must match
the committed protocol/source before execution.

## Runtime and interpretation boundaries

The original corpus's5GiB start floor,1GiB running floor, physically allocated
256MiB reserve and1,200-second per-command process-group deadline are retained.
The separate bounded-public build/download8GiB floor and VM120/80GiB floors are
unchanged. This route reuses already qualified payloads; it does not repeat that
build/download controller under a smaller budget. The first focused test assumed
600seconds and failed against the actual original1,200-second field; only the
incorrect test/prose was corrected before any controller or model invocation.

Children have no network access or writes to inputs. Reads of original source,
SDK/cache roots, annotations, cached oracle arrays and prior result data are
denied. Actual denied-read/write controls and a model-denied version query
precede inference. Every native/public/version trace must pass the existing
strict private-image audit. Parent scoring uses the exact original Python3.12.13
and NumPy2.2.6 environment. No HF or original-NeMo model is rerun.

The metrics adapter admits the new runtime/CPU identity first, then applies the
unchanged valid-Mel scorer to copies with the old identity for that comparison
only. Retained inputs and reported identity stay bounded. Six focused tests cover
exact parity, a one-ULP mismatch below the old tolerance, unchanged absolute
quality failures, wrong identities/placement/oracle, all original inputs and the
original execution budget. These controls are separate from real-model evidence.

At preregistration the16 model calls, static-only controller admission and
independent result collection are unperformed. This check supplies no Metal,
M1-full-corpus, unseen-data, human-word, physical GUI/microphone, fresh-machine,
public signing or complete release claim. Human timing review remains deferred
at the user's request. Main promotion still requires every original gate.

## Execution checkpoint

Static admission completed under committed source
`d1954e86f936372cc2d481fb9da6da4c7bbf6ebe`. The independently checked
[admission receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-bounded-corpus-admission/collection-qualification.json)
contains no child commands, neural calls, scores or image audits. The physical
256 MiB reserve was allocated and released. Input admission passes; completion,
quality and release flags remain false.

The separate full controller is running from the same preregistered source and
protocol. Runtime denied-read/write/network controls and the model-denied version
probe have completed; the first native slice has started. No completed corpus or
quality result is claimed at this checkpoint. The original slices range from
114.944 to 515.904 seconds, so this is a longer run than the earlier short smoke
checks. The controller retains its original 1,200-second deadline for each
command and stops on runtime, identity or resource failure without automatic
retry.

The first result-collection draft was reviewed but never executed. Review found
incomplete command binding and a parse-before-inventory race. The retained V2
collector corrects both: it compares complete commands, environments, policy
bytes and denied-operation controls, captures identities before parsing, and
rechecks sources and copied evidence. Both independent reviews cleared V2
before its admission collection. Full-result collection remains pending.
