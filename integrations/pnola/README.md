# Pnola speaker integration

This overlay versions the tested changes to the separate Pnola desktop app.
Pnola's inspected local Git baseline is `81ae9ba`; it has no configured remote.
The app remains a separate project. Gen2 owns native execution and word
attribution; Pnola owns media conversion, history, speaker names and playback.

Check the current app before applying:

```sh
python3 integrations/pnola/apply.py --pnola-root /absolute/path/to/pnola \
  --gen2-root /absolute/path/to/gen2 --check
python3 integrations/pnola/apply.py --pnola-root /absolute/path/to/pnola \
  --gen2-root /absolute/path/to/gen2
```

The complete file set is validated before writing. Known base hashes and an
already-applied overlay are accepted; unrelated edits are rejected. Changed
files are backed up under `.nemotron-backups/` and individually published by
rename. The Cargo dependency points to the explicit Gen2 checkout. A repeated
application is a no-op; changing only that dependency path is supported.

The desktop app loads Bark and Nemotron independently under one runtime budget.
For local qualification, set `GEN2_BARK_BUNDLE`, `GEN2_ORT_LIBRARY`,
`GEN2_NEMOTRON_ASSETS` and `GEN2_NEMO_LIBRARY` to the pinned artifacts described
in [the Gen2 guide](../../docs/diarization.md). Packaged fallback locations are
`Resources/bark`, `Resources/runtime`, `Resources/nemotron` and
`Resources/nemo/lib`; packaged macOS builds prefer signed native libraries in
`Contents/Frameworks`. Same-host CPU relocation is verified; fresh-machine
qualification remains N5 work. No inference-time downloads occur.

Speaker staging requires a v2 macOS package, whose verifier binds the selected
profile, patches and floor-valid frame contract to its build receipt and original
SDK library identities. Staging selects the exact `gen2-valid-mel-v1`,
`gen2-valid-mel-pool-v1` or `gen2-valid-mel-pool-cpu-v2` lock from that verified package and rejects unknown or
mismatched profile/version identities. Historical v1 packages remain rejected.
The [static pooled staging regression](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-pnola-profile-staging/qualification.json)
retains the initial incorrect old-profile label and corrected result. Both
historical profiles stage successfully; only the candidate bundle's profile field
changes, with all other staged model/native bytes and modes identical. Five Node
and two overlay tests pass. The two known overlay files are upgraded with hash
guards and backups. This model-free staging is followed by the
[pooled complete-app runtime receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-pnola-app-runtime/collection-qualification.json):
automatic packaged CPU recording/live execution and explicit Metal rejection
pass without artifact-path environment overrides. Native GUI, microphone,
fresh-machine, quality and release qualification remain open.

The bounded CPU profile now selects `runtime-valid-mel-pool-cpu.lock.json`
explicitly, preserving exact manifest-to-lock profile/version checks and rejecting
future or suffix identities. Its initial unsupported-profile regression is fixed;
all 15 overlay Node tests and three apply/reapply tests pass. The real qualified
package selects the real bounded lock in a metadata-only check. The manifest
admits the two known previous overlay hashes, and a read-only check of the actual
private checkout reports exactly two file upgrades. No private source is changed.
Complete bounded-profile app construction and execution remain unqualified.

The [current-profile adapter receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-pnola-adapter/qualification.json)
passes ten debug test/CLI cases with explicit relocated artifacts: recorded
text/turn preservation, both partial-success directions, CPU/Metal paced
capture/live paths and retained-handle Metal shutdown. Both live runs cover
80.04 seconds, 11 ASR sections, 152 words and 15 provisional revisions. Complete
loader audits pass. This does not qualify automatic bundled resolution,
signed release artifacts, GUI, physical microphone or attribution accuracy.

The bundled `nemo/package.json` selects CPU or Metal execution. Supplying an
external `GEN2_NEMO_LIBRARY` defaults to CPU; set `GEN2_NEMO_EXECUTION=metal`
explicitly to qualify an external Metal SDK, or `cpu` to request CPU. Unsupported
values and malformed package metadata make speaker labels unavailable while
Bark remains usable. A missing requested Metal device fails speaker loading;
there is no automatic CPU fallback. Recorded and live timelines retain the
selected backend and optional device description, including saved live history.
This selection behavior alone is not a Metal app release receipt.

App shutdown rejects new jobs, fences startup loading, cancels active work and
waits for both native workers to release their models and reservations. CLI
verification uses the same cleanup barrier on success and early errors. Merely
requesting asynchronous unload can leave Metal resources alive at process exit.

Recorded audio uses one continuous native diarization stream, independently of
Bark's completed 60-second windows. Speaker labels have a one-hour cap; longer
ASR recordings remain supported with attribution explicitly unavailable.
Words use complete measured token intervals with their recording offset.
Zero-duration words, intervals over 800 ms, insufficient speech and speaker
transitions remain uncertain. Simultaneously active competing speakers remain
overlap candidates; a cleaned word is not attributed to a particular mixed voice.
These rules do not constitute calibrated confidence or precise forced alignment.

The IPC result and saved history add `attribution`, `audio_path`, `audio_sha256`
and optional ASR failure information. Native timeline results include a session
UUID, revision, completion state and model/runtime identity. Speaker renames
are scoped to that session. Old text-only history remains readable. Opening a
history entry rechecks the original audio SHA-256 before enabling seek buttons.
Edited text clears word labels while preserving native turns and speaker names.

If diarization is unavailable or fails, successful text is retained. If ASR
fails, the valid timeline is returned and an explicit **Save speaker timeline**
action persists it. A scoped child cancellation token prevents aborting a
diarization operation from accidentally cancelling its sibling ASR operation;
parent/user cancellation still reaches both.

Live capture now tees normalized PCM into one continuous native speaker stream
beside the eight-second durable ASR sections. The adapter batches at most 16640
samples, queues four blocks, and never waits for inference on the microphone
writer. Saturation, a model failure or the one-hour speaker cap marks retained
labels incomplete while ASR capture continues. Snapshots replace revisions for
one native session; the UI marks active labels provisional and preserves names
only within that session. Live turns do not assign words to voices.

Stopping closes the input, flushes the native tail and persists the final
speaker snapshot alongside the sectioned transcript. Recovery cannot restore
the native speaker cache: prior active labels become explicitly interrupted,
and resuming saved ASR sections never creates a supposedly continuous replacement
speaker session. Listing history while a native session still runs uses its
authoritative in-memory snapshot. Live section audio is removed after ASR under the existing
recovery policy, so live timeline playback is unavailable.

The real capture-output test feeds a licensed 60-second meeting at 20 ms
microphone cadence, checks provisional revisions, compares saved PCM sample for
sample, and compares final turns with the recording API. It uses recorded audio;
physical microphone and native visual qualification remain open.

Verification from Pnola:

```sh
npm run check
npm test
cargo test --manifest-path src-tauri/Cargo.toml
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
```

The three ignored recorded-file native tests in `transcription.rs` explicitly exercise real
transcription/timeline parity and each direction of partial success. They
require pinned local assets and fresh processes with the documented environment
variables. `capture::tests::real_live_pcm_tee_preserves_audio_and_speaker_session`
is also ignored by default and explicitly requires the same native artifacts.
The ordinary suite does not count these as executed model evidence.

With those variables set, `cargo run --manifest-path src-tauri/Cargo.toml --
--verify-live-speakers /absolute/mono16k.wav /absolute/receipt.json` feeds 80.04
seconds at capture cadence while the actual live worker runs Bark sections. It
requires complete durable ASR/export, provisional native revisions and final
speaker parity against an independent recording call, using the established
1e-6-second tolerance after JSON persistence (in-memory parity remains exact).
The 80.04-second development run passed with 11 ASR sections, 15 observed
provisional revisions and 35 final turns. `--verify-live` retains
the existing ASR-only verification mode.

Real AMI evaluation is intentionally reported with all omissions and unknown
labels included. The initial 60-second clip contains 147 normalized reference
words; Bark emits 117. The current output has 91 single-speaker words, 21 unknown
words and 5 overlap words. Chronological merged WER is 27.21%; speaker-tagged word error is 46.94%
when a word or speaker mismatch costs one substitution and unknown/overlap
cannot match a known person. Conservative
per-speaker concatenated WER is 65.31%, counting unknown/overlap as an unmapped
insertion stream while retaining missing reference words as deletions. This
single-clip investigation does **not** pass as a held-out attribution release
receipt. More corpus evaluation and alignment improvements remain required.

The roadmap's full N4/N5 and release gates remain open. No Linux, Windows or
Metal acceleration claim follows from these app-source changes.


## macOS app bundle

The full desktop build stages the pinned speaker model, card, OpenMDW-1.1 text
and verified CPU or Metal native package explicitly. Set these paths before building:

```sh
export GEN2_NEMO_PACKAGE=/absolute/nemo-cpu-package
export GEN2_NEMOTRON_ASSETS=/absolute/pinned-model-assets
export GEN2_BARK_BUNDLE=/absolute/bark/onnx
export GEN2_ORT_LIBRARY=/absolute/libonnxruntime.dylib
export GEN2_ORT_LICENSE=/absolute/onnxruntime/LICENSE
export PNOLA_SAMPLE_AUDIO=/absolute/licensed-demo.wav
npm run desktop:build
```

`GEN2_ROOT` optionally selects the Gen2 checkout with the packaging tools;
otherwise the helper follows Pnola's actual Cargo dependency path. Python 3 is a
packaging/verification tool only; the application performs native inference.
Builds do not download models or libraries. `PNOLA_SPEAKERS=0` explicitly selects
an ASR-only build and removes stale derived speaker resources. Missing speaker
paths in a full build fail clearly. Bark, ORT, the matching ORT license and
licensed demo audio paths are required for both build variants; no sibling
checkout or Python environment layout is assumed. All assets are staged in a
temporary directory before replacing the prior complete resource set. A failed
preparation preserves the previous assets.

The model and license are checked before publication. Failed staging retains
previous speaker resources. Every private native dependency and the public entry
are also declared under macOS `Frameworks`, letting Tauri sign them with the
same selected identity as the app. The loader prefers that directory; the
original native package remains in `Resources/nemo` for provenance checks.
The final bundle's model, card, license, original package, native identities,
closed dependency paths and individual/deep signatures are verified separately.
The app's deployment minimum is raised to the actual native package minimum:
26.0 for the current artifacts. These measurements qualify only the tested
M4 Pro/macOS 26.3 combination, not every Mac or an older macOS version.

Local builds use ad-hoc signing and do not inherit host notarization credentials.
Developer ID/notarization, fresh-machine GUI/capture and other OS/backend release
qualification remain open. The existing `--verify-live-speakers` command can be
run from the bundled executable with an external licensed audio fixture and
receipt path; it requires no source models, build checkout, SDK or network.
