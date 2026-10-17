# Model-blind word-timing calibration preparation

This prepares the concrete human work needed for N4. It supplies no gold labels,
model comparison, acceptance score or release qualification. The original DER,
attribution, desktop and held-out criteria remain unchanged.

## Frozen selection

Use HCRC Map Task `q1ec1` **[0,60)**, selected before any Bark, aligner or
Nemotron output on this recording. This is a small two-person close-microphone
calibration, not a representative meeting test. Publisher training exposure is
unknown. The [source investigation](nemotron-manual-word-boundary-candidates.md)
retains the unresolved origin of the supplied timing boundaries.

[`hcrc-word-review.protocol.json`](../../tools/nemotron-diarization/hcrc-word-review.protocol.json)
binds the original annotation ZIP/members, participant metadata, recorded audio
ETag/header/range, five original notices, preparer/test sources and numerical
conversion. The fixed crop contains 111 giver and 14 follower source timed
units. Source multiword units remain intact; the last source word crosses the
crop boundary and retains its original end, rather than acquiring an invented
complete-word end at60seconds. Source role-to-participant IDs are metadata hints;
left/right-channel identities have not been established acoustically.

The source is stereo PCM16/20kHz. The package preserves that exact cropped PCM
and also creates the future review/model timebase: float64 equal-channel mean,
SciPy `resample_poly` up4/down5, Kaiser5.0, constant-zero boundary padding, then
clipping and round-to-even PCM16 mono/16kHz. The result has exactly960,000samples.
Python3.12.13, NumPy2.2.6 and SciPy1.18.1 are required. No onset shift is added;
filter-boundary transients and crop-edge uncertainty require actual review.

## Preparation command

With the original ZIP and protocol-bound notice files available outside Git:

```sh
python tools/nemotron-diarization/prepare_word_timing_review.py \
  --protocol tools/nemotron-diarization/hcrc-word-review.protocol.json \
  --annotations /absolute/path/hcrcmaptask.nxtformatv2-1.zip \
  --output /absolute/fresh/path/hcrc-word-review
```

The preparer checks source identities before downloading only the64-byte header
and4,800,000-byte PCM range. It requires exact206responses and the original ETag,
rejects output within this repository or an existing output directory, and
never imports/runs a recognizer, aligner or diarizer. Existing artifacts remain
unchanged. Audio and complete annotation payloads stay outside Git.

The package contains `mono16k.wav`, `stereo20k.wav`, source units explicitly
marked unreviewed, an empty human-review template, original notices, instructions
and byte/hash receipts. The current publisher/audio CC BY4.0 notices and the
archive's older BY-NC-SA2.5 notice remain verbatim; the discrepancy is preserved.
See the publisher [download page](https://groups.inf.ed.ac.uk/maptask/maptasknxt.html)
and [audio notice](https://groups.inf.ed.ac.uk/maptask/signals/dialogues/00LICENSE.html).

## Human work still required

One human must inspect the entire waveform/spectrogram and audio, and a second
must independently review every lexical start/end, word and speaker identity.
Both must remain blind to model/aligner outputs until adjudicated annotations
are frozen. Review the entire crop, not only the supplied source units; preserve
missing or incorrect source words, uncertainty, disagreements, noise and partial
words at the crop edge. Verify speaker identities acoustically using both audio
versions; metadata role hints alone are insufficient. Do not proportionally
split a multiword timed unit. Any individual boundaries require actual inspection.

Copy the template outside the original package before editing; retain the
original inputs and receipts unchanged. The template begins with no annotators
and no words. Running the tool or changing
a status flag cannot establish human review. Record actual annotator identities,
procedure, corrections and unresolved cases, then freeze a separate adjudicated
reference with the exact canonical-audio digest before a later model comparison.
Missing/inserted/substituted words must remain in word-error reporting; timing
error on matched words cannot replace full transcript/attribution evaluation.
No new numerical timing acceptance threshold is introduced.

## Current verification

All240 local tools tests pass, including bounded oversized-input rejection and actual opposite-channel PCM
conversion and a source-unit regression preserving multiword/crop-edge
uncertainty without filling the human template. The NumPy-only CI environment
can run the metadata regression; the numerical conversion test explicitly skips
when SciPy is absent. These are preparer checks, not human annotation evidence.
At this preregistration, audio preparation and human annotation remain unrun.
