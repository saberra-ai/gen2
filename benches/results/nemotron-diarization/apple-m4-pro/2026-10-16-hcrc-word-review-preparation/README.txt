This package is UNREVIEWED timing-calibration material, not human gold.

Selection: q1ec1 [0,60), fixed before Bark/native/aligner output. This is a
close-microphone two-person dialogue, not representative meeting acceptance.
Publisher training exposure is unknown; do not claim unseen test data.

Use mono16k.wav as the exact future model/review timebase. stereo20k.wav retains
both source channels for identity checking; left/right role assignment has NOT
been verified. Source participant IDs are metadata hints, not acoustic proof.

Have one human inspect the complete waveform/spectrogram and audio, then a
second human independently check every word start/end, lexical text and speaker
identity. Neither may view Bark, aligner or diarizer outputs before freezing the
adjudicated annotation. Review the entire crop, including omissions or errors
in source text. Add missing words; retain uncertainty, partial words at crop
edges, noise and disagreements explicitly. Do not distribute a multiword timed
unit's duration proportionally. Enter individual lexical boundaries only when
actually reviewed. Existing source timings have unresolved human-review origin.

Copy review-template.json elsewhere and fill the copy with actual review
identities/procedure and individual words. Preserve this original package and
its receipts unchanged. Source units are aids only. Merely editing a status field or
running this preparer does not establish human review, timing accuracy or gold
admission. Freeze and hash a separately adjudicated annotation before any model
comparison. Count missing/substituted/inserted words as well as matched-word
boundary errors. This package has no acceptance score or inferred labels.

Current publisher download page and separate audio notice grant CC BY 4.0.
The original annotation archive embeds an older CC BY-NC-SA 2.5 notice; both
notices are retained verbatim. Preserve attribution and this discrepancy.
