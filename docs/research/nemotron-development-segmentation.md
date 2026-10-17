# Fixed development segmentation ablations

The original ES2004 accuracy gate remains failed. The separate ES2011a
development contrast showed that native production smoothing raises false
activity relative to lexical references, while reducing missed occupancy against
broader transcriber segments. That is a task tradeoff, not evidence of a wrapper
bug or a reason to substitute references.

Before selecting any production policy, replay the native segmenter exactly on
the saved four ES2011a [0,300) native outputs: both fixed geometries and both
channels. All probability-array identities must match the independently collected
context experiment. Float32 threshold/cadence arithmetic, strict hysteresis,
padding before gap filling, and short-segment filtering after merging are
preserved. Canonicalize equal-start turn order only; every baseline speaker and
boundary must otherwise match exactly, and both prior scorers must reproduce.

The fixed contrasts each remove one operation from the same native default:
edge padding, gap filling, or the minimum-duration filter. A fourth changes only
the onset/offset threshold pair to the already documented 0.5 pair, retaining
all smoothing. This is operation localization, not a threshold sweep. Report
every contrast for every geometry/channel, both transcriber and published NTT MFA
references, and both unchanged 10 ms and independently calibrated continuous
scorers. Preserve all MISS/FA/confusion contributions, deltas, and resulting turns.
Do not choose a winner or alter the original frozen test using these results.

Predictions are that padding/gap filling add lexical false activity, filtering
removes short replies, and higher thresholds increase misses. Interactions mean
an operation's ablation need not change scores monotonically. The operation
order remains fixed in all contrasts; this does not establish general acoustic
causality or the best production policy.

The [protocol](../../tools/nemotron-diarization/development-segmentation.protocol.json)
binds tools, saved arrays/JSON, reference/scorer files and native source evidence.
Current local module bytes are checked before imports and historical Git blobs
are checked before scoring. Every saved input is rebound after replay. No audio
acquisition, native process, model call or parameter search occurs. No ablation
has executed at this preregistration; synthetic segmentation tests run separately.

This is training-exposed development evidence from one meeting with weak
speaker-B coverage. References remain distinct, including automatic lexical
alignment limitations. It does not supply human word-timing gold, untouched test
generalization, compaction parity for a changed policy, public API selection,
GUI/microphone latency, performance, or release qualification.

## Actual saved-output contrast

The diagnostic was frozen at `fa6fe6b3` before execution. The
[10-member receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-development-segmentation/collection-qualification.json)
records four complete development cells, five configurations per cell, two
references and two scorers: 80 score reports, plus all resulting turns and error
component deltas. Separate collection replays every report and four scorer
controls. All 275 native baseline turns, including every boundary and speaker,
match exactly; only equal-start ordering is canonicalized. All 18 saved inputs
and result qualification bytes are rebound after replay. No model or native
helper ran. All 251 model-free tests pass, including the observed red/green
regression for rejecting reversed native start order.

Continuous-score DER percentages against the primary transcriber reference:

| Operation contrast | Low close | Low distant | Large close | Large distant |
| --- | ---: | ---: | ---: | ---: |
| Native default | 33.8593 | 34.3566 | 29.4237 | 32.2385 |
| Without edge padding | 47.7692 | 48.0011 | 43.8115 | 45.5753 |
| Without gap filling | 34.8200 | 35.4453 | 30.7333 | 33.6370 |
| Without short-segment filtering | 34.0999 | 34.3131 | 29.2410 | 32.1177 |
| Only onset/offset pair changed to 0.5 | 32.4724 | 36.0759 | 28.7071 | 32.4729 |

The separately reported automatic NTT MFA lexical reference gives:

| Operation contrast | Low close | Low distant | Large close | Large distant |
| --- | ---: | ---: | ---: | ---: |
| Native default | 33.7233 | 32.3852 | 28.3709 | 28.8127 |
| Without edge padding | 19.6784 | 20.1416 | 15.2161 | 17.7199 |
| Without gap filling | 32.1441 | 29.6695 | 26.4503 | 26.8874 |
| Without short-segment filtering | 34.1169 | 32.8460 | 28.4784 | 29.4838 |
| Only onset/offset pair changed to 0.5 | 36.0801 | 37.0863 | 29.5865 | 32.8120 |

Padding is a large contributor to the lexical false activity in these fixed
replays. For example, close/low lexical false activity falls from 36.182 to 7.442
speaker-seconds without it, while lexical misses rise from 5.585 to 16.656.
Against the broader transcriber reference, the same operation increases misses
from 59.293 to 94.241 speaker-seconds. The other three cells show the same
directional tradeoff. Gap filling has a smaller corresponding tradeoff.

Removing the duration filter makes small mixed changes; it does not generally
recover enough missing speech to explain the primary failure. Changing the
threshold pair improves primary close-channel DER but worsens both distant
cells and all four lexical scores. These one-operation contrasts therefore
provide no uniformly improving replacement. They do not test combinations or
identify a general best policy, and no policy was selected or implemented.

The next policy decision needs product-appropriate acoustic boundary review and
a separately frozen validation set. Choosing the lexical reference because some
of its ablations fall below 20% would change the objective and conceal the
original failures. The original ES2004 gate, human word-timing review and all
desktop/release requirements remain open.
