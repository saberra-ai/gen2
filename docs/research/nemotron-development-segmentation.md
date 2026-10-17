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
