# N4 word-boundary candidates: HCRC and Montclair Map Task

Research checked 2026-10-16 under the research skill. Exactly two distinct candidate corpora were investigated. No model, aligner, native helper, CI, audio playback, account, payment or external message was used. Only HCRC annotations/documentation and a64-byte WAV prefix were acquired outside Git; source audio was not acquired in full. Existing notes, frozen tools/protocols and timing/DER criteria remain unchanged.

**Conclusion:** HCRC Map Task is the strongest **accessible licensed dialogue/reference candidate**, with real word-sized timed units and persistent speaker IDs. However, the inspected primary sources do **not explicitly establish acoustic boundary placement/review by humans**, so it is not yet admitted as human-corrected word-boundary gold. Montclair Map Task is excluded from the current route because access requires institutional IRB oversight and contacting its administrator. Neither finding closes N4 or establishes unseen-data generalization.

## 1. HCRC Map Task: access and actual annotations verified

The [official download page](https://groups.inf.ed.ac.uk/maptask/maptasknxt.html) offers the annotations and individual dialogue WAVs without authentication. It explicitly licenses the downloads linked there under **CC BY4.0**, including accompanying audio. The [audio directory's own license](https://groups.inf.ed.ac.uk/maptask/signals/dialogues/00LICENSE.html) is also CC BY4.0. This is stronger evidence than relying on a software repository license or the separately sold LDC edition. The [corpus home page](https://groups.inf.ed.ac.uk/maptask/) requests intended-use notification as a courtesy; it does not make that a download prerequisite. No notification was sent.

There is an actual versioned notice discrepancy: the downloaded ZIP's `00README.txt` and `00LICENSE.html` still say **CC BY-NC-SA2.5**, while the current publisher download page says CC BY4.0. Retain both original notices and the current grant rather than silently replacing archive contents. The current publisher grant supports research acquisition; the discrepancy should be recorded wherever the package's permission provenance is retained. It is not a reason to label the data unlicensed or require an extra approval flow for this read-only research.

The [original corpus-author description](https://aclanthology.org/H93-1005.pdf) describes unscripted English dyadic route negotiation by64talkers, mostly from Glasgow, captured with separate close-talking microphones on stereo audio. It notes some cross-channel leakage and relatively little interruption/overlap. That is useful conversational calibration, but not a four-/eight-speaker, distant-microphone meeting evaluation. Its original1993transcripts used sample anchors every few turns; those sparse anchors must not be mistaken for the later timed-unit layer.

### Timing provenance and unit limits

The corpus authors' [1998 architecture paper, §§1–2](https://www.isca-archive.org/icslp_1998/isard98_icslp.pdf) describes the base unit as “roughly a word” and says it is produced “semi-automatically from XWaves (TM) xlabel files.” That statement describes a source/format conversion, but does not specify who placed the acoustic boundaries, whether forced alignment initialized them, or a human checking/correction procedure. Do **not** infer either fully manual timing or unreviewed forced alignment from it.

That paper explicitly allows run-together multiword units such as “do you have a” when individual word timings are inappropriate. The acquired archive README similarly distinguishes a timed unit such as `there's` from its untimed lexical parts. [Official structure documentation](https://groups.inf.ed.ac.uk/maptask/file-structure.html) states that start/end attributes refer to seconds on the shared recording timeline and that speaker streams are recorded on separate channels. `.timed-units.xml` provides `<tu>` speech units plus `<sil>` and `<noi>` events; `.tokens.xml` may split a unit but cannot invent its internal acoustic boundaries.

**Gold status remains unresolved.** Actual starts/ends exist; explicit human acoustic-boundary correction provenance was not found in the inspected corpus-author paper, official format documentation, release README or changelog. No boundary-error/reliability claim is imported from a later paper that merely calls these times precise.

### Concrete small metadata and bounded audio check

The [official v2.1 annotation ZIP](https://groups.inf.ed.ac.uk/maptask/hcrcmaptask.nxtformatv2-1.zip) returned200:12,206,088bytes, SHA256`e90e083179c4fcdb7292d2a38c3238401412bbb4c3706b987befdc02e4c8af30`. Only metadata members were inspected; no bundled code was executed.

| Acquired member | Bytes | SHA256 |
| --- | ---: | --- |
| `maptaskv2-1/00README.txt` | 14,035 | `118cc77b72bb77a844e948a4107faef0eed83a52e5231b2371763a57b62ad461` |
| `Data/timed-units/q1ec1.f.timed-units.xml` | 7,549 | `45562fd5e901079708645c7dd17be263c0b31a84e9cf84906c3b0b749e220da2` |
| `Data/timed-units/q1ec1.g.timed-units.xml` | 46,306 | `c8b3d8b9e639141bbf03e01cc6e7d82c505fc04ac77e044c97c5e50d9d68146b` |

Both timed-unit paths are relative to the ZIP's `maptaskv2-1/` root. Static XML inspection finds50follower `<tu>` elements and479giver `<tu>` elements; the giver includes four `going to` units. Do not relabel these529units as529individually timed orthographic words. `Data/corpus-resources/maptask-corpus.xml` maps q1ec1's giver to participant`q1eta1` and follower to`q1etb1`; keep these identities distinct from role labels that can change between dialogues. Exact left/right-channel assignment was not verified from the inspected metadata.

The matching [q1ec1.mix.wav endpoint](https://groups.inf.ed.ac.uk/maptask/signals/dialogues/q1ec1.mix.wav) honored `Range: bytes=0-63` with206and `Content-Range: bytes 0-63/21207164`; ETag`"143987c-492e18fff1b00"`, Last-Modified`Mon, 18 Oct 2010 10:23:08 GMT`. The retained prefix is64bytes, SHA256`4516b8c03861b8d040de59016fe6477effd682743c077db4e346dd148ca70ef9`. Its RIFF header declares PCM16,20kHz,2channels,5,301,780frames/265.089s. This is endpoint/header evidence, **not** a full-audio SHA or decode verification. The full recording is21,207,164bytes and was not downloaded. No recording/crop is selected for evaluation here.

## 2. Montclair Map Task: access restriction rules out this route

The [official Montclair repository](https://digitalcommons.montclair.edu/mmt_corpus/) describes48natural English two-speaker conversations, with two-channel WAVs, corresponding Praat TextGrids and extracted CSVs. It explicitly restricts recording/data access to research users at institutions with IRB oversight and directs users to request access from the corpus administrator. The [description landing page](https://digitalcommons.montclair.edu/mmt_corpus/1/) reiterates that restriction; only its overview PDF is public. Public map-task instruction documents do not grant access to recordings or timing gold. No request was made, no recording/label pair was verified, and TextGrid presence alone establishes neither word-level granularity nor human boundary correction. Excluded under the current no-authentication/no-contact acquisition scope.

## Operational recommendation

Do not treat HCRC's existing timing layer as human-corrected gold without stronger primary provenance. It is nevertheless a concrete CC-licensed source for a **new model-blind manual calibration**: freeze an exact recording/crop and deterministic mono/resampling policy before Bark output, then have independent human annotators inspect every selected lexical start/end and speaker identity, adjudicating uncertainty explicitly. Preserve multiword units until genuinely retimed; do not distribute their duration proportionally. This is proposed work, not completed annotation or acceptance evidence.

Any eventual calibration must disclose whether its recordings overlap prior experiments or publisher training data; no “unseen” claim follows from this search. Broader meeting/overlap/short-reply gates remain open, and no numerical word-timing acceptance threshold is introduced.

All acquired metadata and member hashes are retained in [research cache manifest](/Users/victor/.cache/gen2-nemotron3-n0/research/manual-word-boundaries-2026-10-16/manifest.json). No corpus payload was added to Git.
