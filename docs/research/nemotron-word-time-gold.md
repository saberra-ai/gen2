# Human-reviewed word-time calibration candidates

Research checked 2026-10-16. No model, aligner, audio playback, native helper, CI or held-out evaluation was run. Only small official files and documentation were inspected; no account, agreement, email or permission request was submitted.

**Conclusion:** official TIMIT provides a directly obtainable tiny hand-verified example; official L2-ARCTIC explicitly provides a manually corrected word/phone example, but its WAV returned HTTP403 to this receiver. Neither is a permission-unrestricted meeting benchmark. Buckeye has suitable manual timing provenance, but the official corpus requires registration/licensing. A calibration route exists technically; a demonstrably licensed, fully acquired release-evidence fixture is not established. The focused EpaDB/MFA follow-up below did not identify a permissively licensed manual word-time alternative.

The [current N4 roadmap](/Users/victor/workspace/gen2-nemotron3-n0/docs/plans/nemotron-3-diarization-roadmap.md:532) requires broader timing, short-turn, overlap and attribution evidence. Its AMI comparison uses forced-aligned word times; the 800ms/50%/20% rules are uncertainty policy, not calibrated boundary accuracy. It specifies no manual-word p95 acceptance threshold. These candidates must not change AMI DER references, frozen criteria or historical receipts.

## 1. TIMIT official demonstration: acquired, permission unresolved

The [LDC publisher catalog](https://catalog.ldc.upenn.edu/LDC93S1) states that TIMIT word and phonetic transcriptions were hand verified. Its publicly linked sample is one isolated read utterance: “She had your dark suit in greasy wash water all year.” The following canonical endpoints all returned HTTP200 without credentials on this date:

| Official file | Bytes | SHA256 |
| --- | ---: | --- |
| [WAV](https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav) | 93,638 | `c7e8b51db10ec6a2aea8c92376f2b88af14c2b4950ed82db3d605a4b718c8f35` |
| [Word labels](https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wrd) | 181 | `fcac74838553186a60691ff468a3c10dd2e943c27195dcd45a656e47b510231a` |
| [Phone labels](https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.phn) | 535 | `069015f10f74af9921c91cf1f203c8e4cdb77a259b12177fb8c3e43a9de3be5e` |
| [Transcript](https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.txt) | 62 | `0b4ffafce18d6ec0df273c0189202a7c186cd0bbb9b40659490f5f2ad99e70a5` |

Static WAV parsing establishes mono 16-bit PCM, 16kHz, 46,797 samples / 2.9248125s. There are 11 word rows and 37 phone rows. The [official format documentation](https://catalog.ldc.upenn.edu/docs/LDC93S1/timit.readme.html) defines integer sample-number boundaries; divide by 16,000 to obtain seconds. Do not stretch terminal annotation bounds to the recording end.

**Anomaly to preserve:** the actual `.wrd` has `had`[5723,10337) and `your`[9190,11517): a1,147sample/71.6875ms overlap. This is not evidence of two speakers. A parser must report/preserve the published values, not silently make them contiguous or reconstruct “corrected” words from phones. It limits this example's use as unquestioned acoustic truth.

**Release-evidence limitation:** the catalog links the [LDC non-member agreement](https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf), which limits covered databases to noncommercial education/research/technology development and restricts redistribution outside the research group. This investigation does **not** establish that this unaccepted agreement applies to the public demo. It also found no separate permissive demo grant. Public HTTP access is verified; demonstrably licensed release-fixture redistribution is not. This is an evidence limitation, not an approval requirement for ordinary read-only research. The broader corpus requires LDC acquisition/licensing; no such acquisition occurred.

## 2. L2-ARCTIC LXC/arctic_a0018: explicit manual provenance, WAV blocked here

The [official TAMU project page](https://psi.engr.tamu.edu/l2-arctic-corpus/) explicitly identifies this exact example as manually annotated and says its manual subset has corrected **word and phone boundaries**. This is stronger provenance than merely human transcript text. Direct examples are [WAV](https://psi.engr.tamu.edu/wp-content/uploads/2018/03/lxc_arctic_a0018.wav) and [TextGrid](https://psi.engr.tamu.edu/wp-content/uploads/2018/03/lxc_arctic_a0018.txt). Full-corpus access requires name/email/affiliation, agreement to terms and an emailed download link; none was submitted. The publisher declares CC BY-NC4.0 and requests Zhao etal., Interspeech2018 citation.

The TextGrid returned HTTP200:3,579bytes, SHA256`bb2e198e06fb6646f7f60f6548e5ed7d6d558987bad52c4d0e9a278190d88573`. Its seconds-based `words` tier has five lexical intervals: there[.04,.28), was[.28,.53), a[.53,.64), change[.64,1.01), now[1.01,1.44), plus empty edge intervals. Total declared duration is1.6214965986394558s. It also contains `phones` and `IPA` tiers. WAV requests returned403 with both default headers and ordinary browser User-Agent/official-page Referer; no audio identity, sample rate or channel count was verified. Do not call this an acquired pair.

The [official documentation](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/) distinguishes `/annotation` manual TextGrids from `/textgrid` forced alignment and describes general WAV files as44.1kHz. Those general properties do not replace inspection of this blocked example. It is one read utterance by LXC, a female Mandarin-L1 speaker, with no multi-speaker overlap. [CC BY-NC4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) is conditional noncommercial permission with attribution requirements; applicability to this product-validation use remains unestablished. It is not a permissive commercial fixture license.

## 3. Buckeye: suitable labels, access gate

[Official transcription procedures](https://buckeyecorpus.osu.edu/php/corpusTran.php) describe automatic initial alignment followed by trained assistants correcting word/phone locations using waveform and spectrogram inspection. `.words`/`.phones` are ASCII labels with offset times in seconds, not independent start/end columns. [Corpus information](https://buckeyecorpus.osu.edu/php/corpusInfo.php) describes quiet-room close-headset conversational interviews. Interviewer intervals are event labels; do not infer complete independently transcribed two-speaker/overlap gold.

The [official registration page](https://buckeyecorpus.osu.edu/php/registration.php) requires an email-confirmed account. [FAQ](https://buckeyecorpus.osu.edu/php/faq.php) requires lab/class licensing and protected access, excluding public redistribution. No anonymous official audio-plus-label sample was verified from its linked public pages. This is a future licensed conversational calibration candidate, not an immediately frozen fixture.

## Proposed next experiment; not N4 acceptance

Choose/freeze the fixture before any Bark output and distinguish research calibration from demonstrably licensed release evidence. TIMIT can exercise16kHz interval plumbing, retaining its anomaly; the5-word L2 example requires actual WAV acquisition/hash/format verification first. Freeze raw gold, source-audio identity, any deterministic resampling and timing origin separately. Compare both start/end signed and absolute errors only for explicitly matched words; report missing, substituted, inserted and duplicate words independently with the full lexical denominator. Do not hide recognition failures behind matched-word p95. Treat tiny-sample statistics as descriptive calibration only, with no invented threshold. Broader independent human-reviewed conversational/meeting timing and speaker-attribution evidence remains required.

## Focused EpaDB/MFA follow-up: no permissive gold established

| Candidate | Primary evidence and disposition |
| --- | --- |
| EpaDB | The [original publisher paper, §2.2](https://www.isca-archive.org/interspeech_2019/vidal19_interspeech.pdf) describes expert phone-level alignments/transcriptions in addition to utterance nativeness scores, initialized by MFA. This is more than human scores or transcript text; however, reviewed **word-boundary** provenance is not explicit there. The [author-owned recipe README at24925727090598385cec088900848f8ec8aa40f4](https://github.com/JazminVidal/gop-dnn-epadb/blob/24925727090598385cec088900848f8ec8aa40f4/README.md) declares code **and database** research-only CC BY-NC-ND4.0, not CC BY4.0. Its prerequisite links [the full corpus on Drive](https://drive.google.com/file/d/1sk8mw2I86MitCRfIEm5kwn5hopmMmSIb/view?usp=sharing); no small official paired sample was verified. No Drive archive was acquired. Ruled out as the requested permissive release fixture. |
| Official MFA evaluation | [MFA2026 §4.1](https://arxiv.org/html/2606.18466v1#S4.SS1) identifies TIMIT and Buckeye as its English human-corrected benchmarks; this does not create a new license for their audio. Its [pinned benchmark repository](https://github.com/MontrealCorpusTools/mfa-interspeech2026/tree/8ec1efa7d3caeea1cee3bfd018b38c2167af5dc5) contains scripts, no WAV/TextGrid payloads. The main MFA repository does expose an [acoustic_corpus WAV](https://raw.githubusercontent.com/MontrealCorpusTools/Montreal-Forced-Aligner/93411fc0df3bc770dfd34c4282dc136fdf6e8dcb/tests/data/wav/acoustic_corpus.wav) and [word/phone TextGrid](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/93411fc0df3bc770dfd34c4282dc136fdf6e8dcb/tests/data/textgrid/acoustic_corpus.TextGrid), listed as855,188/4,697bytes, with a26.72325s annotation span. [Tests use it as a reference](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/93411fc0df3bc770dfd34c4282dc136fdf6e8dcb/tests/conftest.py#L408) and the [repository license is MIT](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/93411fc0df3bc770dfd34c4282dc136fdf6e8dcb/LICENSE), but no documented human timing/correction provenance was verified. Do not promote a software-test reference to human-boundary gold. WAV not acquired. |

Stop this focused search here. No new sample is selected for gold evaluation; the gaps above are factual provenance/access limits, not a model result or a reason to alter N4 criteria.

Acquired small files and retrieval metadata are retained outside Git in [research cache](/Users/victor/.cache/gen2-nemotron3-n0/research/word-time-gold-2026-10-16/manifest.json); the manifest also records the L2 WAV403. No publisher sample audio is added to this repository.
