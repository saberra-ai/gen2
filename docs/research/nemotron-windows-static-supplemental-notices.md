# Windows SentencePiece supplemental notices

Research only, 2026-10-16. The concrete packaging recommendation is to retain Darts-clone's complete notice and conservatively add both esaxx's notice and sais-lite's distinct embedded notice, alongside the existing four vcpkg dependency licenses. This adds 3,767 bytes of notice text. The source/build evidence supports Darts-clone in the SentencePiece runtime source; esaxx/sais-lite coverage is explicitly conservative because their identified implementation belongs to a separate training target. This is source and notice attribution evidence, not proof of which static archive members entered the qualified NeMo DLLs, full license compliance, or release qualification. No SDK rebuild, model run, SDK mutation or receipt rewrite was performed.

## Verified inputs and source boundary

The [source-recipe receipt](/Users/victor/workspace/gen2-nemotron3-n0/benches/results/nemotron-diarization/windows-x86_64/2026-10-16-source-recipes/manifest.json) and [earlier provenance research](/Users/victor/workspace/gen2-nemotron3-n0/docs/research/nemotron-windows-static-provenance.md) bind all six port directories to vcpkg baseline `9e593bb18ea69cc5095e012465dcd675a822ed0d`, including the exact cached package port-file/ABI checks. This investigation read the already-acquired SentencePiece v0.2.1 archive, whose actual SHA512 is `012850b63b2323e16acc5dacc0a494ad3f6375425ee86274f0946032e47c088a3b307758b99d752fcf54acf76c82d7d13d0c14bbf07aa9b612c4f1fbd30cf1cf` and size is 13,485,527 bytes. The full source archive remains outside Git. The subsequent packaging change retains the three exact raw notices and their portable provenance in [the tools notice directory](../../tools/nemotron-diarization/notices/sentencepiece/provenance.json). Primary links below identify upstream source paths; the verified archive content hash fixes the bytes behind those tag URLs.

The [pinned port recipe](https://github.com/microsoft/vcpkg/blob/9e593bb18ea69cc5095e012465dcd675a822ed0d/ports/sentencepiece/portfile.cmake) sets `SPM_ABSL_PROVIDER=package`, `SPM_PROTOBUF_PROVIDER=package`, `PROTOBUF_LITE_LIBRARY=protobuf::libprotobuf-lite`, and `SPM_ENABLE_TCMALLOC=OFF`; Windows forces static SentencePiece. The cached port hash matches this recipe. Its [Abseil patch](https://github.com/microsoft/vcpkg/blob/9e593bb18ea69cc5095e012465dcd675a822ed0d/ports/sentencepiece/abseil.diff) replaces the bundled Abseil directory with headers copied from the external package. In [SentencePiece's source build](https://github.com/google/sentencepiece/blob/v0.2.1/src/CMakeLists.txt), package Abseil sets the external-provider definition and links Abseil targets; package protobuf generates the proto sources and empties the bundled `PROTOBUF_LITE_SRCS`. Thus the already-retained **external** Abseil/protobuf licenses are the relevant provider notices for this recipe. The mere presence of their older vendored directories in the archive does not establish that those bundled implementations were used. The [linkage patch](https://github.com/microsoft/vcpkg/blob/9e593bb18ea69cc5095e012465dcd675a822ed0d/ports/sentencepiece/linkage.diff) affects shared-build installation choices; it does not remove the runtime/training source separation below.

The port installs only SentencePiece's root `LICENSE`. That is byte-identical to its cached copyright, but it omits distinct embedded third-party notices. The existing protobuf and utf8-range installed notices were also verified against their corresponding upstream LICENSE files. They should remain intact; the supplemental notices supplement rather than replace them. The `vcpkg-cmake` and `vcpkg-cmake-config` host-helper notices belong in retained build/source provenance if those helpers are included, without representing them as production runtime libraries.

## Runtime evidence and conservative coverage

| Component | Exact source/build evidence | Packaging treatment |
| --- | --- | --- |
| Darts-clone | Direct includes in [normalizer.h](https://github.com/google/sentencepiece/blob/v0.2.1/src/normalizer.h), [normalizer.cc](https://github.com/google/sentencepiece/blob/v0.2.1/src/normalizer.cc), [model_interface.h](https://github.com/google/sentencepiece/blob/v0.2.1/src/model_interface.h), and [unigram_model.h](https://github.com/google/sentencepiece/blob/v0.2.1/src/unigram_model.h). `normalizer.cc` and these model headers belong to `SPM_SRCS`, used to create `sentencepiece-static`. | Add its complete BSD-style license, including Susumu Yata attribution, conditions and disclaimer. Preserve the literal `<ORGANIZATION>` placeholder in the upstream text. This is runtime-source evidence; no exact member-selection claim. |
| esaxx | [unigram_model_trainer.cc](https://github.com/google/sentencepiece/blob/v0.2.1/src/unigram_model_trainer.cc) includes `third_party/esaxx/esa.hxx` and calls `esaxx(...)`. This file is in `SPM_TRAIN_SRCS`, used for `sentencepiece_train-static`, while `sentencepiece-static` uses `SPM_SRCS`. No other direct esaxx include/call was found in the archive's source/header scan. | Add its MIT-style Daisuke Okanohara notice conservatively. It is justified for distribution of the trainer library/tools or relevant source; its presence in the current NeMo runtime is not established by this investigation. |
| sais-lite | [esaxx/esa.hxx](https://github.com/google/sentencepiece/blob/v0.2.1/third_party/esaxx/esa.hxx) includes `sais.hxx`. The initial comment of [sais.hxx](https://github.com/google/sentencepiece/blob/v0.2.1/third_party/esaxx/sais.hxx) has a separate MIT-style notice for Yuta Mori, 2008–2009. `esaxx/LICENSE` names a different author and does not reproduce this attribution. | Whenever including conservative esaxx coverage, retain this distinct notice too. Preserve the original block comment verbatim; the complete header need not be shipped just to reproduce its notice. |

The [source build](https://github.com/google/sentencepiece/blob/v0.2.1/src/CMakeLists.txt) installs both static SentencePiece libraries and builds training/normalization tools; a complete vcpkg binary ZIP therefore has a broader notice scope than a narrowed runtime package. Cached package inventory, source target structure and notice text should not be conflated with native linker output. Retaining all three small notices avoids selecting licensing text based on unproven dead-stripping or assumed absence of training archive members. It does not assert those members are present.

## Exact supplemental data

All selections below are copied verbatim from the verified SentencePiece archive. For sais-lite the selection is precisely bytes `[0, 1174)` of `third_party/esaxx/sais.hxx`, ending with its first `*/`, with no added newline; the full header is 12,385 bytes with SHA256 `6c750f6cf098d346c9d1723291a062bc9154657d5be2c3833601c5a33a5b4d1d`. The review copies in this Markdown are fenced for display; the external raw files are authoritative for byte verification.

| Suggested package path below `notices/nemo-speech/third_party/vcpkg/sentencepiece/` | Upstream member below `sentencepiece-0.2.1/` | Raw bytes | SHA256 |
| --- | --- | ---: | --- |
| `darts-clone-LICENSE` | [`third_party/darts_clone/LICENSE`](https://github.com/google/sentencepiece/blob/v0.2.1/third_party/darts_clone/LICENSE) | 1,481 | `155f59997298ee336602c49f9c1110f268ac394ca2197eb02647a3555935ad52` |
| `esaxx-LICENSE` | [`third_party/esaxx/LICENSE`](https://github.com/google/sentencepiece/blob/v0.2.1/third_party/esaxx/LICENSE) | 1,112 | `7c28553d1d3312d65fe309f76a22ebaf33a3d76c8c1e3b98a88ee4654ecb53db` |
| `sais-lite-NOTICE.txt` | [`third_party/esaxx/sais.hxx`](https://github.com/google/sentencepiece/blob/v0.2.1/third_party/esaxx/sais.hxx) | 1,174 | `162a72e52ad69921a67676af4fb59911233566dce1637e2e33db27659fcac523` |

Keep the existing `sentencepiece/LICENSE` as the Apache 2.0 text. Add these three sibling files through a separately reviewable packaging change; bind their exact bytes, source archive hash and selection policy into the resulting package manifest. Repackage without changing native DLL bytes, then verify the package's notice inventory/hashes and re-run the package-only verification required by the existing deployment gate. This research makes that change concrete but does not perform it or declare the gate passed. Label esaxx/sais-lite as conservative source/training-component attribution in provenance, not as a list of proven linked components.

The [supplemental manifest](/Users/victor/.cache/gen2-nemotron3-n0/research/windows-static-source-closure/supplemental-notices/manifest.json) retains the source archive identity, exact raw source/notice hashes, suggested package paths and hashes of eight inspected source files. It does not alter the prior acquisition manifest or original receipts. Review future full SDK/source distributions separately: this narrow supplement does not attest that every source-file header, optional feature, generator, external tool or upstream dependency has been exhaustively audited.

## Verbatim Darts-clone notice

```text
Copyright (c) 2008-2011, Susumu Yata
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

```

## Verbatim esaxx notice

```text
This is the esaxx copyright.

Copyright (c) 2010 Daisuke Okanohara All Rights Reserved.

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

```

## Verbatim sais-lite notice

```text
/*
 * sais.hxx for sais-lite
 * Copyright (c) 2008-2009 Yuta Mori All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
```

## Subsequent constructor integration

The Windows constructor now installs all three exact notices and `gen2-supplement-provenance.json`, records the source selection in `source_notice_supplement`, and verifies both text and provenance against the pinned policy. Existing historical development packages remain verifiable without that additive field; this does not retroactively repair their notice inventory. Native DLL bytes and original installed upstream notices stay unchanged. The current local macOS constructor/ZIP restoration and 88 model-free tool tests pass. Actual Windows execution of this supplemented artifact, complete attribution compliance, Microsoft distribution rights and release qualification remain separate gates.
