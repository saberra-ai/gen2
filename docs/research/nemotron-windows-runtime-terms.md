# Exact Microsoft VC runtime terms and redistribution provenance

Primary-source inspection, 2026-10-16. Only this note and an external research cache were written. No existing receipts were altered; no package installation, native build or model execution occurred. This records document contents and an engineering retention plan, not legal advice or a finding about the user's entitlement.

## Exact official document inspected

Microsoft's [VC Runtime terms landing page](https://visualstudio.microsoft.com/license-terms/vs2022-cruntime/) embeds an Office viewer whose source is this [official DOCX](https://visualstudio.microsoft.com/wp-content/uploads/2021/09/Visual-C-Runtime-2015-2022-License-1.docx). The link was verified through the page's Embedded Document control. The exact canonical source returned the following bytes:

| Field | Observed value |
| --- | --- |
| Retrieved UTC | `2026-10-16T09:44:56.796537+00:00` |
| Bytes | `39644` |
| SHA256 | `f1e3d56ceb2ad68aae0711b910375009e651ac5530fa0760f0dea6e81e54fae1` |
| Document EULA ID | `Cpp_2015-2022_ENU.1033` |
| HTTP ETag | `"62cd820b-9adc"` |
| HTTP Last-Modified | `Tue, 12 Jul 2022 14:15:39 GMT` |

The bare urllib request returned 403; the successful request used `Mozilla/5.0` User-Agent and the official terms landing-page Referer at the same canonical URL, without query parameters or byte transformations. SHA256, not ETag/Last-Modified, is the retained content identity. Binary cache: [official DOCX](/Users/victor/.cache/gen2-nemotron3-n0/research/Visual-C-Runtime-2015-2022-License-1.docx); [retrieval metadata](/Users/victor/.cache/gen2-nemotron3-n0/research/Visual-C-Runtime-2015-2022-License-1.metadata.json). Text was extracted from DOCX `word/document.xml` for inspection, not substituted for the original. The derived [UTF-8 text](/Users/victor/.cache/gen2-nemotron3-n0/research/Visual-C-Runtime-2015-2022-License-1.txt) is 6,589 bytes, SHA256 `7e96c27c67ef3e7cabc55e1dee05727e67d261f3944155a816a07dbf9b78fb1d`; the binary is authoritative. Both cache files remain outside Git.

## What it covers

The document applies to Microsoft Visual C++ 2015–2022 Runtime and related services/updates unless they supply different terms. Its Installation and Use Rights permits unlimited installation/use copies. Its Scope of License reserves ungranted rights, restricts modifying Microsoft/supplier notices, and prohibits sharing/publishing or supplying the runtime alone or bundled with applications to third parties. It also points to accompanying terms/notices for Microsoft platform and third-party components. Thus this document is runtime **installation/use terms**, not the affirmative grant for our app-local redistribution. Retaining it does not by itself establish a distribution right. [Exact inspected document](https://visualstudio.microsoft.com/wp-content/uploads/2021/09/Visual-C-Runtime-2015-2022-License-1.docx).

The separate [VS2022 REDIST list](https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution#visual-c-runtime-files) identifies VC/redist files that licensed users may distribute unchanged with their program, subject to their applicable VS terms; debug_nonredist is excluded. Microsoft's [deployment guidance](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170) explains individual application-local deployment and recommends central redistribution for servicing. These are distinct sources for distinct roles. A public reference EULA, REDIST-list snapshot, or a hosted runner's VS installation does not establish this project's distributor entitlement.

## Actionable retention for the actual 14.44 DLLs

The [recorded source redist root](/Users/victor/workspace/gen2-nemotron3-n0/benches/results/nemotron-diarization/windows-x86_64/2026-10-16-native-loader-control/cmake-cache.log) is `C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Redist/MSVC/14.44.35112`. That path and this 2021/2022 public EULA are not evidence that the exact downloaded EULA accompanied those particular 14.44 files. Keep that limitation explicit.

1. Freeze the public DOCX above as a clearly labeled **official reference runtime terms** artifact, preserving source URL/date/hash. Package-side provenance may retain the unmodified DOCX and an optional labeled text extraction. Do not call it the applicable 14.44 distribution license solely because the family name matches.
2. Capture each shipped CRT DLL's exact source path under the licensed redist tree, source/install SHA256 equality, actual file/product version and Microsoft signature evidence. Retain the actual accompanying runtime/platform/third-party license and notice files, if present, with path/hash; no universal installation-relative EULA pathname has been established by current evidence.
3. Retain the applicable VS/runtime terms supplied by the distribution channel and the REDIST-list reference separately from end-user runtime terms. Store a short machine-readable provenance record distinguishing public reference documents, artifacts actually accompanying the redist, and the distributor's separately established rights basis. An explicit value such as `redistribution_entitlement: not_established_by_build_receipt` accurately describes the current evidence.
4. If accompanying terms/notices or rights-basis evidence are unavailable, preserve the reproducible development/package evidence and report that specific open provenance gate. Do not silently treat the public DOCX as a permissive redistribution license, relabel the CRTs Apache/MIT, modify Microsoft binaries/notices, or mark release qualification complete.

This plan supplies a concrete reference document and a way to retain applicable evidence; it does not close entitlement, clean-machine packaging, lifecycle, coexistence or model-quality gates.
