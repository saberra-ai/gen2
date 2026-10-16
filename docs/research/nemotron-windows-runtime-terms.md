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

The separate [VS 2022 REDIST list](https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution#visual-c-runtime-files) identifies VC/redist files that licensed users may distribute unchanged with their program, subject to their applicable VS terms; debug_nonredist is excluded. Microsoft's [deployment guidance](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170) explains individual application-local deployment and recommends central redistribution for servicing. These are distinct sources for distinct roles. A public reference EULA, REDIST-list snapshot, or a hosted runner's VS installation does not establish this project's distributor entitlement.

## Actionable retention for the actual 14.44 DLLs

The [recorded source redist root](/Users/victor/workspace/gen2-nemotron3-n0/benches/results/nemotron-diarization/windows-x86_64/2026-10-16-native-loader-control/cmake-cache.log) is `C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Redist/MSVC/14.44.35112`. That path and this 2021/2022 public EULA are not evidence that the exact downloaded EULA accompanied those particular 14.44 files. Keep that limitation explicit.

1. Freeze the public DOCX above as a clearly labeled **official reference runtime terms** artifact, preserving source URL/date/hash. Package-side provenance may retain the unmodified DOCX and an optional labeled text extraction. Do not call it the applicable 14.44 distribution license solely because the family name matches.
2. Capture each shipped CRT DLL's exact source path under the licensed redist tree, source/install SHA256 equality, actual file/product version and Microsoft signature evidence. Retain the actual accompanying runtime/platform/third-party license and notice files, if present, with path/hash; no universal installation-relative EULA pathname has been established by current evidence.
3. Retain the applicable VS/runtime terms supplied by the distribution channel and the REDIST-list reference separately from end-user runtime terms. Store a short machine-readable provenance record distinguishing public reference documents, artifacts actually accompanying the redist, and the distributor's separately established rights basis. An explicit value such as `redistribution_entitlement: not_established_by_build_receipt` accurately describes the current evidence.
4. If accompanying terms/notices or rights-basis evidence are unavailable, preserve the reproducible development/package evidence and report that specific open provenance gate. Do not silently treat the public DOCX as a permissive redistribution license, relabel the CRTs Apache/MIT, modify Microsoft binaries/notices, or mark release qualification complete.

This plan supplies a concrete reference document and a way to retain applicable evidence; it does not close entitlement, clean-machine packaging, lifecycle, coexistence or model-quality gates.


## Follow-up: actual 24ad927 runtime versions and a newer pinned candidate

Research update, 2026-10-16; no installer body was downloaded or executed, and no builds/models were run. The successful private-SDK run 38882923286 at Gen2 source 24ad927 now has [actual Windows provenance](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-private-native-development/windows-provenance.json): `cl.exe` file 19.44.35229.0/product 14.44.35229.0; all eight VC DLLs file/product 14.44.35211.0; exact originals under the recorded `VC/Redist/MSVC/14.44.35112/x64/Microsoft.VC143.CRT` match installed bytes, with observed Microsoft Authenticode Valid status. Its runtime-license candidate search is empty. These observations materially strengthen binary provenance, but do not themselves establish compatibility policy or license applicability/entitlement.

### What Microsoft's version rule establishes

Microsoft's [binary compatibility guidance](https://learn.microsoft.com/en-us/cpp/porting/binary-compat-2015-2017?view=msvc-170) requires the runtime to be at least as new as the newest build tools used by application components. Its [compiler-versioning reference](https://learn.microsoft.com/en-us/cpp/overview/compiler-versions?view=msvc-170) distinguishes compiler numbering from product/toolset numbering. The inspected references do **not** prescribe comparing `cl.exe`'s file/product patch field numerically against each runtime DLL's patch field as a universal test. Therefore `35211 < 35229` alone is not a documented incompatibility verdict. Conversely, sharing the 14.44 prefix does not independently prove every imported API or exact build pairing is supported.

The official [Microsoft lifecycle FAQ](https://learn.microsoft.com/en-us/lifecycle/faq/visual-c-faq) explicitly maps MSVC/v143 14.44 to Redistributable/runtime 14.44.35211, shipped with VS 2022 v17.14 final, and states that serviced releases are required. This confirms that 14.44.35211 is an official runtime version for this build-tools family; it is not a per-file compatibility certificate for the exact compiler/build above. Preserve the distinction. Current defensible result: the package's exact compatibility gate remains open pending a justified runtime selection and actual extracted-package tests; do not invent a strict patch comparison or silently waive the rule.

### Version-pinnable newer acquisition candidate, metadata only

Microsoft's [current supported-download page](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) recommends the latest v14 runtime for supported MSVC builds. Its rolling x64 link `https://aka.ms/vc14/vc_redist.x64.exe` was resolved with HEAD only on 2026-10-16 to the fixed source below. Microsoft's [WinGet package manifest at commit 7d34dbe95888e81befcf61c174a6688fb574be9c](https://github.com/microsoft/winget-pkgs/blob/7d34dbe95888e81befcf61c174a6688fb574be9c/manifests/m/Microsoft/VCRedist/2015%2B/x64/14.51.36247.0/Microsoft.VCRedist.2015%2B.x64.installer.yaml) declares the same URL/version/hash:

| Field | Metadata observation |
| --- | --- |
| Package | `Microsoft.VCRedist.2015+.x64` |
| Declared version/type | `14.51.36247.0`, Burn bundle |
| Fixed publisher URL | [VC_redist.x64.exe](https://download.visualstudio.microsoft.com/download/pr/ebdab8e5-1d7b-4d9f-a11b-cbb1720c3b12/843068991DAAA1F73AD9F6239BCE4D0F6A07A51F18C37EA2A867E9BECA71295C/VC_redist.x64.exe) |
| Manifest-declared SHA256 | `843068991daaa1f73ad9f6239bce4d0f6a07a51f18c37ea2a867e9beca71295c` |
| HEAD Content-Length | `18731856` |
| HEAD Last-Modified | `Fri, 29 May 2026 00:19:00 GMT` |

The SHA is **publisher-repository manifest metadata**, not a locally computed hash; the installer body was not fetched. Future acquisition must verify it and the Microsoft signature before use, retain fixed URL/date/bytes/manifest commit and capture actual product version. The rolling aka.ms link or a SHA-shaped URL path alone is not a byte-verification mechanism. Microsoft's WinGet repository is a primary source for its package-distribution metadata, not evidence that this project has distribution rights.

This newer 14.51 candidate avoids relying on the same 14.44 component-patch ambiguity, under Microsoft's forward binary-compatibility guidance. It does not need a native-source rebuild, but replacing eight runtime dependencies changes the package and requires a new manifest plus actual closure/native/public/lifecycle/coexistence tests. Keep the original successful SDK receipt intact. Newer runtime terms/notices must come from the actual newer distribution; do not relabel the previously inspected 2015–2022 EULA as automatically applicable to 14.51.

### Non-installing inspection routes to implement later

**Preferred when installer execution is prohibited:** read the installer as data with a pinned extractor. WiX's [documented bundle extraction command](https://docs.firegiant.com/wix/tools/wixexe/#wix-burn-extract-subcommand) is:

```text
wix burn extract VC_redist.x64.exe -o payloads -oba bootstrapper
```

This invokes the extraction tool, not the target installer. It explicitly separates attached package containers and bootstrapper application resources: inspect both for actual EULA/notice payloads and the manifest references that select them. The current hosted image also has older WiX tooling; its [pinned 3.14.1 Unbinder source](https://github.com/wixtoolset/wix3/blob/wix3141rtm/src/tools/wix/Unbinder.cs) extracts both UX and attached containers, offering an alternative `dark.exe -x <directory> <bundle>` route after pinning the actual tool/dependency hashes and verifying compatibility with this bundle. This research did not run either extractor or establish the newer bundle's inner filenames. Do not assume the license is a top-level `license.rtf` or that the 2022 DOCX matches it.

The attached payloads are MSI/CAB packages rather than necessarily loose DLLs. Extract/read them without running installation sequences: Microsoft's [MsiOpenDatabase read-only API](https://learn.microsoft.com/en-us/windows/win32/api/msiquery/nf-msiquery-msiopendatabasew), [table export](https://learn.microsoft.com/en-us/windows/win32/api/msiquery/nf-msiquery-msidatabaseexportw) and [cabinet iteration/extraction API](https://learn.microsoft.com/en-us/windows/win32/api/setupapi/nf-setupapi-setupiteratecabinetw) support a data-inspection implementation. A pinned WiX MSI extraction/decompilation path is another option after inspecting its exact outputs. Preserve File/Media/stream mappings so the final CRT DLLs remain traceable to specific MSI/CAB payloads; reject ambiguous matching names or unsupported bundle formats.

**Alternative with non-installing bootstrapper execution:** Microsoft's [documented `/layout` mode](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170) copies complete redistributable contents for a local layout. It is a non-installing deployment operation but still executes the downloaded EXE, can acquire payloads, and is not equivalent to static extraction. It also does not establish that every bootstrapper-internal EULA/notice becomes a loose layout file. No such operation was run here. Likewise `msiexec /a` is an administrative installation according to [Microsoft's command-line reference](https://learn.microsoft.com/en-us/windows/win32/msi/command-line-options), so it should not be described as a zero-installer-execution extractor.

Retain extracted terms/notices byte-for-byte with hashes, exact original bundle/MSI/resource paths and selection metadata. Verify every final DLL's file/product version, AMD64 PE identity, SHA256 and Microsoft Authenticode signature/timestamp; keep Microsoft files unchanged. Keep observed online-host signature status distinct from offline trust validation. License applicability and the separate VS redistribution basis remain explicit open evidence until selected and retained. No newer runtime, extraction output, clean-machine test or compatibility qualification is claimed by this update.
