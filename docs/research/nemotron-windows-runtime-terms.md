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


## Central prerequisite alternative: publisher download, five-DLL Gen2 archive

Research-only deployment choice, 2026-10-16. No installer was run, runtime files were changed, or gate result was promoted. The existing [Windows packaging note](/Users/victor/workspace/gen2-nemotron3-n0/docs/research/nemotron-windows-packaging.md) already permits a central VC Redistributable prerequisite; the variant below distinguishes **linking users to Microsoft** from bundling the installer ourselves.

**Engineering recommendation:** a production candidate can contain only the five private NeMo/GGML DLLs, manifest and required notices for their compiled components, with a declared centrally installed x64 VC runtime prerequisite. Direct the user/administrator to Microsoft's fixed publisher URL for the pinned 14.51.36247.0 candidate above. Keep both the eight VC DLLs and the Microsoft installer out of the Gen2 release archive. In this arrangement Gen2 is not conveying those runtime DLL/installer bytes; the unresolved right to redistribute those particular bytes is not being asserted as the package's distribution basis. This is a scope observation, not a conclusion that every other licensing obligation or entitlement is established.

Microsoft explicitly recommends installing the VC Redistributable as a prerequisite and central servicing, while its redistribution restrictions apply to conveying runtime packages/individual binaries. [Microsoft deployment guidance](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170). The user's installation/use of software obtained from Microsoft is governed by that distribution's applicable runtime terms; it is separate from a VS license's permission to redistribute files. Preserve the prior distinction: the inspected 2015–2022 EULA grants installation/use, not onward supply, and its applicability to a 14.51 distribution has not been established. Do not claim a public link supplies distributor entitlement, or that the old EULA automatically applies to the newer download. An administrator deploying silently must have the appropriate basis to perform that setup; this research does not establish it.

| Choice | Practical implication |
| --- | --- |
| Five native DLLs + publisher-hosted central prerequisite | Smaller Gen2 archive; external prerequisite setup/servicing; compatible with offline inference after setup; no bundled MS runtime payload |
| App-local VC DLL bundle | Portable-directory deployment without central installation, but exact runtime redistribution terms/basis, notices and servicing remain package responsibilities |
| Gen2 bundles VC installer | Still conveys Microsoft bytes and retains redistribution-basis requirements; central installation alone does not remove that issue |

These choices alter the declared delivery contract, not native/quality acceptance criteria. A central-prerequisite release must not be advertised as runnable on bare Windows before setup. The app-local development evidence remains useful for native behavior, but does not prove this central-runtime variant.

### Supported unattended setup and exit handling

Microsoft documents `/install /quiet /norestart /log <path>` for unattended VC setup. `/quiet` suppresses UI/interaction; `/norestart` suppresses automatic restart. The inspected official guidance does **not** define `/quiet` as an independent license-acceptance or entitlement switch. Silent setup is not a way to infer assent or rights from an exit code. Retain the actual acquired terms and the administrator-controlled setup record. [Documented switches](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170#command-line-options-for-the-redistributable-packages).

Future setup invocation, after the operator has selected this prerequisite route and established its applicable terms:

```text
VC_redist.x64.exe /install /quiet /norestart /log C:\evidence\vc-redist-setup.log
```

Launch synchronously and retain the actual process exit code and every referenced package log. Exit 0 is success; 3010 means success **with restart required**, not completed restart. [Microsoft return-code meanings](https://learn.microsoft.com/en-us/windows/win32/msi/error-codes). Microsoft's [container build example](https://learn.microsoft.com/en-us/visualstudio/install/build-tools-container?view=vs-2022) handles 3010 explicitly for Build Tools, demonstrating the container pattern but not proving VC runtime readiness in this particular image. In our qualification record retain `reboot_required` if returned, verify resulting installed files/version/actual model process, and do not claim the reboot requirement was fulfilled. If completing setup requires an actual reboot unavailable in the container route, use a suitable fresh VM for that case. Other nonzero codes fail unless their exact documented condition is separately verified. In particular a newer pre-existing VC runtime can make this installer return an error; detect the actual x64 installed version rather than treating every “already installed” failure as success.

### Actual fresh Server Core test still required

Microsoft's own [IIS Server Core sample](https://github.com/microsoft/iis-docker/blob/main/samples/php/Dockerfile) installs VC Redistributable silently before its application, providing a primary implementation precedent. It does not qualify 14.51.36247.0 for Gen2. Use a digest-pinned Server Core LTSC2022 base and compatible process-isolation host, as required by the [Windows container compatibility matrix](https://learn.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/version-compatibility).

1. Establish the base image's initial VC state. Keep VS/vcpkg/build tools out. Acquire the exact installer directly from Microsoft's fixed URL in the setup stage, verify the declared SHA and Microsoft signature, and retain actual accompanying terms/notices. Record whether the base already contains any relevant runtime, rather than calling it pristine by assumption.
2. Perform the explicitly selected prerequisite installation in the setup stage, capture logs/exit status and registry/file versions, and freeze the resulting test image identity. Installation can precede the network-disabled inference phase. Keep this internal qualification image distinct from the five-DLL production archive; publishing it introduces its own Windows/MS runtime distribution terms.
3. Launch a **new container** from that prepared image with networking disabled and readonly frozen package/model/fixture inputs. Assert the actual network state in CI. A setup network connection does not invalidate a separately controlled offline inference phase, but the receipt must describe both stages clearly and must not claim the prerequisite itself was acquired offline.
4. Place only the five private native DLLs in the release directory: no app-local Microsoft CRTs, original SDK, compiler PATH or dependency caches. Run the same frozen native/public comparison, lifecycle and chat-coexistence gates. Capture loaded CRT paths/version/signatures from the central system installation and private native paths/hashes. Run from a relocated Unicode/space-containing path; test a missing private DLL fails instead of finding another SDK.
5. Record the exact tested central runtime and fail on unexpected fallback. The candidate's metadata pin does not prove actual installer content, installation, closure, API compatibility or offline behavior. Client Windows support and pre-existing newer-runtime variants need their own stated evidence scope.

Selection of this route could remove Microsoft-runtime **redistribution** from Gen2's archive scope, while leaving prerequisite installation rights/terms, actual compatibility and all existing deployment/quality gates explicit. Nothing in this research closes those gates or changes the ongoing task's acceptance criteria.
