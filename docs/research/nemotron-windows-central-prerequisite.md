# Windows central VC runtime prerequisite: acquired bytes and pending qualification

Research/acquisition, 2026-10-16. The fixed publisher installer was downloaded, hash-verified, and inspected statically with already-installed tools. Its actual UI selects Microsoft's current v14 runtime terms through an external link; that linked official DOCX is now retained with exact metadata. **No installer, extracted EXE/DLL, MSI action, native build, or model was executed.** This supports a proposed production contract containing five private native DLLs and their non-Microsoft component notices, with users/administrators acquiring the runtime directly from Microsoft. It does not establish user/distributor entitlement, Windows signature trust, successful installation, or a qualified central-runtime deployment.

## Actual acquisition

Microsoft's [supported-download guidance](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) identifies the official v14 x64 download family and requires a sufficiently recent runtime matching the application target. The [Microsoft WinGet manifest at commit `7d34dbe95888e81befcf61c174a6688fb574be9c`](https://github.com/microsoft/winget-pkgs/blob/7d34dbe95888e81befcf61c174a6688fb574be9c/manifests/m/Microsoft/VCRedist/2015%2B/x64/14.51.36247.0/Microsoft.VCRedist.2015%2B.x64.installer.yaml) declares the fixed URL/version/hash below. Actual downloaded bytes match it:

| Item | Observed value |
| --- | --- |
| Publisher URL | [VC_redist.x64.exe](https://download.visualstudio.microsoft.com/download/pr/ebdab8e5-1d7b-4d9f-a11b-cbb1720c3b12/843068991DAAA1F73AD9F6239BCE4D0F6A07A51F18C37EA2A867E9BECA71295C/VC_redist.x64.exe) |
| Retrieved UTC | `2026-10-16T11:27:27.994999+00:00` |
| Actual bytes | `18,731,856` |
| Actual SHA256 | `843068991daaa1f73ad9f6239bce4d0f6a07a51f18c37ea2a867e9beca71295c` |
| Static version evidence | Burn registration and PE version data report `14.51.36247.0` |
| Signature status | Not validated; static PE certificate region only |

The [installer cache](/Users/victor/.cache/gen2-nemotron3-n0/research/windows-central-prerequisite/VC_redist.x64.exe) and [raw acquisition metadata](/Users/victor/.cache/gen2-nemotron3-n0/research/windows-central-prerequisite/acquisition.json) retain exact source, response headers and hashes outside Git. The installer bootstrapper is an x86 PE; its name is not a PE-machine assertion. The actual manifest selects two x64 MSI packages on non-ARM64 64-bit Windows and a separate ARM64 MSI on ARM64. This agrees with Microsoft's documented combined x64/ARM64 package layout. No target compatibility was tested here.

## Static extraction and actual payload layout

Already-installed `/opt/homebrew/bin/7z` identified itself as p7zip 17.05, installed under Homebrew's `p7zip/17.06`. No extractor was downloaded or installed. Its actual executable SHA256 is `32ab345a0e237e4bc5c6cde591d50c805168dc7d0d10dcedc1bb423469ef66f0`; its `7z.so` is `e2a563a1c37d0ac4b32d2ff92020d694c81461fa7b7696de5c472d50b6d212ad`. The wrapper, executable/library hashes and complete tool output are retained. These record an existing local inspection tool; they do not imply the tool or extraction has independent Windows qualification. [7-Zip's primary feature list](https://www.7-zip.org/) includes CAB/Compound/PE extraction. WiX's own [bundle extraction documentation](https://docs.firegiant.com/wix/tools/wixexe/#wix-burn-extract-subcommand) likewise distinguishes attached packages from bootstrapper resources.

`7z x` statically opened the PE's first CAB as UX resources. `ux/0` is the Burn manifest (9,259 bytes, SHA256 `e82e62b82a1b09568e74a224c454bdcd41240b9f24cfc42fb2c5e0783345af7f`); `ux/u17` maps to `BootstrapperApplicationData.xml` (7,694 bytes, SHA256 `0af0673e1f3ca87701e3c00d5ff004fccb00e2a983df1d72671c429cfef7f3c5`). The manifest declares an attached container of 18,091,661 bytes with SHA1 `a0c032f1194ef5bd77e767149b5a8777670f092a`. A bounded search for CAB signatures found exactly one slice matching that declared length/hash, at installer offset `630000`. It was copied unchanged and extracted as data. CAB signatures/length fields are described by Microsoft's [CAB format specification](https://download.microsoft.com/download/4/D/A/4DA14F27-B4EF-4170-A6E6-5B1EF85B1BAA/%5BMS-CAB%5D.pdf); no custom decompressor or MSI execution was used.

All **24** declared UX/attached payloads have actual sizes and SHA1 values matching their hash-bound Burn manifest declarations. The six attached payloads are:

| Container entry | Manifest logical path | Bytes |
| --- | --- | ---: |
| `a0` | `packages\VC_Runtime_arm64\VC_Runtime_arm64.msi` | 212,992 |
| `a1` | `packages\vcRuntimeMinimum_amd64\vc_runtimeMinimum_x64.msi` | 212,992 |
| `a2` | `packages\vcRuntimeAdditional_amd64\vc_runtimeAdditional_x64.msi` | 212,992 |
| `a3` | `packages\VC_Runtime_arm64\cab1.cab` | 11,169,681 |
| `a4` | `packages\vcRuntimeMinimum_amd64\cab1.cab` | 1,065,893 |
| `a5` | `packages\vcRuntimeAdditional_amd64\cab1.cab` | 5,648,700 |

The other 18 payloads are the bootstrapper DLL, theme/image/localizations and bootstrapper XML. The [static inspection receipt](/Users/victor/.cache/gen2-nemotron3-n0/research/windows-central-prerequisite/static-inspection.json) records all size/hash comparisons and the manifest-to-logical-path mapping. `static-layout/` contains byte-identical copies under those logical paths; this is an inspection layout, **not** a claim that installer `/layout` was executed. The minimum MSI's Compound stream directory was listed; no MSI File/Media table decoding or installed-path mapping was performed. The later minimum-CAB update below adds static DLL extraction/identities/versions/imports, with signatures still unverified. No EULA payload appears in the complete UX or attached manifest inventory; this is narrower than a claim that no additional notice exists in any inner MSI/CAB/resource.

Static `7z -tPE` also exposed a 10,192-byte `CERTIFICATE` region at installer offset `18721664`, retained unchanged with SHA256 `7842dae999a74aa7f9c0512a1ca438bd50095c4969d35173bce76985ffffb633`. Microsoft's [PE specification](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format) distinguishes certificate data from verifying its statements. Presence is not Authenticode image-digest, signer, timestamp, certificate-chain, revocation or Windows policy verification. No such trust checks were performed on macOS; the next Windows host must perform them before execution.

## Actual minimum x64 CAB inspection — 2026-10-16

The actual `attached/a4` minimum x64 CAB is 1,065,893 bytes, SHA256 `6e8ee73933678c55973e83366e48e948b394c7a36743a36b7224c95de979b13f`. Its size/SHA1 already match the Burn payload declaration. Existing `7z x` extracted all 12 CAB members as data into external-cache `minimum-cab/`; the [new inspection metadata](/Users/victor/.cache/gen2-nemotron3-n0/research/windows-central-prerequisite/minimum-x64-cab-inspection.json), SHA256 `af370d9bcd0aad52eab06e459b5e310ae75a740888f0819cba47c149217ceade`, records all 12 member byte hashes and detailed PE results for the eight names from `package_windows.CRT` plus the chat OpenMP candidate `vcomp140.dll`. The remaining CAB members are `vcamp140.dll_amd64`, `vccorlib140.dll_amd64`, and `vcruntime140_threads.dll_amd64`; only their byte inventory was retained in this scoped inspection.

All nine selected files are statically identified as AMD64/PE32+ DLLs, with the expected DLL export name. Both static `FileVersion` and `ProductVersion` resource strings reported by existing `7z -tPE` are **14.51.36247.0** for every selected file. These are parsed resource strings, not observations from the Windows version API. No selected file has delay imports, export-forwarder dependencies, or reported bound imports. The existing repository PE helper was executed only as Python byte-parsing code; no DLL was loaded. Its exact retained source SHA256 is `d194378b4a666444c350f4f4896debcee649568714ae51fc7797844f1cf4d52c`; the `package_windows.py` input defining the eight-name set is `9e4d30d3acbd0ef21a73dff51d0f5f81ca93bea8560457524d9fcda3f3f27248`. Copies of both inputs, the research driver, and per-DLL PE listing logs are retained outside Git and bound by the refreshed cache manifest. Microsoft's [PE format specification](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format) is the primary basis for the static architecture/import/export interpretation.

| CAB logical member | Raw bytes | SHA256 |
| --- | ---: | --- |
| `concrt140.dll_amd64` | 374,200 | `54716f0738af891f283d213b5c8d11b25896bb8ee3097d301eae718560cf974e` |
| `msvcp140.dll_amd64` | 643,512 | `7c26614e1d733892c2deac7e245ce115504b1d80592dd0a01b08e3e5a55f89ca` |
| `msvcp140_1.dll_amd64` | 35,768 | `206c931bf90fdad8816de3b5e2ef80b2bcaa9406c89ecc05fe6fddffe251e982` |
| `msvcp140_2.dll_amd64` | 274,872 | `d50d7883f20d1dc6191768d3746f52dd9cac89c346ffaed5be1f110c2f34a838` |
| `msvcp140_atomic_wait.dll_amd64` | 57,792 | `3d0cbfaa1bf3eecf5a3f4491d2960ee803cb994f30292c6adc4a07c498f60e2b` |
| `msvcp140_codecvt_ids.dll_amd64` | 31,160 | `8a65c7596ef2e6938731f5a1058e7e40145b6d97967cc649231a076b9a608d78` |
| `vcomp140.dll_amd64` | 212,920 | `95d4ce4a6802d1e18b5e0e1722cc30ea72ca7e033f83828f05c0b7b993fe7cbf` |
| `vcruntime140.dll_amd64` | 178,616 | `d1f4225df2cd877dbf130d5668a021dce3f94118455ff5ec952061c30afc9ce7` |
| `vcruntime140_1.dll_amd64` | 50,112 | `a7146c08f89fe5b04541ab507cdb59ff7b44534d4ba3c668a426c6450a03434e` |

`vcomp140.dll_amd64` is the static OpenMP candidate: its PE name is `VCOMP140.DLL`, its resource description identifies the Microsoft C/C++ OpenMP runtime, and its only static import is `kernel32.dll`. This demonstrates that the fixed publisher bundle contains an appropriate named AMD64 runtime payload for the chat fixture's OpenMP dependency; it does not yet prove where setup installs it or that chat successfully loads it. Preserve that distinction in the future central-runtime receipt.

The full ordinary import lists are in the JSON. The selected CRTs import one another plus kernel32/advapi32 and CRT API-set contracts. A concrete change from the older app-local package assumptions is `msvcp140_atomic_wait.dll` importing **`api-ms-win-core-synch-l1-2-0.dll`**. That contract is absent from the currently observed `package_windows.PLATFORM` allowlist. Do not silently reuse that old allowlist or mistake a static API-set contract name for a missing app-local DLL: the central variant needs an explicitly reviewed target-platform policy and actual Windows loader resolution evidence. Static inspection did not resolve API sets, dynamic LoadLibrary calls, or transitive Windows platform dependencies.

**Mapping and trust remain pending:** CAB member names such as `vcomp140.dll_amd64` are logical file identifiers. They are not verified installed basenames or System32 paths. Decode the matched MSI File/Component/Directory/Media mapping or capture actual setup and loaded-module evidence before making those claims. No selected DLL's Authenticode digest, signer, timestamp, chain/revocation, or Windows policy has been verified here. The future disposable Windows run should compare central installed/loaded file hashes with these payload hashes where that mapping is established, record any difference rather than normalize it, and retain actual System32 paths. All 12 extracted DLL bodies remain outside Git; this doc and manifest preserve metadata only. The same five-private-DLL release contract remains unchanged.

## Actual bundle-selected terms

The extracted `BootstrapperApplicationData.xml` contains `WixStdbaInformation LicenseUrl="https://aka.ms/VCRedistLicense"`. The English localization (`ux/u2`, 3,088 bytes, SHA256 `21a76eb702fe727ca06de6632be2294ed1caa9e1d3798d01ed1428e0f40e19ea`) directs users to the same address and separately provides the agreement-checkbox label. This actual installer association replaces the earlier uncertainty about using an unrelated public 2015–2022 reference EULA.

On `2026-10-16T11:28:01.246976+00:00`, [the selected link](https://aka.ms/VCRedistLicense) resolved to Microsoft's [Visual C++ V14 Redistributable and Runtime 2026 terms page](https://visualstudio.microsoft.com/license-terms/vs2026-ga-visualcpp-v14-redist-runtime/). Its embedded Office viewer selects this exact [official DOCX](https://visualstudio.microsoft.com/wp-content/uploads/2025/10/Visual-C-V14-License-Redistributable_and_Runtime_ENU.docx):

| Item | Actual retained value |
| --- | --- |
| DOCX retrieval UTC | `2026-10-16T11:28:21.838079+00:00` |
| Raw DOCX bytes / SHA256 | `39,553` / `08651651a7602fc7c0e2763de0fde1ff9f868df2780597cd1775ee9d6441c783` |
| Internal title/date/ID | Microsoft Visual C++ V14 Redistributable and Runtime; October 1, 2025; `Cpp_v14_ENU.1033` |
| Derived UTF-8 text bytes / SHA256 | `11,894` / `923d8b11a03c277d59f3262727abc7e422158f5243a5565d67d5c270c6b05e3c` |
| Landing-page raw bytes / SHA256 | `236,346` / `4b729a5fc1b8c029a0456bb04c341a69fa3c7575a379f0308bf8a70604d7730d` |

The terms permit installation/use of any number of copies subject to their conditions. They tie acceptance to use, refer to third-party notices, reserve other rights and restrict onward distribution. They also say terms may change and direct users to current terms. This is observed current publisher-selected terms provenance, not a warranty of historical or future applicability, evidence of anyone's assent, or a redistribution grant. The separate VS licensed-user REDIST permission is not established by acquiring this document. Keeping the installer/CRT bytes out of the Gen2 release archive avoids asserting that unresolved grant for those excluded bytes; it does not infer a user's rights. See [the retained terms metadata](/Users/victor/.cache/gen2-nemotron3-n0/research/windows-central-prerequisite/runtime-terms.metadata.json), [raw DOCX](/Users/victor/.cache/gen2-nemotron3-n0/research/windows-central-prerequisite/Visual-C-V14-License-Redistributable_and_Runtime_ENU.docx), and [labeled text extraction](/Users/victor/.cache/gen2-nemotron3-n0/research/windows-central-prerequisite/runtime-terms-derived.txt). The DOCX bytes are authoritative; the XML-derived text is for reading.

## Reproduction without installer execution

Use the acquired hash-verified installer and the existing recorded `7z` only as a data extractor. In the external cache directory, the exact extraction route is:

```sh
7z l -slt VC_redist.x64.exe
7z x -y -oux VC_redist.x64.exe
python3 verify_static_bundle.py --carve-only
7z x -y -oattached attached-container.cab
7z l -slt -tPE VC_redist.x64.exe
7z x -y -tPE -ope-data VC_redist.x64.exe CERTIFICATE
python3 verify_static_bundle.py
7z x -y -ominimum-cab attached/a4
PYTHONDONTWRITEBYTECODE=1 python3 inspect_minimum_cab.py
```

The retained script first enforces installer size/SHA256, parses only the extracted manifest, requires a unique container slice matching its declared size/SHA1, and then validates every extracted payload before making logical-path copies. It executes no payload. Follow the actual UX's license URL, retain redirect destination/landing HTML, read the viewer's `src` parameter, download that Microsoft DOCX, hash its raw bytes, and extract `word/document.xml` text only as a labeled derivative. Retain retrieval time because the link and document can change. The [cache manifest](/Users/victor/.cache/gen2-nemotron3-n0/research/windows-central-prerequisite/manifest.json), SHA256 `b3e3270200adbf1ddc077ae8ec5fc67a0f1ef1c74623038e3c2c27675784c383`, records acquisition/inspection/terms metadata and script/log hashes. The full installer, extracted binary payloads, original terms DOCX/full derived terms and complete landing HTML remain outside Git. The subsequent static receipt below retains reviewed metadata, inspection scripts and logs.

## Concrete next Windows qualification — proposed, not performed

1. Start from a digest-pinned, fresh Server Core image on a compatible Windows process-container host. Record base/host identity and runtime baseline before provisioning. Fetch only the fixed publisher URL; require the exact installer size/SHA256 above. Before execution, use Windows [Get-AuthenticodeSignature](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.security/get-authenticodesignature) or equivalent Windows trust APIs; require the selected policy's valid Microsoft signer result and retain status, signer/timestamp certificate identities, trust context/time and hash. Keep this actual host trust result separate from the macOS static certificate observation.
2. Provision the central prerequisite only inside the disposable qualification image under the administrator's applicable terms. Wait for `VC_redist.x64.exe /install /quiet /norestart /log C:\evidence\vc-redist-install.log` and retain process exit plus bootstrapper/MSI logs. Microsoft [documents these unattended switches](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170#command-line-options-for-the-redistributable-packages); `/quiet` controls UI, not a separate rights grant. The actual selected terms' use-based acceptance language must be considered separately.
3. Treat exit `0` as success subject to resulting-state checks. Preserve `3010` as **successful setup requiring restart**, not a reboot-complete pass. A container recreation alone is not evidence that a requested Windows reboot occurred. If the protocol cannot satisfy an actual restart requirement, leave that result unqualified. Preserve unexpected codes and logs; do not turn `1638`/a newer pre-existing version into success by default. Microsoft's [installer status definitions](https://learn.microsoft.com/en-us/windows/win32/msi/error-codes) distinguish these outcomes. Freeze one explicit acceptable-state policy before the run.
4. Capture pre/post x64 runtime registry values (`Installed`, `Version`, `Major`, `Minor`, `Bld`, `Rbld` as present), with explicit 32/64-bit registry views; do not depend on one Wow6432Node path guess. Also capture the actual required `%windir%\System32` CRT files' paths, bytes/SHA256, AMD64 PE identity, file/product versions and signature observations. Preserve relevant package registration and pending-restart evidence. Registry version alone is insufficient to prove loaded bytes. Enumerate the runtime requirement from the five native DLLs' actual import graph, including transitive CRT imports.
5. Freeze the provisioned image identity; launch a fresh qualification container with `--network none`, locked model/fixtures already available, five private native DLLs and non-Microsoft notices, and **no app-local CRT copies**. Record container network configuration and actual loaded CRT module paths/hashes under central System32. Run the existing public/native/lifecycle/coexistence gates and missing-runtime/basename-conflict controls. Verify the package still contains only its declared five DLLs. Keep image/tool/setup provenance internal to qualification; publishing a Windows image or bundling Microsoft's installer is a separate distribution scope. Only actual results can close this variant's deployment gate.

These instructions progress the central-prerequisite contract; they do not change the roadmap's quality or deployment gates. A successful app-local development package does not qualify this central variant.

## Subsequent receiver and construction checkpoints

The [30-file static receiver receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-central-prerequisite-static/qualification.json) independently starts from the exact installer, re-extracts UX and the manifest-hash-matched attached cabinet, rechecks all 24 payloads and 12 minimum-CAB files, recomputes all nine selected PE identities/import reports and static resource versions, verifies the actual selected license URL and recomputes the retained DOCX text derivative. It does not execute any installer/payload or establish Windows trust/MSI destinations/runtime behavior. The recorded `package_windows-input-copy.py` is an exact inspection-time snapshot from the evolving worktree, used only to extract the eight-name CRT constant via Python AST; it is not asserted to be an immutable native-build source revision. The actual PE inspector hash is separately retained and checked.

The [19-file central-construction receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-central-prerequisite-construction/qualification.json), source `39d477f`, verifies an actual 39-entry archive and fresh Unicode restoration on macOS. It carries five unchanged qualified private native DLLs, all eight original upstream notices, three supplemental notice texts and exact prerequisite metadata, with no Microsoft installer or CRT DLL payload. All 90 model-free tool tests pass. The app-local 13-DLL format remains distinct. This is byte construction only: the proposed Windows signature/setup/state/module/real-model/coexistence sequence above is still pending for the central variant.
