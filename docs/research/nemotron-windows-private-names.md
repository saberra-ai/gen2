# Windows native DLL names: build-only hook and qualification infrastructure

Read-only primary-source investigation, 2026-10-16. The following hook is proposed source for implementation review; it was not configured, built or executed. No model/package downloads, installs or model runs were performed. Existing [Windows evidence/status](/Users/victor/workspace/gen2-nemotron3-n0/docs/research/nemotron-windows-packaging.md) remains unchanged by this research.

## Concrete naming integration

The pinned [NeMo ASR CMake file](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/asr/CMakeLists.txt) creates real SHARED targets `nemo_speech_asr` and `nemo_speech_asr_c`; the latter links the former by logical target name. Both use `install(TARGETS ...)`, and the C ABI target installs its Windows import library under the configured lib directory. The pinned [GGML target definitions](https://github.com/ggml-org/llama.cpp/blob/bd4f514db14d87fded667787a7a963bfbaa98e89/ggml/src/CMakeLists.txt) create `ggml-base`, `ggml` and, through `ggml_add_backend_library`, `ggml-cpu`; they link each other as targets. [CPU creation](https://github.com/ggml-org/llama.cpp/blob/bd4f514db14d87fded667787a7a963bfbaa98e89/ggml/src/ggml-cpu/CMakeLists.txt), [GGML install rules](https://github.com/ggml-org/llama.cpp/blob/bd4f514db14d87fded667787a7a963bfbaa98e89/ggml/CMakeLists.txt). The recorded CPU profile has BUILD_SHARED_LIBS ON and GGML_BACKEND_DL OFF; reject any configuration violating those assumptions.

Use `-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES:FILEPATH=<absolute audited hook>` through the existing `-CMakeArgs` list in [platform_native.py](/Users/victor/workspace/gen2-nemotron3-n0/tools/nemotron-diarization/platform_native.py). This preserves pristine upstream source and the existing separately recorded LF compatibility overlay. CMake includes this user hook early in the first `project()`, when targets do not exist yet. Define a constant no-argument function and defer it until the root directory finishes. Do not set target properties immediately in the early include. CMake documents both [top-level inclusion timing](https://cmake.org/cmake/help/v3.31/variable/CMAKE_PROJECT_TOP_LEVEL_INCLUDES.html) and [deferred end-directory execution/late argument expansion](https://cmake.org/cmake/help/v3.31/command/cmake_language.html#deferring-calls).

Proposed hook body (LF UTF-8; final implementation must freeze its actual bytes/hash):

```cmake
include_guard(GLOBAL)
if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Gen2 native naming hook must enter at the root")
endif()
function(gen2_private_nemo_windows_names)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows" OR
       NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR
       NOT CMAKE_BUILD_TYPE STREQUAL "Release" OR
       NOT BUILD_SHARED_LIBS OR GGML_BACKEND_DL OR GGML_OPENMP OR
       GGML_CUDA OR GGML_VULKAN)
        message(FATAL_ERROR "Unsupported Gen2 private Windows CPU profile")
    endif()
    foreach(target IN ITEMS nemo_speech_asr_c nemo_speech_asr
                            ggml ggml-base ggml-cpu)
        if(NOT TARGET "${target}")
            message(FATAL_ERROR "Missing expected native target: ${target}")
        endif()
        get_target_property(kind "${target}" TYPE)
        get_target_property(imported "${target}" IMPORTED)
        get_target_property(alias "${target}" ALIASED_TARGET)
        if(NOT kind STREQUAL "SHARED_LIBRARY" OR imported OR alias)
            message(FATAL_ERROR "Unexpected native target kind: ${target}")
        endif()
        # Refuse overrides that would supersede the generic OUTPUT_NAME.
        foreach(property IN ITEMS RUNTIME_OUTPUT_NAME ARCHIVE_OUTPUT_NAME
                                  LIBRARY_OUTPUT_NAME OUTPUT_NAME_RELEASE
                                  RUNTIME_OUTPUT_NAME_RELEASE
                                  ARCHIVE_OUTPUT_NAME_RELEASE
                                  LIBRARY_OUTPUT_NAME_RELEASE)
            get_property(present TARGET "${target}" PROPERTY "${property}" SET)
            if(present)
                message(FATAL_ERROR "Unexpected name override: ${target}/${property}")
            endif()
        endforeach()
        string(REGEX REPLACE "^nemo_speech_" "" stem "${target}")
        set_target_properties("${target}" PROPERTIES OUTPUT_NAME "gen2_nemo_${stem}")
    endforeach()
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/gen2-native-paths-$<CONFIG>.txt"
         CONTENT "entry=$<TARGET_FILE:nemo_speech_asr_c>\nimport=$<TARGET_LINKER_FILE:nemo_speech_asr_c>\n")
endfunction()
cmake_language(DEFER CALL gen2_private_nemo_windows_names)
```

This deliberately supports the current single-config Ninja Release path; a future multi-config build needs an explicit reviewed adaptation. Keep any pre-existing top-level includes explicitly declared and audited; do not silently replace a hook list or allow arbitrary extra hooks. Do not propagate this include into compiler try-compile projects. Check the final generated paths and import graph rather than treating a successful configure as proof.

[CMake OUTPUT_NAME](https://cmake.org/cmake/help/latest/prop_tgt/OUTPUT_NAME.html) defines the physical output basename; target identity remains unchanged. Runtime/archive/per-configuration names can override it, hence the guards. [Runtime-specific naming](https://cmake.org/cmake/help/v3.31/prop_tgt/RUNTIME_OUTPUT_NAME.html), [archive-specific naming](https://cmake.org/cmake/help/latest/prop_tgt/ARCHIVE_OUTPUT_NAME.html). Expected Release payload is `gen2_nemo_asr_c.dll`, `gen2_nemo_asr.dll`, `gen2_nemo_ggml.dll`, `gen2_nemo_ggml-base.dll`, `gen2_nemo_ggml-cpu.dll`. CMake/MSVC should generate corresponding import libraries and imports during linking; inspect them and generated install scripts to confirm this unexecuted proposal. The C ABI import library becomes `lib/gen2_nemo_asr_c.lib`; the independent probe's hard-coded `nemo_speech_asr_c.lib` argument must change to the installed new name. Existing CMake-linked examples need no logical-target rename. The generated TARGET_LINKER_FILE record locates the build-tree import library; do not confuse it with its installed location.

The qualifier/packager's configured entry path must use `bin/gen2_nemo_asr_c.dll`, including negative controls. Do not restore generic native DLL aliases beside it. Keep eight Microsoft CRT names/bytes unchanged. Re-run SDK/probe/public lifecycle and both chat load orders after relinking: the binary identity has changed even though inference sources/ABI exports have not. Check all PE imports/delay imports/forwarded exports for native generic names; check the public export set and actual loaded paths. GGML dynamic backends remain OFF. This naming change closes a basename collision mechanism only after those checks; it is not qualification by itself.

## Exact original-source guards

Before configuring, verify the two Git revisions and these SHA256s against actual bytes, not just names or newline-normalized text. Values below were read from the clean pinned local source; the three GGML inputs were also compared with immutable `git show` objects. Guard originals before preparing the LF overlay/staging tree; retain the existing 19 patch hashes and record the resulting prepared source separately.

| Input under NeMo source root | SHA256 |
| --- | --- |
| `CMakeLists.txt` | `e79e24832e52b440808dd59a4b274ad657ed6e7bb42df992dcaf0ee87f9d39fa` |
| `src/asr/CMakeLists.txt` | `7cf799a2a8f98de94e749bbdf72bab5cabba8dd61921b78ac9be6fa2f8850e48` |
| `scripts/windows/build.ps1` | `068c58cceea80cc6b021ec09e60f0caea1241ab2963ba3ffe1a5f7e3b547c68f` |
| `vcpkg.json` | `3ba4dc4a35bd4cd7958eaa5525714f231ab1729515d05600fee94fcb037b1992` |
| `cmake/llama_cpp.cmake` (original) | `22357d4a5d8a67716daa8b4e374abd63120432bd57540f97d572a525d5354c7b` |
| `llama.cpp/ggml/CMakeLists.txt` | `c118116572c8e5bc8d9302d44cc6806ce69a31face60f2cefdb36801fdf72fd3` |
| `llama.cpp/ggml/src/CMakeLists.txt` | `38ddf053a1adfc14bb84b7722e3c4ec01909694f7e51e32013f36e7a324fa4f0` |
| `llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt` | `ed4d3cf43377accac79bd09b4b93c10725e66f189e1ddfcd528c7c29508cf3d0` |

Preserve the LF overlay's original/updated hashes separately as already recorded; this external naming hook must not rewrite those identities. Add hook SHA256, exact configure arguments, generated target path record, final PE hashes/imports and installation inventory to the new receipt. Reject source/config drift rather than expanding the target map automatically.

## Actual redist provenance: present and missing

The [ad9 cache](/Users/victor/workspace/gen2-nemotron3-n0/benches/results/nemotron-diarization/windows-x86_64/2026-10-16-native-loader-control/cmake-cache.log) records `MSVC_REDIST_DIR=C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Redist/MSVC/14.44.35112`. Its [install log](/Users/victor/workspace/gen2-nemotron3-n0/benches/results/nemotron-diarization/windows-x86_64/2026-10-16-native-loader-control/sdk-install.stdout.log) records the eight CRT destination files. The earlier [runner log](/Users/victor/workspace/gen2-nemotron3-n0/benches/results/nemotron-diarization/windows-x86_64/2026-10-16-probe-build/38873722221/github-job.log:17) identifies Server 2022 Datacenter, image `20261012.333.1` and its [published image inventory](https://github.com/actions/runner-images/blob/win22/20261012.333/images/windows/Windows2022-Readme.md), which lists VS Enterprise 2022 `17.14.37710.0` and its installation root. These are useful provenance anchors, not a retained Microsoft license/notice file or exact per-DLL redist source path.

No inspected receipt/log captures an actual VS/redist license file path/content/hash, Microsoft signature checks or source-to-installed CRT byte comparisons. The build log's “See LICENSE.txt” refers to vcpkg bootstrap context, not proof of a Microsoft VC redistribution license. The next packaging controller should resolve each permitted CRT file within the recorded redist root (normally an x64 CRT subdirectory), record the exact real path/version/hash and compare it with installed bytes; require applicable license/notice input with recorded origin/hash rather than guess a nonexistent universal EULA path. Keep the existing [Microsoft REDIST/terms gate](https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution); current evidence does not close entitlement or license capture.

## Hosted Windows feasibility

**Windows Sandbox is not the supported route on this `windows-2022` runner.** Microsoft's [Sandbox applicability and edition list](https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/) covers Windows 10/11 client editions; the actual host is Server 2022. The earlier general suggestion of Sandbox applies to an eligible separate client host, not this Server CI image.

**Nested Hyper-V is unproven, not established impossible.** GitHub documents standard hosted Windows jobs as fresh Azure VMs with administrative access, but the [inspected runner documentation](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) does not promise Windows nested-virtualization extensions or a controllable parent hypervisor. Microsoft's [nested Hyper-V requirements](https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/enable-nested-virtualization) require virtualization exposure configured on the parent host. Admin inside the hosted guest does not establish that control. No retained run probes Hyper-V capability/feature state; do not spend a qualification attempt assuming it is available or call a hosted-runner job itself a clean machine when VS/CRTs remain preinstalled.

A plausible narrower CI route is a **process-isolated Windows Server Core LTSC2022 container**, with immutable image digest, no VS/vcpkg, disabled container network and readonly artifact/assets. The actual published runner image lists Docker, and Microsoft's [compatibility matrix](https://learn.microsoft.com/en-us/virtualization/windowscontainers/deploy-containers/version-compatibility) supports Server 2022 process containers on Server 2022 without requiring Hyper-V isolation. Docker daemon/container-mode availability, the chosen base-image digest and test executable/API compatibility still need a bounded CI preflight; none was run here. This route would qualify the declared Server Core environment, not a Windows 11 desktop machine. For desktop-wide claims use a separately controlled fresh eligible client VM/Sandbox or self-hosted hypervisor with pinned OS provenance. Package extraction, actual module-path capture and frozen model/lifecycle/coexistence checks remain required in either route.

## Implementation checkpoint after this investigation

`platform_native.py --windows-private-dlls` now prepares a guarded external
hook, checks all eight original-source hashes before the LF compatibility
change, and uses private DLL/import-library paths for the probe and public
loader. The implementation uses the explicit basename prefix
`gen2_nemo_windows_` on each unchanged logical target (for example,
`gen2_nemo_windows_nemo_speech_asr_c.dll`). These are the authoritative names
in the implementation; the shorter names in the proposal above are illustrative.
The hook requires the single-config Windows x64 CPU Release profile and rejects
existing name overrides. The new bounded, read-only `windows_pe.py` inspector
checks AMD64/PE32+, ordinary imports, RVA delay imports and ordinal/named
export forwarders; unsupported legacy delay pointers and malformed or ambiguous
file mappings fail. Bound imports are reported and rejected by this qualification.
Every delivered native DLL and qualification executable is checked for residual
generic NeMo imports; all five private export-directory identities and the installed public C ABI
import library are required. Actual runtime calls and chat coexistence remain separate.

A real CMake target/install generation test on macOS and synthetic malformed-PE
regressions pass. The [read-only inspection](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-pe-inspection/static-inspection.json) of the earlier `4bd34aa` Windows payload also
parses all thirteen DLLs and both executables, with no delay/bound imports or
forwarders observed. That static C ABI dependency closure reaches the five native
DLLs and `msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`; its other imports
name Windows/UCRT contracts. This does not establish a minimum safe runtime
payload because arbitrary dynamic loads are outside static inspection. The new
private Windows build has not yet executed; no private package, provenance,
offline closure or coexistence pass is claimed by these source tests.

The private-build workflow additionally captures `windows-provenance.json`:
actual x64 redist source and installed byte equality, file/product versions,
Authenticode signer/timestamp observations, actual compiler/OS/CPU identity and
vcpkg installed status/metadata hashes. It lists only license candidates actually
found under the recorded redist root and keeps redistribution qualification
false. This read-only PowerShell collector awaits actual Windows execution;
license selection, entitlement and corresponding-source closure remain open.

The separate `diarization-windows-preflight.yml` probe now observes the actual
Server 2022 build VM's OS/CPU/service identity and existing default-context Docker
version, engine information and local image inventory. It does not pull images,
change daemon/features, create containers, install tools or execute models. Each
observation is bounded to 30 seconds, endpoint/context environment overrides
prevent Docker queries, and only selected non-secret fields are retained. A
candidate result requires kernel build 20348 and an x64 Windows engine;
`runtime_qualified` remains false. The actual [read-only preflight](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-container-prerequisites/qualification.json)
passed in run 38882586356 at `5ed163d`: all four observational commands succeeded,
with a local Windows Docker 29.7.2 process engine on kernel build 20348 and
preloaded Server Core/Nano Server images. No container was created in that run. Even a
positive result establishes only prerequisites for a later fresh offline container
run, not desktop support or usable nested Hyper-V.

The first actual private build at `564dcc7` linked and installed all five private
DLLs and linked the independent probe successfully. Its checker then incorrectly
required all five internal target import libraries to be installed; upstream
installs only the public C ABI import library. The [retained failed gate](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-private-import-library-control/qualification.json)
stops before provenance capture, Gen2/model execution or lifecycle checks. The
corrected checker now requires only the installed public C ABI library while
retaining all five DLL/import/export checks. Its new fixture fails the old check
and passes the correction. A read-only macOS receiver rehashed all 66 delivered
files, checked all fifteen PE files and confirmed all five native export sets
are unchanged from `4bd34aa`, with no generic native imports. That receiver did
not execute Windows code; actual Windows requalification is pending.

A separate controls tool now pins the observed Server Core digest and image ID
in `windows-container.lock.json` and requires it already present (`--pull never`).
It creates only its own disposable process container with network none, read-only
Unicode-path control inputs and a writable result folder. It freezes image/source/
tool/input hashes before creation, requires actual input read success plus write
denial with Win32 error 5, no operational non-loopback interface and no expected
development commands/Visual Studio directory, then checks actual Docker mount
sources/flags, isolation, image and exit status. The managed container is removed
and input inventory must stay identical. This implementation awaits actual
execution and tests infrastructure controls only; it contains no Gen2/native/model/
package payload and keeps runtime qualification false. It is separate from desktop
machine qualification. The next workflow preserves LF before checkout, and Docker
JSON is explicitly decoded as UTF-8 so the Unicode relocation path is preserved
under Windows legacy locales.
