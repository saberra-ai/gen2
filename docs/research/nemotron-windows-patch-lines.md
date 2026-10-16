# Windows NeMo patch materialization: newline compatibility fix

Read-only research. No source/configuration edits, native builds or inference were performed. The Windows CI failure occurs while materializing the pinned upstream patch series, before SDK compilation. The narrow remedy is a build-script newline fix with byte-equivalence guards; preserve the upstream native patch series and inference code.

## Observed failure and scope

CI run38862692131/job116848381672 retained the failing generated patch and pristine-source diagnostics. The archived `ggml/include/ggml.h` has zeroCRLF,116,144bytes,SHA256 `12ee71f99db7db9b353bc02b1fbb57c344ee17c01ac5fb7952b41a637a747ea9`. The failed generated patch has95,636bytes and2,113CRLF newlines. Replacing only CRLF with LF produces byte-for-byte the pinned first patch, `ggml-add-fused-attention-op-for-fastconformer-and-magpietts.patch`:93,523bytes,SHA256 `58a43285e49f3ab9569bfb98a459979c062084b87da8d54cdbdc377ee1724152`. The investigation already reproduced the five context failures with those CRLF bytes and successful `git apply --check` after normalization; this note did not repeat builds. Git's explicit LF settings were captured, so another checkout/autocrlf adjustment does not address the write operation creating the failing artifact. [Pristine-source diagnostic](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-patch-materialization/upstream-patch-diagnostic.json), [failed generated patch](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-patch-materialization/failed-upstream.patch.log), [Git configuration evidence](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-16-patch-materialization/git-line-ending-config.stdout.log).

Upstream pin is NeMo-Speech.cpp`083861c241061562b8ec29eb06d7f6cdc3af8775`, llama.cpp`bd4f514db14d87fded667787a7a963bfbaa98e89`. Its [cmake/llama_cpp.cmake](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/cmake/llama_cpp.cmake) normalizes the in-memory patch string, then line124 uses `file(WRITE "${normalized}" "${content}")` immediately before `git apply`. Original CMake file SHA256 is `22357d4a5d8a67716daa8b4e374abd63120432bd57540f97d572a525d5354c7b`. Preserve this hash as the compatibility transformation's input precondition.

## Why normalized strings became CRLF files

CMake's audited v3.31.6 `HandleWriteImpl` opens `cmsys::ofstream` with `std::ios::out`/`app`, without binary mode. Windows CRT text-mode output translates LF to CRLF. Normalizing a CMake string therefore does not guarantee LF bytes on disk. CMake's explicit-newline `file(CONFIGURE)` path sets binary output and emits the requested newline characters itself. This source behavior explains the retained artifact; the evidence bundle did not identify the exact installed CMake version, so record its version/executable hash in the next receipt instead of claiming v3.31.6 was necessarily the failed runner's binary. [CMake WRITE source](https://github.com/Kitware/CMake/blob/v3.31.6/Source/cmFileCommand.cxx#L123), [CONFIGURE binary/newline path](https://github.com/Kitware/CMake/blob/v3.31.6/Source/cmFileCommand.cxx#L3611), [generated-stream binary flag](https://github.com/Kitware/CMake/blob/v3.31.6/Source/cmGeneratedFileStream.cxx#L64), [Microsoft text/binary translation rules](https://learn.microsoft.com/en-us/cpp/c-runtime-library/file-translation-constants?view=msvc-170).

## Recommended minimal compatibility transformation

Replace only the patch-file materialization operation in a separately prepared build-source copy:

```cmake
# Preserve the preceding upstream READ and CRLF-to-LF string normalization.
string(SHA256 expected_patch_bytes "${content}")
file(CONFIGURE OUTPUT "${normalized}" CONTENT "${content}"
     @ONLY NEWLINE_STYLE UNIX)
file(SHA256 "${normalized}" written_patch_bytes)
if(NOT written_patch_bytes STREQUAL expected_patch_bytes)
    message(FATAL_ERROR "Patch materialization changed normalized source bytes")
endif()
```

`file(CONFIGURE)` is available since3.18, below upstream's3.26minimum. It writes during command execution, so the immediately following upstream `git apply` can consume it. `@ONLY` prevents substitution of literal `${...}` sequences in patch payloads. **It still permits `@VAR@` expansion and `#cmakedefine` processing**, so the raw-byte hash guard is essential; a newline flag alone is insufficient. Auditing all19pinned patches found zero matches for CMake's actual `@[A-Za-z_0-9/.+\-]+@` token pattern and zero`#cmakedefine`/`#cmakedefine01` directives; all are LF-only and have a final LF. The output hash check also catches unexpected substitutions, preprocessing, extra terminal newlines and future source drift. [Versioned CONFIGURE contract](https://cmake.org/cmake/help/v3.26/command/file.html#configure), [configure_file substitution rules](https://cmake.org/cmake/help/v3.26/command/configure_file.html), [actual CMake substitution patterns](https://github.com/Kitware/CMake/blob/v3.31.6/Source/cmMakefile.cxx#L118).

Optional preflight guards can reject those substitution/directive forms before writing, providing clearer error messages. Retain the post-write SHA256 comparison regardless. Comparing text returned by another `file(READ)` is weaker because Windows text-mode reads may normalize the very CRLF bytes under investigation; use `file(SHA256)` on the output.

Do not use `file(GENERATE)` here: its output is delayed until the generation phase, after the immediate patch application. `configure_file(COPYONLY ... NEWLINE_STYLE UNIX)` is also invalid: COPYONLY and NEWLINE_STYLE cannot be combined. `configure_file(... @ONLY NEWLINE_STYLE UNIX)` is a possible equivalent but still needs substitution/directive guards and has a larger change to this content-based call site. [Generation timing](https://cmake.org/cmake/help/v3.26/command/file.html#generate), [COPYONLY restriction](https://cmake.org/cmake/help/v3.26/command/configure_file.html).

## Provenance and option hygiene

Treat this as **pinned upstream plus an explicit build compatibility overlay**, not an unmodified upstream build:

1. Verify the clean parent revision, llama submodule pin, original CMake file hash, `patches/series` hash and all19patch-content hashes before transformation. Refuse a different original hash or anything other than one exact intended write-site match.
2. Apply a reviewed, versioned compatibility diff to an ephemeral source/build copy. Record its SHA256, the resulting CMake file hash, transformation tool hash, original revision and exact changed path. Keep the original-source hashes intact in the upstream manifest and add an explicit overlay section instead of relabeling the changed file as pristine upstream. No model, ABI, C++ inference/kernel files or pinned patch payloads should differ.
3. Before applying each upstream patch, verify the generated normalized bytes against the expected normalized content hash. Preserve the generated patch evidence if it fails. Preserve the normal upstream patch-order/stamp logic; materialize a clean build directory after changing the compatibility overlay.
4. Retain `NEMO_SPEECH_GGML_PATCHED=ON` for the WindowsCPU configuration being qualified. Turning it OFF merely skips the failing stage and changes the implementation being qualified. Keep CPU ISA, component, dependency and install flags identical to the failed candidate unless separately justified and recorded. Do not add `git apply --ignore-whitespace`, `--reject`, `--3way` or broad whitespace rewriting to make unrelated contexts fit.
5. A successful patch stage alone is not Windows qualification. Follow with the already-required build and same-host native/public-lifecycle checks in a new receipt, retaining the original failed receipt. This research does not claim those later gates have passed.

These recommendations preserve the source identity while describing the precise additional transformation. The upstream patch materialization already changes the prepared ggml tree by design; this overlay should not change the *intended* prepared-tree bytes.

## Alternative if modifying the CMake build-source copy is undesirable

The upstream `NEMO_SPEECH_LLAMA_CPP_SOURCE_DIR` override accepts an external tree as-is. A controller could make a pristine export at the locked llama commit, apply the exact19LF patch payloads outside CMake with a byte-preserving tool, hash the resulting tree, then pass that directory while keeping the patched-runtime compile option ON. This bypasses the faulty materializer but requires more staging/provenance logic. It is an alternative, not the minimal fix. [Source override resolution](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/cmake/llama_cpp.cmake), [patched runtime compile-definition gate](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/083861c241061562b8ec29eb06d7f6cdc3af8775/src/runtime/ggml/CMakeLists.txt).

Do not assume `CMAKE_PROJECT_TOP_LEVEL_INCLUDES` alone replaces the upstream helper: upstream later directly includes`cmake/llama_cpp.cmake`, defining its function again. A pre-include would have to prepare and set the **documented external-source override**, or supply an explicit audited source-copy edit. General command interception would hide a wider behavioral change. [CMake top-level-include scope](https://cmake.org/cmake/help/v3.26/variable/CMAKE_PROJECT_TOP_LEVEL_INCLUDES.html).
