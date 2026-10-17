# Windows CPU worker diagnostic: checkout correction

The first diagnostic run, [39024922313](https://github.com/saberra-ai/gen2/actions/runs/39024922313),
failed at native source checkout, before compilation, model acquisition or any
matrix case. Its [61-member collection](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-checkout-lfs-failure/collection-qualification.json)
binds the failed run/artifact to source `614aaa0dc5998a219806702e9a82128714b6fcd2`,
all48 tools and266 repository files, and all seven ZIP members. Local clone
succeeded; checkout returned128 because Git LFS tried to retrieve the unrelated
TTS asset `src/tts/tokenizer/mandarin_data/hmm_model.utf8`, whose LFS object was
unavailable. No performance or configured-worker observation was produced.

The driver omitted `GIT_LFS_SKIP_SMUDGE=1`, which the existing local source
preparer uses. The separately versioned v2 driver/source verifier/test/protocol
and workflow preserve every original input. The correction adds this variable
to the diagnostic child environment only. It changes no host Git configuration,
native inference source, CPU settings, model acquisition or required asset hash.
The new protocol requires all three derived source-hash maps and runtime tags to
equal the original preregistration exactly.

The regression uses the actual installed Git LFS filter and isolated temporary
repositories containing a pointer to an intentionally unavailable local object.
A fresh checkout without the policy must fail at smudge; a separate fresh clone
using the corrected mapping must check out the exact pointer bytes. The original
mapping must remain unchanged. The fixture has no network remote or host Git
configuration writes. Git-LFS-unavailable test environments explicitly skip this
one regression; the Windows workflow has Git LFS available.

All [original hypotheses, fixed4/2/1/4 order and numerical/timing rules](nemotron-windows-worker-diagnostic.md)
remain unchanged. The new diagnostic still requires three fresh SDKs, exact
derived source and LF-overlay admission, independent19-patch/compiler audits,
MSVC/private-DLL/state/version checks before assets, full6000x8 histories and
final-turn parity at1e-6, exact PCM/geometry and configured/attached-pool telemetry.
Release, package and development-native qualification remain false. A successful
matrix would supply native diagnostic evidence only.

Status: checkout regression passes locally; actual corrected Windows execution
is pending. The first failed run is preserved rather than retried or relabeled.
