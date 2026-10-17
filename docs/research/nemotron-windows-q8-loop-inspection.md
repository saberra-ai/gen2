# Q8 generated-loop inspection after the failed 2×4 prediction

Saved assembly confirms explicit vector stack accesses in both existing Q8
helpers. The 2×4 helper has fewer such accesses than 4×2, while its
[actual speedup prediction still fails](nemotron-windows-q8-mx4-results.md).
Static access counts alone therefore do not justify choosing another tile or
claiming a measured spill bottleneck. No compiler, native code or model runs
during this inspection.

The [inspection record](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-loop-inspection/inspection.json)
binds the two original disassembly files from producer
`934bb2734ffd1aadfe3596d3f00a6b47d7948f3f` to their existing collection manifest.
The control log is 1,922,731 bytes, SHA-256
`b5f0feeee40fcf0bb74c016673f961bc8c7ab6f4d675af520934db28611bf3c8`;
the candidate log is 1,924,752 bytes, SHA-256
`e1ee1704c722fece87bc20884746b521066315fed236cc048b445c61f339083e`.

## Method and observations

Select the unique `gemm4xN<2>` and `gemmMx4<2>` function headers for
`tinyBLAS_Q0_AVX<block_q8_0,block_q8_0,float>` in each object disassembly.
Within each function, select the unique backwards `jne` and its target to
identify the block-reduction loop. Count only explicit `xmmword`/`ymmword`
memory operands relative to `rsp` within those inclusive bounds, using operand
widths 16/32 bytes. The record retains every counted instruction and offset.
These are static instruction operand widths, not measured cache or DRAM traffic.
Function prologues, epilogues, tile setup and final horizontal reduction lie
outside this count.

| Existing helper | Loop target → backedge | Instructions in loop | FMA instructions | Vector stack loads | Vector stack stores | Sum of operand widths |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4×2 (`gemm4xN<2>`) | `0x3c0` → `0x5b1` | 94 | 8 | five 32-byte + one 16-byte | five 32-byte + one 16-byte | 352 bytes |
| 2×4 (`gemmMx4<2>`) | `0x3d0` → `0x5a7` | 95 | 8 | three 32-byte | three 32-byte | 192 bytes |

For each helper, the loop's instruction text and offsets match between control
and candidate objects. This does not assert identical complete object bytes or
resolved constant addresses. Full-function comparison reveals differing
`.rdata` relocation offsets in the pre-loop `vpbroadcastw`; no constant-value
equivalence was independently established here. The previously admitted exact
source, vector controls and complete half tables remain the numerical evidence.

## Next timing diagnostic

The actual first four-worker candidate process is substantially slower on many
large shapes than the second candidate process; every recorded observation
stays included in the failed result. The one/two-worker observations are mixed.
Neither this timing pattern nor these static loop counts identifies its cause.

Before another production candidate, use a separately frozen diagnostic with
both existing kernels interleaved in the same process and persistent worker
pool, complete per-worker kernel and whole-dispatch wall observations, and
repeated surrounding controls. Preserve all 22 shapes and one/two/four-worker
cases, arithmetic, full outputs, support/half-table guards, input immutability
and every original release gate. Declare the process/worker-scheduling and
kernel-work hypotheses and decision rules before execution; new observations
must not replace, filter or retroactively qualify the failed experiment.

That new diagnostic is not implemented or executed by this inspection. No
production optimization, SDK/model RTF or release qualification is established.
