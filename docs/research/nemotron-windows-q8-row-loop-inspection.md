# One-row generated loop: no explicit vector stack operands, failed speed gate

Saved assembly from the completed [one-row experiment](nemotron-windows-q8-row-results.md)
has no explicit RSP-relative vector memory operands in its reduction loop.
Normalized to eight outputs, its loop has90 instructions versus95 for production
Mx4. The original0/48 speed result and failed observer gate remain unchanged.
Removing these static accesses and reducing the instruction count did not
establish the required dispatch improvement in that actual experiment.

The [17-member inspection receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-row-loop-inspection/collection.json)
binds original producer2dbf3a008bde6ad405dccd33bb7f015da1901b46,
run39070438530, the admitted original result manifest/replay/member map and both
actual kernel objects/raw disassemblies. Parsed metadata and admitted inputs
retain their first byte identities through copying and final rechecks.
The original producer's thirteen successful feature-CI jobs are retained too;
no new compiler, kernel, SDK or model invocation occurs during this inspection.

## Identification and counts

The existing parser is changed only to accept a fixed expected FMA count,
eight for `gemmMx4<2>` and four for `gemmMx4<1>`. Select the unique Q8/Q8/F32
helper signature and its unique innermost backward conditional loop with that
count. Four wrong-signature/count controls reject. Count explicit XMM/YMM
operands relative to RSP using16/32byte widths. Three pure publication-seam
controls also pass. Prologues, epilogues, tile setup and final reduction are
outside these loop counts.

| Helper | Loop bounds | Outputs per block iteration | Instructions | FMAs | Explicit vector stack operand widths |
| --- | --- | ---: | ---: | ---: | ---: |
| Production2×4 | `0x3d0`–`0x5a7` | 8 | 95 | 8 | 192bytes |
| One-row1×4 | `0x290`–`0x372` | 4 | 45 | 4 | 0bytes |
| One-row, normalized to8outputs | Two such iterations | 8 | 90 | 8 | 0bytes |

The zero applies to explicit vector stack operands in the selected loop. It
does not assert a stack-free function, no scalar stack operands, no generated
spills elsewhere or lower measured traffic. Normalized static counts do not
include changed tile setup, scheduling, caches or instruction costs.

## Consequence for the next hypothesis

A stack-access reduction alone is insufficient evidence to select a production
optimization. The one-row source reloads B across A rows; production reuses each
B vector across two rows. These are source work counts, not measured traffic
or an established cause of the failed timings.

A distinct packed-B-scale2×2 clone can test fewer live vectors while preserving
that two-row B reuse. It will repeat A work across adjacent column tiles and
double tile count/alter worker ownership. The older generic scalar-scale2×2
experiment already [failed](nemotron-windows-q8-tile-v2-results.md); its failure
cannot be removed or relabeled. Any new packed helper needs its own frozen
source, complete numerical/cleanup controls and unchanged timing/observer gates.
No new candidate or production result follows from this saved inspection.
