# Windows Q8 2×4 result: exact outputs, failed speedup prediction

The captured-geometry standalone experiment completes construction and every
numerical/support control. All 264 output vectors match byte for byte across
the twelve fresh control/candidate and worker processes. None of the twelve
primary shapes passes the original speedup predicate, and five additionally
fail the control-drift bound. The candidate is not qualified for production.

Producer `934bb2734ffd1aadfe3596d3f00a6b47d7948f3f` completed
[run 39054094395](https://github.com/saberra-ai/gen2/actions/runs/39054094395),
job `117458403727`. The [frozen experiment](nemotron-windows-q8-mx4.md) preserves
the candidate, geometry selection, process order and original integer timing
criteria. Its pending status describes the pre-execution checkpoint; this
document records the subsequent actual result.

## Admitted construction and controls

The [saved receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-mx4/collection-qualification.json)
binds artifact `12109212382`, 208,542,016 bytes, SHA-256
`857291688ebc5ec14fbda54dfadb06810e98909521b6e34761b096110e1aa767`.
Independent saved-data replay verifies all 3,904 delivered members, 112 frozen
tools, 2,807 producer repository references, the complete 3,425-file canonical
tree, pristine archive, nineteen patch reads and exact candidate transform.
The candidate compiler command contains the required Windows AVX2/F16C dispatch
guard definition. Original compiler/linker fingerprints, contraction-off and
global/Release options, source inputs, objects, disassemblies and executable
dependency evidence are admitted. No compilation or native execution is
repeated during collection.

All 95 observed commands retain expected exit codes and admitted owned cleanup:
59 zero exits, and twelve each of 73, 74 and 75 for the separately supervised
support rejections. Every one of the 36 rejections matches its exact diagnostic.
All twelve full half tables match and independently validate all 63,488 finite
values plus signed zero, infinity signs and NaN classes. All 264 complete,
finite/nonzero output vectors match per shape across variants and one/two/four
workers. The bound harness checks immutable inputs, padded strides, output
guards and repeat equality after every call.

Actual Windows prerequisites pass 15 compiler, seven original tile, seven
corrected support, six new geometry/candidate and ten owned-Job tests. The local
producer suite passed 479 tests with seven Windows-only skips. Geometry replay
independently admits the prior bounded model receipt, all eight enabled worker
observations, six converted Q8 geometries, root counts and full column sets.
The chosen shapes use those dimensions with synthetic deterministic values;
they are not captured model tensor payloads.

## Timing result

Totals below include all eight timed calls in milliseconds, on four workers,
in the fixed control/candidate/candidate/control order. Every individual integer
nanosecond observation, all 22 shapes and all worker counts remain retained.

| Primary `(m,n,k)` | Control 1 | Candidate 1 | Candidate 2 | Control 2 | Control drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(512,13,2048)` | 2.1288 | 3.3767 | 2.0879 | 2.1326 | 0.1785% |
| `(512,292,2048)` | 40.0854 | 74.5325 | 42.3553 | 53.0025 | 32.2240% |
| `(512,535,2048)` | 71.0574 | 122.5473 | 69.5815 | 82.0859 | 15.5206% |
| `(2048,13,512)` | 2.3682 | 3.6966 | 2.2262 | 2.3139 | 2.2929% |
| `(2048,292,512)` | 46.2572 | 74.6310 | 46.4195 | 45.9290 | 0.7095% |
| `(2048,535,512)` | 94.1013 | 139.4176 | 84.0728 | 84.0169 | 10.7165% |
| `(1536,13,512)` | 1.8245 | 2.6442 | 1.7300 | 1.8049 | 1.0743% |
| `(1536,292,512)` | 32.5554 | 59.0451 | 38.1229 | 35.0843 | 7.7680% |
| `(1536,535,512)` | 61.6917 | 86.6718 | 63.5314 | 62.9026 | 1.9628% |
| `(512,13,512)` | 1.2881 | 0.8918 | 0.8077 | 0.8369 | 35.0283% |
| `(512,292,512)` | 13.0183 | 18.4202 | 12.8509 | 13.8248 | 6.1951% |
| `(512,535,512)` | 22.3865 | 22.5729 | 22.0232 | 24.9120 | 11.2814% |

Every primary fails `20*max(candidates) <= 19*min(controls)`. Five also fail
`10*abs(last-first) <= first`: indices 5, 6, 9, 13 and 15. The overall kernel
speedup prediction is false. Many first candidate totals are substantially
slower; the second candidates are mixed. Dropping a candidate, averaging across
shapes or relaxing drift would change the frozen experiment and is not done.
Timing includes harness dispatch/wake and scheduling effects; these observations
do not identify a register-spill, reuse or frequency cause.

The host reports Windows Server 2022 build 20348, Intel Xeon Platinum 8370C at
2.80 GHz, two cores/four logical processors, AVX2 and FMA. It differs from the
EPYC 7763 bounded model-profile host. Comparisons across those jobs cannot
establish a kernel-to-model or cross-host speedup.

Retain the existing production kernel. Before selecting another implementation,
inspect the actual generated kernel loops and distinguish arithmetic work from
worker scheduling using a separately frozen diagnostic. No SDK/model is built
or loaded here. Full-model/native/public output, original RTF, lifecycle,
package/offline/coexistence, desktop, quality and signing gates remain open.
Human word-time annotation and independent review stay documented and deferred
at the user's request. Main remains unchanged and unqualified for promotion.

## Retained evidence

The 607-file receipt includes 606 manifested payloads: 475 artifact copies,
frozen tool sources, original metadata/logs and executed receiver/publisher
provenance. The complete member map binds canonical source, pristine archive,
native objects and executable retained in the admitted local cache. All
receiver stages completed successfully on their first actual invocation.
