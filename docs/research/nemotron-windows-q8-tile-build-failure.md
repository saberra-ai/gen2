# Windows Q8 tile control: standalone link failure

The first actual Windows attempt fails before any benchmark process runs. It
establishes neither kernel output parity nor a timing result. The standalone
harness omitted the original CPU half-conversion table required by the complete
SGEMM translation units. Main remains unqualified.

Producer `72bcb3d28487d803048122af03501895fe384871` ran in
[run 39043559355](https://github.com/saberra-ai/gen2/actions/runs/39043559355),
job `117427696242`. All 15 compiler controls, seven tile admission controls and
ten actual Windows Job controls pass. The build step fails; artifact preservation
succeeds. The [frozen experiment](nemotron-windows-q8-tile-probe.md) and its inputs
remain unchanged.

## Admitted facts

The [saved receipt](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-tile-build-failure/collection-qualification.json)
binds artifact `12104713750`, 25,328,962 bytes, SHA-256
`67a4fbb97a6320015a46703768f324652f2a93890167df73f35b04c7269d4ea6`.
Saved-data admission verifies all 3,519 members, 95 frozen tool inputs and 1,739
repository references at the producer commit. The complete canonical 3,425-file
tree, pristine archive and all nineteen patch reads match the preceding admitted
source audit. The candidate differs only by the declared Q8 dispatch change.

All 44 supervised commands retain their observed exit status, reaped process tree
and closed Job without cleanup errors. Before cleanup, 43 snapshots are empty;
configuration records one membership-confirmed `vctip.exe` descendant, which is
handled by owned-process cleanup. The first 43 commands return zero;
the build returns `4294967295`. Its linker log records the underlying link exit
one and `undefined symbol: ggml_table_f32_f16`, referenced by the original Q8
kernels and other retained instantiations. The configured host is Windows Server
2022 build 20348, reporting an AMD EPYC 7763, two cores/four logical processors,
AVX2 and FMA. No timing comparison with another host/run is made.

There are zero benchmark cases and zero SDK/model calls. Actual kernel objects,
final executable, support rejection controls and vector/timing acceptance never
reach completed admission. Successful Python admission tests do not establish
that the C++ negative controls ran.

## Required correction

A separately versioned harness must provide and initialize the original
half-conversion table without changing the canonical kernel arithmetic. The
compiler also warns that all three throwing C-linkage support guards have a
non-throwing contract under the actual `/EHsc` settings. Replace that mechanism
with a valid fail-closed control and demonstrate the negative cases on Windows.
The failed link command includes `libomp.lib`; explicitly disable OpenMP in the
standalone configuration and retain the resulting dependency evidence.

Keep the original twelve-process order, guarded shapes, every raw vector,
worker counts and integer timing predicates. A successful kernel experiment
would still require observed model execution paths and a separate full SDK/model
comparison. Original quality, native/public performance, lifecycle, packaging,
offline, desktop and signing gates remain open, as does the user-deferred human
word-reference review.

## Collection provenance

The first two saved-data collectors hit historical shared entry limits: the first
before extraction and the second during its post-extraction inventory. Their
helpers, logs and failed state remain preserved. The separately versioned third
receiver uses explicit bounded inventories suitable for the retained source tree;
both review axes cleared it before invocation. It hashes every member before
extraction and checks the exact inventory afterward. Collection does not repeat
compilation, benchmark execution or inference.

The receipt retains 93 artifact payload copies, the complete member map, producer
metadata, frozen tool sources and executed receiver provenance. The verified ZIP,
pristine archive and complete canonical tree stay in the local admitted delivery.
The local Python suite passed 430 tests with seven Windows-only skips before the
producer push. No production or release gate is promoted by this failed build.
