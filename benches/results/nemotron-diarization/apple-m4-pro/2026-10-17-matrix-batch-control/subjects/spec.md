# Batched matrix observer emission control

Status: implemented; native graph execution pending. This is a separate candidate
after the [Windows matrix profile](nemotron-windows-matrix-results.md) preserved
four exact full-model histories but failed both original observer-overhead
bounds. Earlier observer sources, protocols and receipts remain unchanged.

Ranked hypotheses declared before execution: per-row stderr writes cause much
of the observer overhead; row formatting dominates; or timers and contention
within matrix operations dominate. Batching writes predicts that a separately
qualified full-model observer will satisfy the original overhead bound, if the
first hypothesis accounts for enough of the cost. Local graph controls alone
cannot establish that prediction or a Windows/model speedup.

## Candidate and failure contract

The new transform first requires the original canonical 132,918-byte CPU file
and applies the unchanged v1 matrix observer. Replace exactly its final emission
block, after the existing final barrier. Keep every capture, tensor lifetime,
timer, arithmetic operation, dispatch decision, chunk assignment, fusion and
barrier unchanged. Keep the v1 wire schema and decoder; this is a different
source identity, not a reinterpretation of the original diagnostic.
The resulting CPU file is 143,331 bytes, SHA-256
`2d0ea19dcc51f53ffc376f51ebd53aa5c575adeead0b29cf4bf1993a9ac58000`.

Format captured rows into a temporary heap buffer with a checked 2,048-byte
budget per captured root, at most 256 roots / 512 KiB per worker. Retain the
existing small shape/stride arrays; add no large stack buffer. Submit the entire
successful worker/graph buffer with one `fwrite`, then free it. Disabled and
empty cases allocate and write nothing. Reject allocation failure, negative or
truncated row formatting, and short writes with an `emission-error` marker that
the unchanged decoder rejects. Incomplete evidence cannot qualify a run.

Executable C seam tests run both actual emission blocks on 256 records with
extreme 64-bit dimensions/strides and require identical stderr bytes and one
batch write. Inject allocation, formatting, truncation and write failures; check
cleanup and rejection. These prove the emission contract, not model timing.

## Frozen local native control

Use a new exclusive build/output root. Freeze the controller, this specification,
complete local import/test closure, canonical fixture, unchanged graph harness
and explicit-LLAMAFILE CMake project before compilation. Admit all 3,425 canonical
source descriptors and require the CPU C file to be the only changed path.
Retain actual cache, compile commands, source/object/compiler identities, linked
libraries, dependency reports and build logs. Require LLAMAFILE on, portable CPU
configuration and the same disabled accelerator/OpenMP/backend-loading options
as the earlier Mac control. Recheck frozen source/build subjects, binary/library
identities and previous outputs between processes and afterward.

Run twelve fresh processes: workers 1/2/4, each off/on/on/off. Keep the unchanged
four F32/Q8 by 32/1-column graphs, three evaluations each, with one persistent
pool and the original RMS/multiply fusion. All 144 finite, nonzero outputs must
match byte-for-byte per shape across modes/repeats/workers. Decode all enabled
traces: twelve graphs per process, every worker, one matrix root, expected
routes [2,4,5,5], exact type/shape, stage attempts/rejections, input conversion
and fusion. Retain worker walls separately and preserve failed attempts.

Only a passing local control permits a new Windows SDK/model protocol. That
protocol must keep exact compiler/source/ABI/state checks, actual Windows graph
controls before model acquisition, four fresh full-model histories and complete
84-graph/four-worker/129-root coverage. Retain the original 10% drift and 10%
observer-overhead bounds, with every observation preserved. No production,
public API, package, quality, desktop or release gate is closed by this local
experiment. Human annotation/review remains documented and deferred; main
remains unqualified.
