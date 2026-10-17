# Windows compiler diagnostic: versioned ownership correction

Status: preregistered, not executed. The
[first compiler job](nemotron-windows-compiler-job-failure.md) records live
descendants in its Job after the first MSVC build worker exits. It preserves actual
cleanup success, but not the wrapped exit code or remaining process identities.
Its114-member failure receipt and every actual producer input remain unchanged.
This correction completes process ownership before repeating the original
[compiler comparison](nemotron-windows-compiler-diagnostic.md).

The ranked investigation remains: surviving tool background work, delayed process
teardown, unfinished child work, or a wrapper/accounting transition. The original
data cannot choose among them. Record observations rather than assuming a named
compiler/telemetry service or a successful build status.

The new controller is an exact five-site derivation of its frozen predecessor:
select the v2 supervisor, protocol and supervision label, then retain the helper's
observation on both returned results and raised errors. A guard rejects missing
or duplicated source sites and any extra controller changes. The SDK compiler
adapter, all inference sources, native probe compiler/source, portable ISA,
runtime identity, assets, streaming geometry, full-output1e-6 limits and fixed
MSVC/Clang/Clang/MSVC sequence are unchanged. The integer prediction rule still
requires both Clang observations at least10% faster than both MSVC controls and
absolute control drift at most10%; it never changes the original below-one gate.

Each command still uses a kill-on-close Windows Job and the original gated worker
payload. No native command can spawn before successful assignment. Preserve the
worker PID, assignment outcome, whether normal exit was observed, the exit code
observed before cleanup, and the code after reaping. A code observed only after
timeout is retained as such; it does not turn a timed-out command into success.

After normal worker exit allow0.25seconds for teardown, then take a bounded Job
PID snapshot. Include nested Job members through the documented
[process-list contract](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_process_id_list).
Grow an incomplete list, including a successful API response with fewer returned
than assigned PIDs, for at most five attempts and4096 entries. Reject duplicate,
invalid or inconsistent observations. Open only read-query handles to those
reported PIDs, and confirm
[membership in this exact Job](https://learn.microsoft.com/en-us/windows/win32/api/jobapi/nf-jobapi-isprocessinjob)
before querying an executable image. A raced or inaccessible process retains
its query/membership error; it does not establish an identity. Never inspect an
image after the handle no longer belongs to the Job. This is not a host-wide
process census or a proof of which tool caused the original failure.

Then terminate the owned Job, reap the worker, require zero active Job processes
and close the Job handle before returning the original observed command status.
Apply that cleanup to normal zero/nonzero exit, timeout, assignment failure and
other errors. Background descendants after normal exit are drained rather than
automatically replacing the wrapped status with the old live-descendant error.
No cleanup is performed by descendant PID or image name. An unassigned gated
worker is the only direct child killed separately; it received no payload and
has no command descendants. Preserve cleanup/snapshot errors and never publish
an unverified tree as a successful command. SDK installation, source/compiler
audit, state/version and full model gates must still reject incomplete outputs.

Windows preflight runs the original eight compiler controls and two new
controller/metadata controls. The versioned Job tests inherit the original five
conformance controls, plus PID-list layout/bounds, malformed PID rejection,
successful-partial-list growth and the reused-PID membership guard. The actual
normal-background regression creates a real descendant for both zero and nonzero
top-level exits; it requires that PID's confirmed Job membership/image, preserves
the exact wrapped code and verifies that its delayed write never occurs. The
inherited actual timeout and assignment-failure controls now exercise v2 as well.
These Windows API cases remain unexecuted until the workflow runs and gate every
SDK build. Local skips must not be reported as Windows passes.

After preflight, repeat the original fresh-build experiment on one Windows2022
host. Preserve every child command, its process observation, all compiler and
SDK provenance, every full probability array, all individual timings and any
failure. Retain both controls without averaging or cross-host timing promotion.
No SDK binary, model, geometry, thread budget or numerical threshold is changed
to make this experiment pass. No public/lifecycle, package, cleanup probe,
coexistence, quality, human annotation, physical desktop or supported-release
gate is closed by this process-ownership correction.
