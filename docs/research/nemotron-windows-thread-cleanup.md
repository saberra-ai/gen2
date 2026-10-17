# Windows own-process failed-load cleanup control

Status on 2026-10-17: implemented and model-free tested locally; no actual Windows
observer or failed-load helper execution yet. The bounded SDK's performance gate
is still failed, so the strict receiver cannot admit that SDK for this experiment.

The [observer](../../tools/nemotron-diarization/windows_own_threads.h) uses
`CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)`, then retains only entries whose
owner PID matches `GetCurrentProcessId`. The API snapshots all threads for this
flag; its PID argument does not select just the caller. The implementation
publishes own counts, not other-process records, and requires successful snapshot
close before admitting a count.
See Microsoft's [snapshot contract](https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/nf-tlhelp32-createtoolhelp32snapshot).

It initializes/resets `THREADENTRY32.dwSize`, rejects returned metadata that cannot
cover the owner field or exceeds the buffer, rejects empty/duplicate/zero own IDs
and bounds enumeration. Failure of the first call is not an empty count; only
`ERROR_NO_MORE_FILES` ends subsequent enumeration. Other query or close errors
fail admission. These choices follow the [first-entry contract](https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/nf-tlhelp32-thread32first),
[structure definition](https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/ns-tlhelp32-threadentry32)
and [next-entry contract](https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/nf-tlhelp32-thread32next).

The separate [known-worker control](../../tools/nemotron-diarization/windows_thread_observer_control.cpp)
requires observing both the main and a deliberately live worker, then observing
the worker's retirement after joining, with at most100ms requested settling.
It imports no NeMo library and calls no model/frontend API. The
[builder](../../tools/nemotron-diarization/build_windows_failed_load_probe.py)
must compile/run that control successfully before compiling the C ABI failure
probe. Source hashes, actual compiler command/status, binary/PE identities and
observations are retained. Compilation does not execute the failure probe.

The [failure probe](../../tools/nemotron-diarization/native_failed_load_probe_windows.cpp)
requires the exact bounded tag and a missing model path. All101 actual failed
constructions must clear the model handle and return the exact file error.
Immediate counts, settled counts and requested10–100ms waits are retained for
every call. After the first warm-up failure, persistent growth above two
incidental workers fails. This observes thread cleanup only; it cannot prove
absence of all native memory leaks or qualify successful inference.

The [versioned offline receiver](../../tools/nemotron-diarization/qualify_bounded_windows_package.py)
places that probe first in a fresh offline container, removes model/reference
directories from the input copy, permits no extra asset mount, and verifies the
input inventory. Actual runtime module and System32 CRT observations still apply.
It then runs unchanged full-row parity, lifecycle and missing-dependency controls;
native and public observed RTF must both remain below one. The
[chat receiver](../../tools/nemotron-diarization/qualify_bounded_windows_chat.py)
inherits these gates and preserves isolated chat plus both load orders and
concurrent execution checks. Historical receivers stay unchanged.

All350 model-free tools tests pass locally. Nine compiled fake-API cases exercise
the observer's filtering, errors and handle cleanup; six synthetic receiver tests
cover failed-load histories, known-worker observations, model-free input copies,
execution order and late RTF failure. They make no Windows API, Docker or model
call. Actual Windows compilation/observation, helper cleanup, bounded package,
receiver orchestration and offline/chat qualification remain open. A new frozen
receiver protocol and qualified producer are required before execution.
