# Host runtime control before any guest search-bit experiment

Preregistered at `d5c4f4e5983404d032c30b8c97a872d77d61e708`, then executed
successfully as recorded below. This control addresses the runtime prerequisite identified by the
[stopped-guest inspection](nemotron-macos-vm-permissions-inspect.md). It does
not mount, clone or modify a guest, resolve guest ACL policy or authorize a
guest permission change.

The selected `/usr/bin/python3 -B` resolves to Xcode's Python3.9.6 on the
Apple M4 Pro, macOS26.3 arm64, effective UID502. The frozen
`macos-search-bit-control.protocol.json` binds both executed local modules,
both interpreter entry/resolved binaries and sandbox/host-query binaries,
expected host, exact fresh fixture
path, command and60-second deadline. A bounded offline child must return zero
and be reaped. No VM storage minimum is weakened: the earlier preflight observed
about9.5GiB available, below the separate120GiB initial/80GiB running VM requirement.
No VM operation is admitted by this host-only test.

The control exclusively creates a fresh private host fixture and two empty
directories named `users` and `groups`. It sets its new `Default` directory to
0600, then opens it with ordinary `O_RDONLY|O_DIRECTORY|O_NOFOLLOW`. Both
parent-relative child lookups must fail EACCES before the operation. Through
that exact inspected descriptor, `fchmod` may add only owner search (0700).
Both child metadata lookups must then succeed, retain mode0600 and stay on the
same device. No directory enumeration, file contents or account records are read.

The same descriptor must restore0600 on success, failure, interruption and a
mutator that raises after changing the mode. Ordinary SIGINT/SIGTERM delivery
must be deferred before mutation through restoration and descriptor closure;
uncatchable termination remains outside this
guarantee. Restore must retain exact inode/device/type, original permissions,
links, flags, size, mtime and apparent ownership, and the original anchored path.
Raw security responses before/during/after must agree. A metadata or path
inconsistency remains failed even if the descriptor's mode was restored; the
replacement path is never changed. Ctime is recorded separately because mode
restoration does not restore ctime or prove a pristine fixture.

Eleven real-filesystem tests cover successful access/restoration, existing-path
and symlink rejection, partial mutation, BaseException/descriptor closure,
changed security metadata, changed mtime, leaf/root replacement, failed
restoration and termination at cleanup entry/during restoration. Their
injected security responses do not qualify
the actual Darwin API; separate execution must retain its actual raw response.

Even a passing actual result proves only this runtime operation on a newly
created host directory. Guest ACL absence/authorization, DataRW operation,
guest UID/GID, provisioning, boot readiness and fresh-machine app execution
remain open. Any guest experiment needs separately reviewed admission, fresh
paired clone, exact mount/identity controls and sufficient storage. The prior
failed guest clones remain quarantined. Human word annotation/review stays
documented and deferred; original accuracy, Windows RTF, desktop and release
gates remain unchanged.

## Actual host result

The [saved receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-search-bit-host-control/collection-qualification.json)
records two zero-exit, reaped sandbox children: negative controls for model
reads, writes outside the fixture and network connection, followed by the
actual descriptor operation. The control uses the pinned Xcode Python3.9.6
runtime at effective UID502 on the Apple M4 Pro, macOS26.3 arm64.

Both `users`/`groups` stat calls first fail EACCES13. Adding owner search on
the ordinary open `Default` descriptor changes0600 to0700 and reaches both
same-device mode0600 child directories. Restoration returns that same inode
to0600 with the original root/path identity, flags, mtime, size, links and
apparent ownership. Ctime changes and is retained. All three actual raw
security responses equal the canonical12-byte12/8/0 response; this does not
prove ACL absence or guest authorization.

The model-free tool suite passes556 tests with seven Windows-only skips.
Eleven focused filesystem tests cover the operation and failure cleanup.
Two additional real-child controls inject SIGINT/SIGTERM after spawn but
before `Popen` returns; both child groups are killed and reaped. The saved
collector replays input/source identities, raw security/state transitions and
all command records without another permission or native operation.

At execution admission,17,373,478,912bytes were available, still below the
separate VM120GiB initial/80GiB running reserve. No cache or other project
files were deleted in this investigation. No guest clone, mount, boot or
permission change occurred. Actual guest ACL/access, stored ownership,
provisioning and fresh-machine complete-app operation remain unqualified.

## CI signal failure and V2 repair

Feature CI39069334234 at `b91ea5d` fails two signal cleanup tests on Linux
Python3.12. A new deterministic control also reproduces the defect on the
selected Mac Python3.9: delivering SIGTERM directly to an unblocked worker
thread while restoration starts dispatches the Python handler on the main
thread before mode restoration. Masking only the main thread does not cover
that case. The original host execution above remains retained, but supplies
no general multithreaded termination guarantee.

V2 installs nonraising Python handlers for SIGINT/SIGTERM before mutation,
retains them through restoration and descriptor closure, restores both prior
handlers, then redelivers queued signals. It does not change other threads'
OS signal masks or widen the filesystem operation. The deterministic regression
must retain `restore entry`, `mode restored`, `signal` in that order; both
prior handlers and original mode must be restored. Existing failure controls
remain required. The separate `macos-search-bit-control-v2.protocol.json`
freezes a fresh host fixture and changed source; the original protocol and
21-member receipt remain unchanged. This is no guest admission or
permission-policy change.

Actual V2 execution at `ceaf7590b5766387a0415d10f2d69bfb49133f36` passes
both zero-exit, reaped children in a separate fresh host fixture. The
[V2 receipt](../../benches/results/nemotron-diarization/apple-m4-pro/2026-10-17-search-bit-host-control-v2/collection-qualification.json)
retains exact0600/0700/0600 restoration, unchanged root/path identity and
identical12/8/0 security responses, with changed ctime. All557 tool tests
pass with seven Windows-only skips, including twelve focused filesystem
controls; both spawn-window termination controls also pass. The original CI
failure and deterministic worker-delivery red/green logs are preserved.
Actual admission has11,122,368,512bytes available, below the unchanged VM
reserve. Guest access, ACL policy and every original release gate remain open.

The revised producer's [feature CI](https://github.com/saberra-ai/gen2/actions/runs/39069873270)
now completes all thirteen jobs successfully. Linux evidence checks pass557
tests with seventeen platform skips, three Pnola integration tests and five
frontend asset tests. This closes the earlier CI cleanup regression; feature
compatibility does not establish guest operation or platform model qualification.
