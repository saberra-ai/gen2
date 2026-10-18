# Windows feature-check checkout regression

Feature-check run [39088182262](https://github.com/saberra-ai/gen2/actions/runs/39088182262)
at source `ba48a1d0101aa454e74f6eacd9838e9ae724d5a4` fails all four Windows
matrix entries during repository checkout. No Windows compilation or test runs
in those jobs. The eight Linux/macOS feature jobs and the evidence-tools job
pass.

All four failed logs report `Filename too long` for the two nested retained
thread-CPU review sources. Their absolute paths under `D:\a\gen2\gen2` are
264 and 275 characters. The fetch succeeds, and a scan of all 28,179 tracked
paths at that source finds no invalid Windows filename components. This
distinguishes the observed failure from a model/runtime failure, network fetch
failure or invalid filename. The subsequent bounded-corpus receipt raises the
current number of paths at least 260 characters long to 22 under that root.

[Git for Windows documents](https://gitforwindows.org/git-cannot-create-a-file-or-directory-with-a-long-path.html)
the `core.longpaths` option for its built-in commands.
[Git documents](https://git-scm.com/docs/git-config/2.50.0#Documentation/git-config.txt-GITCONFIGCOUNT)
the `GIT_CONFIG_COUNT`/`GIT_CONFIG_KEY_n`/`GIT_CONFIG_VALUE_n` runtime configuration
interface. The feature matrix and normal CI now supply `core.longpaths=true`
through that process environment before checkout. No registry, host Git config,
credential, model, runtime, scorer or frozen receipt changes are needed.

The existing four Windows checkout failures are the regression at the actual
seam. A Windows-only step additionally verifies the effective Git setting and
the existence of every tracked path whose absolute length is at least 260
characters before toolchain setup. The repair must pass the complete existing
feature matrix; a local POSIX checkout cannot establish the Windows fix.

For a manual full repository clone on Windows, the same process option can be
supplied as `git -c core.longpaths=true clone https://github.com/saberra-ai/gen2.git`.
This describes checkout requirements, not a supported Windows runtime release.
Historical diagnostic workflows retain their preregistered source and settings;
their checkout environment must be included when a justified new diagnostic is
preregistered. No unchanged model experiment is dispatched for this repair.

At this implementation checkpoint the repaired Windows jobs have not run.
Original AMI accuracy, Windows real-time performance, human review, physical app,
fresh-machine, other-platform and public release gates remain open. Main remains
unpromoted.
