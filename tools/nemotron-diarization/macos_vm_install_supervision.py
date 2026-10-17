"""Cancel an installer owner before bounded kill/reap; never infer VM teardown."""
import os
import signal
import subprocess
import time


def execute_install_owner(argv, *, cwd, environment, record, persist, guard,
                          timeout, cleanup_timeout, stdout, stderr):
    process = None
    try:
        process = subprocess.Popen(argv, cwd=cwd, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        record['pid'] = process.pid
        persist()
        deadline = time.monotonic() + timeout
        last_persist = time.monotonic()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('installer owner operation deadline exceeded')
            try:
                record['returncode'] = process.wait(timeout=min(1, remaining))
                return
            except subprocess.TimeoutExpired:
                if guard.check() < guard.running_minimum:
                    raise OSError('installer running storage floor exceeded')
                if time.monotonic() - last_persist >= 5:
                    persist()
                    last_persist = time.monotonic()
    except BaseException as error:
        record['abort_error'] = repr(error)
        # Free diagnostics first. Failure must not skip cancellation/reaping.
        try:
            guard.release()
        except BaseException as release_error:
            record['reserve_release_error'] = repr(release_error)
        if process is None:
            record['startup_error'] = repr(error)
        else:
            try:
                if process.poll() is None:
                    record['cancellation_signal'] = 'SIGTERM'
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    # Persistence is diagnostic; cannot veto the cleanup wait.
                    try:
                        persist()
                    except BaseException as diagnostic_error:
                        record['cleanup_persist_error'] = repr(diagnostic_error)
                    process.wait(timeout=cleanup_timeout)
            except BaseException as cleanup_error:
                record['owner_cleanup_wait_error'] = repr(cleanup_error)
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                        record['forced_process_group_termination'] = True
                    except ProcessLookupError:
                        pass
                record['returncode'] = process.wait()
                record['owner_exit_does_not_prove_vm_teardown'] = True
        raise
