"""Own a gated command and every descendant in a Windows kill-on-close Job."""
import ctypes
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

DWORD, HANDLE, SIZE = ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t
MAX_PAYLOAD = 4096


class BasicLimits(ctypes.Structure):
    _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64), ('flags', DWORD),
        ('minimum_working_set', SIZE), ('maximum_working_set', SIZE), ('active_limit', DWORD),
        ('affinity', SIZE), ('priority', DWORD), ('scheduling', DWORD)]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic', BasicLimits), ('io', ctypes.c_uint64 * 6),
        ('process_memory', SIZE), ('job_memory', SIZE), ('peak_process_memory', SIZE), ('peak_job_memory', SIZE)]


class Accounting(ctypes.Structure):
    _fields_ = [('times', ctypes.c_int64 * 4), ('faults', DWORD), ('total', DWORD),
        ('active', DWORD), ('terminated', DWORD)]


class Job:
    def __init__(self):
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([ctypes.c_void_p, ctypes.c_wchar_p], HANDLE),
            'SetInformationJobObject': ([HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD], ctypes.c_int),
            'AssignProcessToJobObject': ([HANDLE, HANDLE], ctypes.c_int),
            'QueryInformationJobObject': ([HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD, ctypes.c_void_p], ctypes.c_int),
            'TerminateJobObject': ([HANDLE, DWORD], ctypes.c_int),
            'OpenProcess': ([DWORD, ctypes.c_int, DWORD], HANDLE),
            'CloseHandle': ([HANDLE], ctypes.c_int)}
        for name, (arguments, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = arguments, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway permission.
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        handle = self.api.OpenProcess(0x0101, False, process.pid)  # SET_QUOTA | TERMINATE
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.api.AssignProcessToJobObject(self.handle, handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            if not self.api.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())

    def empty(self, timeout=5):
        deadline = time.monotonic() + timeout
        while True:
            value = Accounting()
            if not self.api.QueryInformationJobObject(self.handle, 1, ctypes.byref(value), ctypes.sizeof(value), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if value.active == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(.01)

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, None
            if not self.api.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())


def _child():
    # The parent sends nothing until assignment succeeds. No command can spawn
    # before this gate, including when OpenProcess/AssignProcessToJobObject fails.
    payload = sys.stdin.buffer.readline(MAX_PAYLOAD + 1)
    if not payload.endswith(b'\n') or len(payload) > MAX_PAYLOAD:
        raise ValueError('bounded complete owned-command payload required')
    value = json.loads(payload)
    if (not isinstance(value, dict) or set(value) != {'command', 'cwd'}
            or not isinstance(value['command'], list) or not value['command']
            or any(not isinstance(item, str) or not item or '\0' in item for item in value['command'])
            or not isinstance(value['cwd'], str) or not value['cwd'] or '\0' in value['cwd']):
        raise ValueError('explicit owned command and directory required')
    return subprocess.run(value['command'], cwd=value['cwd'], stdin=subprocess.DEVNULL).returncode


def run_in_job(command, *, cwd, env, stdout, stderr, timeout):
    if platform.system() != 'Windows' or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise ValueError('actual Windows x64 owned-process execution required')
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('positive owned-command timeout required')
    payload = (json.dumps({'command': command, 'cwd': str(cwd)}, ensure_ascii=True) + '\n').encode('ascii')
    if len(payload) > MAX_PAYLOAD:
        raise ValueError('owned command exceeds bounded control-pipe capacity')
    job = Job()
    worker = failure = None
    try:
        worker = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--job-child'],
            cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, bufsize=0)
        job.assign(worker)
        try:
            worker.communicate(input=payload, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise subprocess.TimeoutExpired(command, timeout) from error
        if not job.empty():
            raise RuntimeError('owned command left live descendant processes')
    except BaseException as error:
        failure = error
    finally:
        try:
            if worker is not None:
                try:
                    job.terminate()
                finally:
                    # On assignment failure this gated worker is outside the
                    # job, but it has received no payload and has no children.
                    if worker.poll() is None:
                        worker.kill()
                    worker.wait(timeout=5)
                    if worker.stdin is not None and not worker.stdin.closed:
                        worker.stdin.close()
                if not job.empty():
                    raise RuntimeError('owned process-tree termination not verified')
        finally:
            job.close()
    if failure is not None:
        failure.process_tree_reaped = True
        raise failure
    result = subprocess.CompletedProcess(command, worker.returncode)
    result.process_tree_reaped = True
    return result


if __name__ == '__main__':
    if sys.argv[1:] != ['--job-child'] or platform.system() != 'Windows':
        raise ValueError('internal gated Windows Job worker only')
    status = _child()
    # Preserve all32 bits of an actual Windows exit status, including crashes.
    exit_process = ctypes.WinDLL('kernel32').ExitProcess
    exit_process.argtypes, exit_process.restype = [DWORD], None
    exit_process(DWORD(status))
