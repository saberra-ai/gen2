"""Versioned owned-job cleanup after normal exit, timeout and command error."""
import ctypes
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

from windows_job import DWORD, HANDLE, MAX_PAYLOAD, SIZE, Job as BaseJob, _child

MAX_PROCESSES = 4096


def process_id_buffer(capacity):
    if type(capacity) is not int or not 1 <= capacity <= MAX_PROCESSES:
        raise ValueError('bounded Job PID capacity required')

    class ProcessIds(ctypes.Structure):
        _fields_ = [('assigned', DWORD), ('count', DWORD), ('pids', SIZE * capacity)]

    return ProcessIds()


class Job(BaseJob):
    def __init__(self):
        super().__init__()
        try:
            self.api.IsProcessInJob.argtypes = [HANDLE, HANDLE, ctypes.POINTER(ctypes.c_int)]
            self.api.IsProcessInJob.restype = ctypes.c_int
            self.api.QueryFullProcessImageNameW.argtypes = [HANDLE, DWORD, ctypes.c_wchar_p, ctypes.POINTER(DWORD)]
            self.api.QueryFullProcessImageNameW.restype = ctypes.c_int
        except BaseException:
            self.close()
            raise

    def process_ids(self):
        capacity = 16
        for _ in range(5):
            value = process_id_buffer(capacity)
            okay = self.api.QueryInformationJobObject(self.handle, 3, ctypes.byref(value), ctypes.sizeof(value), None)
            if value.assigned > MAX_PROCESSES:
                raise ValueError('owned Job process count exceeds bound')
            if okay:
                if value.count > capacity or value.assigned < value.count:
                    raise ValueError('inconsistent owned Job process count')
                if value.count == value.assigned:
                    pids = list(value.pids[:value.count])
                    if len(set(pids)) != len(pids) or any(not 0 < pid <= 0xffffffff for pid in pids):
                        raise ValueError('invalid or duplicate owned Job PIDs')
                    return {'assigned_count': int(value.assigned), 'pids': sorted(pids)}
            elif (error := ctypes.get_last_error()) != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(error)
            capacity = max(capacity * 2, int(value.assigned))
            if capacity > MAX_PROCESSES:
                raise ValueError('owned Job process count exceeds bound')
        raise RuntimeError('owned Job process membership did not stabilize')

    def snapshot(self):
        value = self.process_ids()
        records = []
        for pid in value['pids']:
            record = {'pid': pid, 'membership_confirmed': False}
            records.append(record)
            handle = self.api.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION only.
            if not handle:
                record['open_error'] = ctypes.get_last_error()
                continue
            try:
                member = ctypes.c_int()
                if not self.api.IsProcessInJob(handle, self.handle, ctypes.byref(member)):
                    record['membership_error'] = ctypes.get_last_error()
                    continue
                if not member.value:
                    record['membership_changed'] = True
                    continue  # Never query a potentially reused PID outside the owned Job.
                record['membership_confirmed'] = True
                image, size = ctypes.create_unicode_buffer(32768), DWORD(32768)
                if self.api.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
                    if not image.value or not 0 < size.value < 32768:
                        raise ValueError('invalid owned process image length')
                    record['image'] = image.value
                else:
                    record['image_error'] = ctypes.get_last_error()
            finally:
                if not self.api.CloseHandle(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
        return {'assigned_count': value['assigned_count'], 'processes': records,
                'limits': 'Job membership snapshot; processes can exit between enumeration and image query.'}


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
    observation = {'worker_pid': None, 'worker_assigned': False, 'normal_worker_exit_observed': False,
        'observed_worker_exit_code': None, 'reaped_worker_exit_code': None, 'remaining_before_cleanup': None,
        'process_tree_reaped': False, 'job_handle_closed': False, 'cleanup_errors': []}
    try:
        worker = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--job-child'],
            cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, bufsize=0)
        observation['worker_pid'] = worker.pid
        job.assign(worker)
        observation['worker_assigned'] = True
        try:
            worker.communicate(input=payload, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise subprocess.TimeoutExpired(command, timeout) from error
        observation['normal_worker_exit_observed'] = True
        observation['observed_worker_exit_code'] = worker.returncode
        job.empty(timeout=.25)  # Brief teardown grace; cleanup still drains any owned background work.
    except BaseException as error:
        failure = error
    finally:
        try:
            if worker is not None and observation['observed_worker_exit_code'] is None:
                observation['observed_worker_exit_code'] = worker.poll()
        except BaseException as error:
            if failure is None:
                failure = error
        try:
            observation['remaining_before_cleanup'] = job.snapshot()
        except BaseException as error:
            observation['snapshot_error'] = repr(error)
            if failure is None:
                failure = error
        try:
            job.terminate()
        except BaseException as error:
            observation['cleanup_errors'].append(repr(error))
        try:
            if worker is not None:
                # If assignment failed this gated worker is outside the Job,
                # with no payload and no descendants; it still must be reaped.
                if worker.poll() is None:
                    worker.kill()
                worker.wait(timeout=5)
                observation['reaped_worker_exit_code'] = worker.returncode
                if worker.stdin is not None and not worker.stdin.closed:
                    worker.stdin.close()
        except BaseException as error:
            observation['cleanup_errors'].append(repr(error))
        try:
            if not job.empty() or (worker is not None and observation['reaped_worker_exit_code'] is None):
                raise RuntimeError('owned process-tree termination not verified')
            observation['process_tree_reaped'] = True
        except BaseException as error:
            observation['cleanup_errors'].append(repr(error))
        try:
            job.close()
            observation['job_handle_closed'] = True
        except BaseException as error:
            observation['cleanup_errors'].append(repr(error))
    if observation['cleanup_errors']:
        if failure is not None:
            observation['command_error'] = repr(failure)
        failure = RuntimeError('owned process cleanup failed: ' + repr(observation['cleanup_errors']))
    if failure is not None:
        failure.process_tree_reaped = observation['process_tree_reaped']
        failure.job_observation = observation
        raise failure
    result = subprocess.CompletedProcess(command, observation['observed_worker_exit_code'])
    result.process_tree_reaped = observation['process_tree_reaped']
    result.job_observation = observation
    return result


if __name__ == '__main__':
    if sys.argv[1:] != ['--job-child'] or platform.system() != 'Windows':
        raise ValueError('internal gated Windows Job worker only')
    status = _child()
    exit_process = ctypes.WinDLL('kernel32').ExitProcess
    exit_process.argtypes, exit_process.restype = [DWORD], None
    exit_process(DWORD(status))
