"""Keep diagnostic capacity and atomic metadata during bounded qualification."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


class StoragePressureError(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.next')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class StorageGuard:
    def __init__(self, output, start_minimum, running_minimum, reserve_bytes):
        self.output = Path(output)
        self.start_minimum, self.running_minimum = start_minimum, running_minimum
        self.reserve = self.output / 'diagnostic-reserve.bin'
        self.record = {'start_minimum_bytes': start_minimum, 'running_minimum_bytes': running_minimum,
                       'reserve_requested_bytes': reserve_bytes, 'checks': 0, 'released': False}
        if not 0 < running_minimum < start_minimum or reserve_bytes <= 0:
            raise ValueError('invalid qualification storage limits')
        if self.reserve.exists():
            raise FileExistsError('refusing an existing diagnostic reserve')
        self.before_model()
        try:
            with self.reserve.open('xb') as stream:
                remaining = reserve_bytes
                block = b'\0' * min(1024 * 1024, reserve_bytes)
                while remaining:
                    written = stream.write(block[:remaining])
                    if not written:
                        raise OSError('failed to allocate diagnostic reserve')
                    remaining -= written
                stream.flush()
                os.fsync(stream.fileno())
            allocated = self.reserve.stat().st_blocks * 512
            self.record['reserve_reported_allocated_bytes'] = allocated
            if allocated < reserve_bytes:
                raise StoragePressureError('diagnostic reserve was not physically allocated')
            self.before_model()
        except BaseException:
            self.release()
            raise

    def check(self):
        free = shutil.disk_usage(self.output).free
        self.record['checks'] += 1
        self.record['minimum_observed_free_bytes'] = min(free, self.record.get('minimum_observed_free_bytes', free))
        self.record['last_observed_free_bytes'] = free
        return free

    def before_model(self):
        free = self.check()
        if free < self.start_minimum:
            raise StoragePressureError('insufficient free storage before native admission: ' + str(free))
        return free

    def release(self):
        self.reserve.unlink(missing_ok=True)
        self.record['released'] = True

    def wait(self, process, record, timeout):
        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(record['argv'], timeout)
                try:
                    return process.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    free = self.check()
                    if free < self.running_minimum:
                        record['disk_pressure_abort_free_bytes'] = free
                        raise StoragePressureError('free storage fell below running threshold')
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    record['forced_process_group_termination'] = True
                except ProcessLookupError:
                    pass
                process.wait()
            record['returncode'] = process.returncode
            self.release()
            raise
