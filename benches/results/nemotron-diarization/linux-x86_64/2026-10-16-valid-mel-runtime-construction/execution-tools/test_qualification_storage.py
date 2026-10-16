import errno
import json
from pathlib import Path
import signal
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qualification_storage import StorageGuard, StoragePressureError, atomic_json


class QualificationStorageTests(unittest.TestCase):
    def test_reserve_is_allocated_and_released_and_low_start_space_rejects_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('qualification_storage.shutil.disk_usage', return_value=SimpleNamespace(free=100000)):
                guard = StorageGuard(root, 50000, 10000, 8192)
                self.assertGreaterEqual(guard.record['reserve_reported_allocated_bytes'], 8192)
                self.assertEqual(guard.reserve.stat().st_size, 8192)
                guard.release(); guard.release()
                self.assertFalse(guard.reserve.exists())
            with patch('qualification_storage.shutil.disk_usage', return_value=SimpleNamespace(free=49999)):
                with self.assertRaises(StoragePressureError):
                    StorageGuard(root, 50000, 10000, 8192)
            self.assertFalse((root / 'diagnostic-reserve.bin').exists())
            reserve = root / 'diagnostic-reserve.bin'; reserve.write_bytes(b'prior attempt')
            with self.assertRaises(FileExistsError):
                StorageGuard(root, 50000, 10000, 8192)
            self.assertEqual(reserve.read_bytes(), b'prior attempt')

    def test_disk_pressure_kills_process_group_and_releases_diagnostic_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('qualification_storage.shutil.disk_usage', return_value=SimpleNamespace(free=100000)):
                guard = StorageGuard(directory, 50000, 10000, 8192)
            process = Mock(pid=123, returncode=-9)
            process.wait.side_effect = [subprocess.TimeoutExpired(['fixture'], 1), -9]
            process.poll.return_value = None
            record = {'argv': ['fixture']}
            with patch('qualification_storage.shutil.disk_usage', return_value=SimpleNamespace(free=9999)), patch('qualification_storage.os.killpg') as kill:
                with self.assertRaises(StoragePressureError):
                    guard.wait(process, record, 5)
            kill.assert_called_once_with(123, signal.SIGKILL)
            self.assertEqual(record['returncode'], -9)
            self.assertEqual(record['disk_pressure_abort_free_bytes'], 9999)
            self.assertTrue(guard.record['released'])
            self.assertFalse(guard.reserve.exists())

    def test_deadline_kills_group_and_releases_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('qualification_storage.shutil.disk_usage', return_value=SimpleNamespace(free=100000)):
                guard = StorageGuard(directory, 50000, 10000, 8192)
            process = Mock(pid=123, returncode=-9)
            process.wait.side_effect = [subprocess.TimeoutExpired(['fixture'], 1), -9]
            process.poll.return_value = None
            record = {'argv': ['fixture']}
            with patch('qualification_storage.shutil.disk_usage', return_value=SimpleNamespace(free=100000)), patch('qualification_storage.time.monotonic', side_effect=[0, 0, 2]), patch('qualification_storage.os.killpg') as kill:
                with self.assertRaises(subprocess.TimeoutExpired):
                    guard.wait(process, record, 1)
            kill.assert_called_once_with(123, signal.SIGKILL)
            self.assertTrue(record['forced_process_group_termination'])
            self.assertTrue(guard.record['released'])

    def test_atomic_replacement_failure_keeps_last_complete_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'commands.json'
            atomic_json(path, {'status': 'started'})
            with patch('qualification_storage.Path.replace', side_effect=OSError(errno.ENOSPC, 'fixture disk full')):
                with self.assertRaises(OSError):
                    atomic_json(path, {'status': 'finished'})
            self.assertEqual(json.loads(path.read_text()), {'status': 'started'})
            self.assertEqual(json.loads(path.with_name(path.name + '.next').read_text()), {'status': 'finished'})


if __name__ == '__main__':
    unittest.main()
