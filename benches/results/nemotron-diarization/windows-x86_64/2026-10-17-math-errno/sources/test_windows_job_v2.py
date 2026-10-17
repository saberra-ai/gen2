"""Actual owned-background cleanup and membership-scoped observation controls."""
import ctypes
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_windows_job as original_tests
from windows_job_v2 import Job, process_id_buffer, run_in_job


class JobV2ControlTest(original_tests.JobControlTest):
    def setUp(self):
        for name, value in [('run_in_job', run_in_job), ('Job', Job)]:
            replacement = patch.object(original_tests, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_process_id_layout_and_allocation_bounds(self):
        value = process_id_buffer(16)
        self.assertEqual(type(value).pids.offset, 8)
        self.assertEqual(ctypes.sizeof(value), 8 + 16 * ctypes.sizeof(ctypes.c_size_t))
        for invalid in (0, -1, 4097, True):
            with self.assertRaises(ValueError):
                process_id_buffer(invalid)

    def test_invalid_pid_observations_fail_closed(self):
        for pids, assigned in [([7, 7], 2), ([0], 1), ([7], 5000)]:
            def query(_handle, _kind, pointer, _size, _length):
                value = pointer._obj
                value.assigned, value.count = assigned, len(pids)
                for index, pid in enumerate(pids):
                    value.pids[index] = pid
                return 1
            job = object.__new__(Job)
            job.handle, job.api = 1, SimpleNamespace(QueryInformationJobObject=query)
            with self.subTest(pids=pids, assigned=assigned), self.assertRaises(ValueError):
                job.process_ids()

    def test_reused_pid_outside_job_is_never_image_queried(self):
        def member(_process, _job, pointer):
            pointer._obj.value = 0
            return 1
        job = object.__new__(Job)
        job.handle = 1
        job.process_ids = lambda: {'assigned_count': 1, 'pids': [123]}
        image = Mock(side_effect=AssertionError('must not inspect an unowned process'))
        close = Mock(return_value=1)
        job.api = SimpleNamespace(OpenProcess=lambda *_: 456, IsProcessInJob=member,
                                  QueryFullProcessImageNameW=image, CloseHandle=close)
        result = job.snapshot()
        self.assertFalse(result['processes'][0]['membership_confirmed'])
        self.assertTrue(result['processes'][0]['membership_changed'])
        image.assert_not_called()
        close.assert_called_once_with(456)

    def test_successful_partial_pid_list_is_retried(self):
        calls = []
        def query(_handle, _kind, pointer, _size, _length):
            value = pointer._obj
            capacity = len(value.pids)
            calls.append(capacity)
            value.assigned, value.count = 20, min(20, capacity)
            for index in range(value.count):
                value.pids[index] = index + 1
            return 1
        job = object.__new__(Job)
        job.handle, job.api = 1, SimpleNamespace(QueryInformationJobObject=query)
        self.assertEqual(job.process_ids(), {'assigned_count': 20, 'pids': list(range(1, 21))})
        self.assertEqual(calls, [16, 32])

    @unittest.skipUnless(platform.system() == 'Windows', 'actual Windows Job API required')
    def test_normal_background_cleanup_preserves_zero_and_nonzero_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            late_files = []
            for status in (0, 9):
                late, spawned = root / ('late-' + str(status)), root / ('spawned-' + str(status))
                late_files.append(late)
                grandchild = 'import time,pathlib;time.sleep(4);pathlib.Path(' + repr(str(late)) + ').write_text("escaped")'
                command = 'import subprocess,sys,pathlib;p=subprocess.Popen([sys.executable,"-c",' + repr(grandchild) + ']);pathlib.Path(' + repr(str(spawned)) + ').write_text(str(p.pid));sys.exit(' + str(status) + ')'
                with (root / ('out-' + str(status))).open('wb') as out, (root / ('err-' + str(status))).open('wb') as err:
                    result = run_in_job([sys.executable, '-c', command], cwd=root, env=os.environ.copy(),
                        stdout=out, stderr=err, timeout=15)
                self.assertEqual(result.returncode, status)
                observation = result.job_observation
                self.assertTrue(observation['normal_worker_exit_observed'])
                self.assertEqual(observation['observed_worker_exit_code'], status)
                self.assertEqual(observation['reaped_worker_exit_code'], status)
                self.assertTrue(observation['worker_assigned'])
                self.assertTrue(observation['process_tree_reaped'])
                self.assertTrue(observation['job_handle_closed'])
                self.assertEqual(observation['cleanup_errors'], [])
                descendant_pid = int(spawned.read_text())
                descendant, = [row for row in observation['remaining_before_cleanup']['processes'] if row['pid'] == descendant_pid]
                self.assertTrue(descendant['membership_confirmed'])
                self.assertTrue(Path(descendant['image']).samefile(sys.executable))
            time.sleep(4.2)
            self.assertFalse(any(path.exists() for path in late_files))


if __name__ == '__main__':
    unittest.main()
