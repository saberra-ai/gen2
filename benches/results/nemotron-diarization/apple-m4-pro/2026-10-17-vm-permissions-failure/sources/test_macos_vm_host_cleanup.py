"""Failure at PID persistence or diagnostic writes cannot orphan cleanup work."""
import signal
import subprocess
import unittest
from unittest.mock import Mock, patch

from qualify_macos_vm_host import cleanup_metadata, execute_child


class VMHostCleanupTests(unittest.TestCase):
    def test_pid_write_failure_kills_and_reaps_the_started_group(self):
        process = Mock(pid=12345)
        process.poll.return_value = None
        process.wait.return_value = -signal.SIGKILL
        guard, record = Mock(), {}
        with patch('qualify_macos_vm_host.subprocess.Popen', return_value=process), \
                patch('qualify_macos_vm_host.os.killpg') as kill:
            with self.assertRaises(OSError):
                execute_child(['fake'], cwd='.', environment={}, record=record,
                    persist=Mock(side_effect=OSError('disk full')), timeout=1, guard=guard)
        kill.assert_called_once_with(12345, signal.SIGKILL)
        process.wait.assert_called_once()
        guard.wait.assert_not_called()
        guard.release.assert_called_once()
        self.assertTrue(record['forced_process_group_termination'])

    def test_cleanup_captures_despite_diagnostic_failure_without_storage_or_pid_write(self):
        qualification = {}
        cleanup_metadata(Mock(side_effect=OSError('disk full')), qualification)
        process = Mock(pid=12345, returncode=0)
        process.communicate.return_value = (b'owned image', b'')
        persist = Mock(side_effect=AssertionError('cleanup must not persist before command'))
        with patch('qualify_macos_vm_host.subprocess.Popen', return_value=process):
            self.assertEqual(execute_child(['fake-detach'], cwd='.', environment={}, record={},
                persist=persist, timeout=1), (b'owned image', b''))
        persist.assert_not_called()
        self.assertIn('disk full', qualification['cleanup_metadata_errors'][0])

    def test_cleanup_timeout_still_kills_and_reaps_the_group(self):
        process = Mock(pid=12345)
        process.communicate.side_effect = subprocess.TimeoutExpired(['fake'], 1)
        process.poll.return_value = None
        process.wait.return_value = -signal.SIGKILL
        record = {}
        with patch('qualify_macos_vm_host.subprocess.Popen', return_value=process), \
                patch('qualify_macos_vm_host.os.killpg') as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                execute_child(['fake'], cwd='.', environment={}, record=record, persist=Mock(), timeout=1)
        kill.assert_called_once_with(12345, signal.SIGKILL)
        process.wait.assert_called_once()
        self.assertEqual(record['returncode'], -signal.SIGKILL)
