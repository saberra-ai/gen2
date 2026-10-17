"""Failure controls for enforced clones and phase-aware installer supervision."""
import errno
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from macos_vm_seed_clone import clone_regular
from macos_vm_install_supervision import execute_install_owner


class InstallControlTests(unittest.TestCase):
    def test_clone_error_never_falls_back_or_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary) / 'source', Path(temporary) / 'destination'
            source.write_bytes(b'original')
            source.chmod(0o600)
            destination.mkdir(mode=0o700)
            with patch('macos_vm_seed_clone._fclone', side_effect=OSError(errno.EXDEV, 'cross-volume')) as clone:
                with self.assertRaises(OSError):
                    clone_regular(source, destination, 'disk')
                clone.assert_called_once()
                self.assertEqual(source.read_bytes(), b'original')
                self.assertFalse((destination / 'disk').exists())
                (destination / 'disk').symlink_to(source)
                clone.reset_mock()
                with self.assertRaises(FileExistsError):
                    clone_regular(source, destination, 'disk')
                clone.assert_not_called()

    def test_clone_source_symlink_rejected_before_syscall(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source'
            source.symlink_to(Path(temporary) / 'absent')
            with patch('macos_vm_seed_clone._fclone') as clone:
                with self.assertRaises(OSError):
                    clone_regular(source, temporary, 'target')
                clone.assert_not_called()

    def supervise(self, process, guard, persist, record):
        with patch('macos_vm_install_supervision.subprocess.Popen', return_value=process):
            execute_install_owner(['owner'], cwd='.', environment={}, record=record, persist=persist,
                guard=guard, timeout=1, cleanup_timeout=2, stdout=Mock(), stderr=Mock())

    def test_pid_persistence_failure_requests_cancel_then_reaps_without_kill(self):
        process = Mock(pid=12345)
        process.poll.side_effect = [None, 1]
        process.wait.return_value = 1
        record, guard = {}, Mock()
        with patch('macos_vm_install_supervision.os.killpg') as kill:
            with self.assertRaises(OSError):
                self.supervise(process, guard, Mock(side_effect=OSError('metadata full')), record)
        kill.assert_called_once_with(12345, signal.SIGTERM)
        self.assertEqual(record['returncode'], 1)
        self.assertNotIn('forced_process_group_termination', record)
        guard.release.assert_called_once()

    def test_unresponsive_owner_gets_cancel_grace_then_kill_and_reap(self):
        process = Mock(pid=12345)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired(['owner'], 1),
            subprocess.TimeoutExpired(['owner'], 2), -signal.SIGKILL]
        record, guard = {}, Mock(running_minimum=1)
        guard.check.return_value = 0
        with patch('macos_vm_install_supervision.os.killpg') as kill:
            with self.assertRaises(OSError):
                self.supervise(process, guard, Mock(), record)
        self.assertEqual([call.args for call in kill.call_args_list], [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])
        self.assertEqual(process.wait.call_args_list[1].kwargs, {'timeout': 2})
        self.assertTrue(record['owner_exit_does_not_prove_vm_teardown'])
        self.assertTrue(record['forced_process_group_termination'])

    def test_success_does_not_signal_or_claim_guest_cleanup(self):
        process = Mock(pid=12345)
        process.wait.return_value = 0
        record = {}
        with patch('macos_vm_install_supervision.os.killpg') as kill:
            self.supervise(process, Mock(), Mock(), record)
        kill.assert_not_called()
        self.assertEqual(record['returncode'], 0)
        self.assertNotIn('owner_exit_does_not_prove_vm_teardown', record)
