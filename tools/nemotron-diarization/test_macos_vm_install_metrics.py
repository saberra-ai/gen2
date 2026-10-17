"""A callback, process exit or stopped claim alone cannot admit a VM seed."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from macos_vm_install_metrics import owner_metrics, installed_seed_metrics


class InstallMetricsTests(unittest.TestCase):
    def evidence(self):
        value = {'schema_version': 1, 'installation_qualified': True, 'virtual_machine_created': True,
            'installation_started': True, 'installer_callback_received': True, 'installer_callback_success': True,
            'cancellation_requested': False, 'cleanup_confirmed': True, 'final_state': 0,
            'normal_start_called': False, 'custom_listener_created': False, 'display_window_created': False,
            'logging_failed': False, 'errors': [], 'event_count': 5,
            'configuration_validated': True, 'cpu_count': 4, 'memory_bytes': 8 * 1024**3,
            'stop_requested_after_installer_callback': False, 'stop_callback_received': False, 'stop_callback_success': False}
        events = [{'event': name, 'elapsed_seconds': index, 'state': 0} for index, name in enumerate(
            ['configuration_validated', 'virtual_machine_created', 'installation_started', 'installer_completion', 'terminal'])]
        events[0].update(cpu_count=4, memory_bytes=8 * 1024**3)
        events[3]['success'] = True
        events[-1].update(success=True, cleanup_confirmed=True)
        return value, {'returncode': 0}, events

    def test_complete_callback_stopped_and_exit_evidence_passes(self):
        value, command, events = self.evidence()
        self.assertEqual(owner_metrics(value, command, events), value)

    def test_cancelled_killed_unconfirmed_or_callback_only_rejected(self):
        for key, bad in [('cancellation_requested', True), ('cleanup_confirmed', False), ('final_state', 1),
                ('installer_callback_received', False), ('logging_failed', True), ('errors', [[{'code': 10006}]])]:
            value, command, events = self.evidence()
            value[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                owner_metrics(value, command, events)
        for key, bad in [('returncode', 1), ('abort_error', 'timeout'), ('forced_process_group_termination', True)]:
            value, command, events = self.evidence()
            command[key] = bad
            with self.subTest(command=key), self.assertRaises(ValueError):
                owner_metrics(value, command, events)

    def test_stop_must_follow_installer_callback_and_have_successful_completion(self):
        value, command, events = self.evidence()
        value.update(stop_requested_after_installer_callback=True, stop_callback_received=True, stop_callback_success=True, event_count=7)
        events[4:4] = [{'event': 'post_callback_destructive_stop_requested', 'elapsed_seconds': 3.1, 'state': 1},
            {'event': 'post_callback_stop_completion', 'elapsed_seconds': 3.2, 'state': 0}]
        self.assertEqual(owner_metrics(value, command, events), value)
        wrong = copy.deepcopy(events)
        wrong[2]['event'], wrong[4]['event'] = wrong[4]['event'], wrong[2]['event']
        with self.assertRaises(ValueError):
            owner_metrics(value, command, wrong)
        value['stop_callback_success'] = False
        with self.assertRaises(ValueError):
            owner_metrics(value, command, events)

    def seed(self, temporary, *, written):
        root = Path(temporary) / 'seed'
        root.mkdir(mode=0o700)
        original = {}
        for name in ['hardware-model.bin', 'machine-id.bin', 'auxiliary-storage.bin']:
            data = name.encode()
            (root / name).write_bytes(data)
            (root / name).chmod(0o600)
            original[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        with (root / 'disk.img').open('xb') as stream:
            stream.truncate(100 * 1024**3)
            if written:
                stream.write(b'installed control')
                stream.flush()
                os.fsync(stream.fileno())
        (root / 'disk.img').chmod(0o600)
        original['disk.img'] = {'bytes': 100 * 1024**3, 'sha256': None}
        return root, original

    def test_installed_sparse_disk_is_stat_only_with_unchanged_platform_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, original = self.seed(temporary, written=True)
            read = Path.read_bytes
            def bounded_read(path):
                if path.name == 'disk.img':
                    raise AssertionError('must not read sparse disk')
                return read(path)
            with patch.object(Path, 'read_bytes', bounded_read):
                result = installed_seed_metrics(root, original, uid=os.getuid())
            self.assertIsNone(result['disk.img']['sha256'])
            self.assertGreater(result['disk.img']['allocated_bytes'], 0)

    def test_unwritten_disk_or_changed_installed_identity_rejects(self):
        for mutation in ['unwritten', 'identity', 'mode', 'symlink']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root, original = self.seed(temporary, written=mutation != 'unwritten')
                if mutation == 'identity':
                    (root / 'machine-id.bin').write_bytes(b'other')
                elif mutation == 'mode':
                    (root / 'disk.img').chmod(0o644)
                elif mutation == 'symlink':
                    (root / 'machine-id.bin').unlink()
                    (root / 'machine-id.bin').symlink_to(root / 'hardware-model.bin')
                with self.assertRaises(ValueError):
                    installed_seed_metrics(root, original, uid=os.getuid())
