"""Real filesystem controls for reversible, descriptor-bound owner search."""
import os
from pathlib import Path
import signal
import stat
import sys
import tempfile
import unittest

from macos_vm_search_bit_control import control


@unittest.skipUnless(os.name == 'posix' and os.geteuid() != 0, 'non-root POSIX filesystem control')
class SearchBitControlTests(unittest.TestCase):
    def fixture_security(self, parent, name):
        return {'status': 'ok', 'raw_base64': 'fixture', 'acl_policy_qualified': False}

    def assert_restored(self, root, report):
        self.assertEqual(stat.S_IMODE((root / 'Default').stat().st_mode), 0o600)
        self.assertTrue(report['restoration_confirmed'])
        self.assertFalse(report['guest_access_qualified'])
        self.assertFalse(report['acl_policy_qualified'])

    def test_actual_directory_lookup_changes_and_original_mode_restores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = Path(temporary) / 'fresh', {}
            control(root, report, query=self.fixture_security)
            self.assertTrue(report['control_passed'])
            self.assertEqual(report['denials_before'], {'users': 13, 'groups': 13})
            self.assertEqual(set(report['children']), {'users', 'groups'})
            self.assert_restored(root, report)

    def test_existing_directory_and_symlink_are_never_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'existing').mkdir(mode=0o711)
            (root / 'link').symlink_to(root / 'existing', target_is_directory=True)
            for path in [root / 'existing', root / 'link']:
                with self.assertRaises(FileExistsError):
                    control(path, {}, query=self.fixture_security)
            self.assertEqual(stat.S_IMODE((root / 'existing').stat().st_mode), 0o711)

    def test_mutator_failure_after_changing_mode_still_restores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report, calls = Path(temporary) / 'fresh', {}, []
            def changer(fd, mode):
                calls.append(mode)
                os.fchmod(fd, mode)
                if mode == 0o700:
                    raise OSError('injected post-mutation failure')
            with self.assertRaisesRegex(OSError, 'post-mutation'):
                control(root, report, query=self.fixture_security, changer=changer)
            self.assertEqual(calls, [0o700, 0o600])
            self.assertFalse(report['control_passed'])
            self.assert_restored(root, report)

    def test_base_exception_restores_and_closes_descriptor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report, seen = Path(temporary) / 'fresh', {}, []
            def interrupt(fd):
                seen.append(fd)
                raise KeyboardInterrupt('injected interruption')
            with self.assertRaises(KeyboardInterrupt):
                control(root, report, query=self.fixture_security, observe=interrupt)
            self.assert_restored(root, report)
            with self.assertRaises(OSError):
                os.fstat(seen[0])

    def test_security_change_is_retained_and_not_called_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report, queries = Path(temporary) / 'fresh', {}, []
            def changed(parent, name):
                queries.append(name)
                return {'status': 'ok', 'raw_base64': str(len(queries))}
            with self.assertRaisesRegex(ValueError, 'security restoration'):
                control(root, report, query=changed)
            self.assertEqual(stat.S_IMODE((root / 'Default').stat().st_mode), 0o600)
            self.assertFalse(report['restoration_confirmed'])
            self.assertIn('restoration_error', report)

    def test_metadata_change_does_not_skip_restoring_same_inode_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = Path(temporary) / 'fresh', {}
            def touch(fd):
                before = os.fstat(fd)
                os.utime(fd, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
            with self.assertRaisesRegex(ValueError, 'security restoration'):
                control(root, report, query=self.fixture_security, observe=touch)
            self.assertEqual(stat.S_IMODE((root / 'Default').stat().st_mode), 0o600)
            self.assertFalse(report['restoration_confirmed'])

    def test_replaced_path_does_not_change_replacement_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = Path(temporary) / 'fresh', {}
            def replace(fd):
                (root / 'Default').rename(root / 'displaced')
                (root / 'Default').mkdir(mode=0o711)
            with self.assertRaisesRegex(ValueError, 'no longer occupies'):
                control(root, report, query=self.fixture_security, observe=replace)
            self.assertEqual(stat.S_IMODE((root / 'displaced').stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE((root / 'Default').stat().st_mode), 0o711)
            self.assertFalse(report['restoration_confirmed'])

    def test_replaced_root_does_not_qualify_original_fixture_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = Path(temporary) / 'fresh', {}
            displaced = Path(temporary) / 'displaced'
            def replace(fd):
                root.rename(displaced)
                root.mkdir(mode=0o711)
            with self.assertRaisesRegex(ValueError, 'root no longer occupies'):
                control(root, report, query=self.fixture_security, observe=replace)
            self.assertEqual(stat.S_IMODE((displaced / 'Default').stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o711)
            self.assertFalse(report['restoration_confirmed'])

    def test_failed_restore_remains_failed_and_fixture_can_be_recovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report = Path(temporary) / 'fresh', {}
            def changer(fd, mode):
                if mode == 0o600:
                    raise PermissionError('injected restore denial')
                os.fchmod(fd, mode)
            with self.assertRaisesRegex(PermissionError, 'restore denial'):
                control(root, report, query=self.fixture_security, changer=changer)
            self.assertFalse(report['control_passed'])
            self.assertFalse(report['restoration_confirmed'])
            self.assertIn('restoration_error', report)
            self.assertEqual(stat.S_IMODE((root / 'Default').stat().st_mode), 0o700)
            os.chmod(root / 'Default', 0o600)  # Unit-test fixture cleanup only.

    def test_pending_signal_during_restore_cannot_skip_mode_restoration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report, phases = Path(temporary) / 'fresh', {}, []
            def handler(number, frame):
                phases.append('signal')
                raise InterruptedError('deferred ordinary termination')
            def changer(fd, mode):
                os.fchmod(fd, mode)
                if mode == 0o600:
                    phases.append('restoring')
                    os.kill(os.getpid(), signal.SIGTERM)
                    phases.append('restored')
            previous = signal.signal(signal.SIGTERM, handler)
            try:
                with self.assertRaisesRegex(InterruptedError, 'deferred'):
                    control(root, report, query=self.fixture_security, changer=changer)
            finally:
                signal.signal(signal.SIGTERM, previous)
            self.assertEqual(phases, ['restoring', 'restored', 'signal'])
            self.assertFalse(report['control_passed'])
            self.assert_restored(root, report)

    def test_signal_at_cleanup_entry_is_deferred_until_after_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, report, delivered = Path(temporary) / 'fresh', {}, []
            def handler(number, frame):
                delivered.append('signal')
                raise InterruptedError('cleanup-entry termination')
            def tracer(frame, event, arg):
                # The observer marks the transition; the first subsequent line
                # in control is the cleanup-entry signal injection point.
                if frame.f_code is control.__code__ and event == 'line' and delivered == ['observe']:
                    delivered.append('cleanup')
                    os.kill(os.getpid(), signal.SIGTERM)
                return tracer
            def observe(fd):
                delivered.append('observe')
            previous = signal.signal(signal.SIGTERM, handler)
            old_trace = sys.gettrace()
            try:
                sys.settrace(tracer)
                with self.assertRaisesRegex(InterruptedError, 'cleanup-entry'):
                    control(root, report, query=self.fixture_security, observe=observe)
            finally:
                sys.settrace(old_trace)
                signal.signal(signal.SIGTERM, previous)
            self.assertEqual(delivered, ['observe', 'cleanup', 'signal'])
            self.assertFalse(report['control_passed'])
            self.assert_restored(root, report)
