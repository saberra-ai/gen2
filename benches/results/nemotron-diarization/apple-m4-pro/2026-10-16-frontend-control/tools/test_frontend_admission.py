import json
from pathlib import Path
import sys
import tempfile
import unittest

from frontend_admission import require_linked_sdk, run_probe


class FrontendAdmissionTests(unittest.TestCase):
    def test_copied_sdk_rejected_even_if_same_library_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual, copied = root / 'linked-sdk', root / 'copied-sdk'
            for path in [actual, copied]:
                (path / 'lib').mkdir(parents=True)
                (path / 'lib/library').write_bytes(b'identical library bytes')
            rpaths = [str(actual.resolve() / 'lib')]
            require_linked_sdk(actual, str(actual), rpaths)
            with self.assertRaises(ValueError):
                require_linked_sdk(copied, str(actual), rpaths)
            with self.assertRaises(ValueError):
                require_linked_sdk(actual, str(actual), [*rpaths, '/other/lib'])

    def test_nonzero_probe_preserves_command_status_and_both_streams(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            argv = [sys.executable, '-c', 'import sys; print("observed"); print("cause", file=sys.stderr); sys.exit(7)']
            with self.assertRaises(ValueError):
                run_probe(argv, output, 'failure', {})
            self.assertEqual((output / 'failure.stdout.log').read_text().strip(), 'observed')
            self.assertEqual((output / 'failure.stderr.log').read_text().strip(), 'cause')
            record = json.loads((output / 'failure.command.json').read_text())
            self.assertEqual(record['returncode'], 7)
            self.assertEqual(record['argv'], argv)

    def test_startup_error_retains_attempt_and_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaises(OSError):
                run_probe([str(output / 'nonexistent')], output, 'startup', {})
            record = json.loads((output / 'startup.command.json').read_text())
            self.assertIsNone(record['returncode'])
            self.assertTrue(record['startup_error'])
            self.assertTrue((output / 'startup.stdout.log').exists())
            self.assertTrue((output / 'startup.stderr.log').exists())


if __name__ == '__main__':
    unittest.main()
