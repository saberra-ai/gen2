import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from evidence_commands import CommandRunner


class EvidenceCommandTests(unittest.TestCase):
    def test_utf8_output_survives_windows_legacy_locale_without_byte_changes(self):
        original = io.open
        def legacy_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
            if 'b' not in mode and encoding in {None, 'locale'}:
                encoding = 'cp1252'
            return original(file, mode, buffering, encoding, errors, newline, closefd, opener)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = CommandRunner(root, root)
            script = "import sys;sys.stdout.buffer.write('C:/inputs ü'.encode('utf-8'))"
            with patch('io.open', side_effect=legacy_open):
                result = runner.run('docker-json', [sys.executable, '-c', script])
            self.assertEqual(result, 'C:/inputs ü')
            self.assertEqual((root / 'docker-json.stdout.log').read_bytes(), 'C:/inputs ü'.encode('utf-8'))


if __name__ == '__main__':
    unittest.main()
