import hashlib
from pathlib import Path
import tempfile
import unittest

from windows_chat_inputs import model_inventory


class ChatInputsTests(unittest.TestCase):
    def test_exact_locked_large_model_inventory_without_weakening_runtime_archive(self):
        from windows_archive import MAX_FILE_BYTES, byte_inventory
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'model.gguf'
            path.write_bytes(b'abc')
            lock = {'artifacts': [{'name': path.name, 'size': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}]}
            self.assertEqual(model_inventory(root, lock)[path.name]['bytes'], 3)
            path.write_bytes(b'abd')
            with self.assertRaises(ValueError):
                model_inventory(root, lock)
            path.write_bytes(b'abc')
            (root / 'extra').write_bytes(b'')
            with self.assertRaises(ValueError):
                model_inventory(root, lock)
            (root / 'extra').unlink()
            path.unlink()
            path.symlink_to(root / 'missing')
            with self.assertRaises(ValueError):
                model_inventory(root, lock)
            path.unlink()
            with path.open('wb') as output:
                output.truncate(MAX_FILE_BYTES + 1)
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                byte_inventory(root)
