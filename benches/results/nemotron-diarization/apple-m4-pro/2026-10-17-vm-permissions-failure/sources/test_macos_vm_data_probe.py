"""Exercise descriptor confinement and remove probe writes without a VM/mount."""
import os
from pathlib import Path
import tempfile
import unittest

from macos_vm_data_probe import bounded_file, directory, existing_write_open, write_probe


class DataProbeTests(unittest.TestCase):
    def test_real_create_fsync_readback_mode_and_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'private/var/db').mkdir(parents=True)
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                report = write_probe(fd, 'private/var/db', 'gen2-probe')
                self.assertTrue(report['probe_removed'])
                self.assertEqual(report['probe']['mode'], 0o600)
                self.assertEqual(list((root / 'private/var/db').iterdir()), [])
            finally:
                os.close(fd)

    def test_symlink_and_parent_escape_cannot_read_or_write_outside(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'guest').mkdir()
            (root / 'outside').mkdir()
            (root / 'outside/secret').write_bytes(b'outside')
            (root / 'guest/private').symlink_to(root / 'outside', target_is_directory=True)
            fd = os.open(root / 'guest', os.O_RDONLY | os.O_DIRECTORY)
            try:
                for relative in ['private', '../outside']:
                    with self.subTest(relative=relative), self.assertRaises((OSError, ValueError)):
                        directory(fd, relative)
                with self.assertRaises(OSError):
                    bounded_file(fd, 'private/secret', 100)
                self.assertEqual((root / 'outside/secret').read_bytes(), b'outside')
                self.assertEqual(len(list((root / 'outside').iterdir())), 1)
            finally:
                os.close(fd)

    def test_existing_entry_is_never_overwritten_or_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'guest').mkdir()
            path = root / 'guest/control'
            path.write_bytes(b'original')
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with self.assertRaises(FileExistsError):
                    write_probe(fd, 'guest', 'control')
                self.assertEqual(path.read_bytes(), b'original')
                report = existing_write_open(fd, 'guest/control')
                self.assertTrue(report['write_open_without_write'])
                self.assertEqual(path.read_bytes(), b'original')
            finally:
                os.close(fd)

    def test_bounded_regular_read_rejects_symlink_and_oversize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'guest').mkdir()
            (root / 'guest/file').write_bytes(b'12345')
            (root / 'guest/link').symlink_to('file')
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                self.assertEqual(bounded_file(fd, 'guest/file', 5), b'12345')
                with self.assertRaises(ValueError):
                    bounded_file(fd, 'guest/file', 4)
                with self.assertRaises(OSError):
                    bounded_file(fd, 'guest/link', 5)
            finally:
                os.close(fd)
