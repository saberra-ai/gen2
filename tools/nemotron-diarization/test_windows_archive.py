from pathlib import Path
import tempfile
import unittest
import zipfile
import hashlib
import stat
import struct
from unittest.mock import patch

from windows_archive import byte_inventory, restore_archive, safe_relative, write_archive


class WindowsArchiveTests(unittest.TestCase):
    def test_rejects_windows_alias_paths_even_on_posix_receiver(self):
        for name in ['../escape', '/absolute', 'C:/drive', 'x:stream', 'folder\\file',
                     'folder/../file', 'file.', 'file ', 'CON.txt', 'nul', 'COM1', 'LPT9.log',
                     'CON .txt', 'AUX .log', 'CONIN$', 'CONOUT$']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_relative(name)

    def test_relocation_preserves_bytes_and_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'source'
            (original / 'lib').mkdir(parents=True)
            (original / 'lib/private.dll').write_bytes(b'unchanged PE fixture')
            manifest = write_archive(original, root)
            restored = restore_archive(root / manifest['archive'], manifest, root / 'relocated runtime ü')
            self.assertEqual(byte_inventory(restored), byte_inventory(original))
            with self.assertRaises(FileExistsError):
                restore_archive(root / manifest['archive'], manifest, restored)

    def test_rejects_case_collisions_symlinks_and_missing_archive_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            (source / 'file').write_bytes(b'data')
            manifest = write_archive(source, root)
            archive = root / manifest['archive']
            # A correctly hashed outer archive still cannot omit a manifest member.
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('runtime/', '')
            manifest['archive_bytes'] = archive.stat().st_size
            manifest['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                restore_archive(archive, manifest, root / 'missing')
            self.assertFalse((root / 'missing').exists())
            (source / 'file').unlink()
            try:
                (source / 'alias').symlink_to(root)
            except OSError:
                self.skipTest('symlink creation unavailable')
            with self.assertRaises(ValueError):
                byte_inventory(source)
            (source / 'alias').unlink()
            (source / 'Name').write_bytes(b'a')
            (source / 'name').write_bytes(b'b')
            if len(list(source.iterdir())) == 2:
                with self.assertRaises(ValueError):
                    byte_inventory(source)

    def test_correct_outer_hash_does_not_accept_link_members_extra_files_or_changed_payload(self):
        for mutation in ['symlink', 'extra', 'changed']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / 'source'
                source.mkdir()
                (source / 'file').write_bytes(b'data')
                manifest = write_archive(source, root)
                archive = root / manifest['archive']
                with zipfile.ZipFile(archive) as stream:
                    entries = [(info, stream.read(info)) for info in stream.infolist()]
                with zipfile.ZipFile(archive, 'w') as stream:
                    for info, data in entries:
                        if info.filename == 'runtime/file':
                            if mutation == 'symlink':
                                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                            elif mutation == 'changed':
                                data = b'evil'
                        stream.writestr(info, data)
                    if mutation == 'extra':
                        stream.writestr('runtime/extra', b'not declared')
                manifest['archive_bytes'] = archive.stat().st_size
                manifest['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    restore_archive(archive, manifest, root / 'restored')
                self.assertFalse((root / 'restored').exists())

    def test_central_directory_is_bounded_before_zipfile_allocates_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            (source / 'file').write_bytes(b'data')
            manifest = write_archive(source, root)
            archive = root / manifest['archive']
            with zipfile.ZipFile(archive, 'a') as stream:
                for index in range(3000):
                    stream.writestr(f'extra/{index}', b'')
            data = bytearray(archive.read_bytes())
            # Forge a small advertised count; the actual central-directory
            # extent must also be bounded before Python builds ZipInfo objects.
            struct.pack_into('<HH', data, len(data) - 22 + 8, 2, 2)
            archive.write_bytes(data)
            manifest['archive_bytes'] = len(data)
            manifest['archive_sha256'] = hashlib.sha256(data).hexdigest()
            with patch('windows_archive.zipfile.ZipFile', side_effect=AssertionError('unbounded constructor reached')):
                with self.assertRaises(ValueError):
                    restore_archive(archive, manifest, root / 'restored')


if __name__ == '__main__':
    unittest.main()
