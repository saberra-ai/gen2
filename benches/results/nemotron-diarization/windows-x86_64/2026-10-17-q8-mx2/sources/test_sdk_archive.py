import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from sdk_archive import inventory, restore_archive, write_archive


class SDKArchiveTests(unittest.TestCase):
    def test_delivered_archive_restores_executable_aliases_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdk = root / 'source sdk'
            (sdk / 'lib').mkdir(parents=True)
            (sdk / 'bin').mkdir()
            (sdk / 'lib/real.so.1').write_bytes(b'native-library')
            (sdk / 'lib').chmod(0o775)
            (sdk / 'lib/real.so.1').chmod(0o664)
            (sdk / 'lib/entry.so').symlink_to('real.so.1')
            (sdk / 'bin/probe').write_bytes(b'#!/bin/sh\nexit 0\n')
            (sdk / 'bin/probe').chmod(0o755)
            snapshots = []
            for name in ['first', 'second']:
                output = root / name
                output.mkdir()
                summary = write_archive(sdk, output)
                archive = output / summary['archive']
                snapshots.append(archive.read_bytes())
                self.assertEqual(hashlib.sha256(snapshots[-1]).hexdigest(), summary['archive_sha256'])
                delivered = root / (name + ' relocated ü')
                manifest = json.loads((output / 'development-sdk-files.json').read_text())
                with tarfile.open(archive) as tar:
                    for relative, record in manifest['files'].items():
                        name = 'sdk' if relative == '.' else 'sdk/' + relative
                        self.assertEqual(tar.getmember(name).mode, record['mode'])
                restore_archive(archive, manifest, delivered)
                self.assertEqual(inventory(delivered), inventory(sdk))
                self.assertTrue((delivered / 'lib/entry.so').is_symlink())
                self.assertEqual((delivered / 'lib/entry.so').readlink().as_posix(), 'real.so.1')
                self.assertEqual((delivered / 'lib/real.so.1').read_bytes(), b'native-library')
                self.assertEqual((delivered / 'bin/probe').stat().st_mode & 0o777, 0o755)
                self.assertEqual(manifest['files'], inventory(sdk))
                self.assertFalse(manifest['release_qualified'])
            self.assertEqual(*snapshots)

    def test_corrupt_delivery_does_not_publish_or_replace_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdk = root / 'sdk'
            sdk.mkdir()
            (sdk / 'file').write_bytes(b'input')
            output = root / 'output'
            output.mkdir()
            summary = write_archive(sdk, output)
            archive = output / summary['archive']
            manifest = json.loads((output / 'development-sdk-files.json').read_text())
            archive.write_bytes(archive.read_bytes() + b'changed')
            destination = root / 'delivered'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                restore_archive(archive, manifest, destination)
            self.assertFalse(destination.exists())
            destination.mkdir()
            (destination / 'user-file').write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):
                restore_archive(archive, manifest, destination)
            self.assertEqual((destination / 'user-file').read_bytes(), b'preserve')

    def test_escaping_or_dangling_alias_is_rejected_before_archive_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdk = root / 'sdk'
            sdk.mkdir()
            output = root / 'output'
            output.mkdir()
            (root / 'outside').write_text('outside')
            alias = sdk / 'alias'
            for target in ['../outside', str(root / 'outside'), 'missing']:
                alias.symlink_to(target)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    write_archive(sdk, output)
                self.assertEqual(list(output.iterdir()), [])
                alias.unlink()


if __name__ == '__main__':
    unittest.main()
