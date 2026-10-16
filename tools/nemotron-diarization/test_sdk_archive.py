import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from sdk_archive import inventory, write_archive


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
                delivered.mkdir()
                manifest = json.loads((output / 'development-sdk-files.json').read_text())
                with tarfile.open(archive) as tar:
                    for relative, record in manifest['files'].items():
                        name = 'sdk' if relative == '.' else 'sdk/' + relative
                        self.assertEqual(tar.getmember(name).mode, record['mode'])
                    tar.extractall(delivered, filter='data')
                # Python's data filter intentionally drops directory modes and
                # group-write bits; restore the validated inventory's modes.
                for relative, record in manifest['files'].items():
                    if record['type'] != 'symlink':
                        (delivered / 'sdk' / relative).chmod(record['mode'])
                self.assertEqual(inventory(delivered / 'sdk'), inventory(sdk))
                self.assertEqual(manifest['files'], inventory(sdk))
                self.assertFalse(manifest['release_qualified'])
            self.assertEqual(*snapshots)

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
