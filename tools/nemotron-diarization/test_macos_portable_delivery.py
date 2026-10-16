"""A receiver admits only the exact artifact and complete frozen file set."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import os

from macos_portable_delivery import admit_delivery


class PortableDelivery(unittest.TestCase):
    def fixture(self, root):
        archive = root / 'artifact.zip'
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('nested/proof.txt', b'known construction proof\n')
        expected = {'id': 123, 'name': 'fixed-artifact', 'run_id': 456, 'source_commit': '1' * 40,
                    'zip': {'bytes': archive.stat().st_size, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()},
                    'files': {'nested/proof.txt': {'bytes': 25, 'sha256': '3423ab7ca250cfa4a1aef58b4e0472ec57b263c22464645f4cf87de884a83b87'}}}
        api = {'id': 123, 'name': 'fixed-artifact', 'size_in_bytes': expected['zip']['bytes'],
               'digest': 'sha256:' + expected['zip']['sha256'],
               'workflow_run': {'id': 456, 'head_sha': '1' * 40}}
        metadata = root / 'metadata.json'
        metadata.write_text(json.dumps(api))
        return archive, metadata, expected

    def test_exact_delivery_is_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, metadata, expected = self.fixture(root)
            destination = root / 'delivery'
            admit_delivery(archive, metadata, expected, destination)
            self.assertEqual((destination / 'nested/proof.txt').read_bytes(), b'known construction proof\n')

    def test_wrong_run_source_digest_or_frozen_member_is_not_published(self):
        for defect in ['run', 'source', 'digest', 'member', 'extra', 'archive']:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive, metadata, expected = self.fixture(root)
                api = json.loads(metadata.read_text())
                if defect == 'run':
                    api['workflow_run']['id'] = 457
                elif defect == 'source':
                    api['workflow_run']['head_sha'] = '2' * 40
                elif defect == 'digest':
                    api['digest'] = 'sha256:' + '0' * 64
                elif defect == 'member':
                    expected['files']['nested/proof.txt']['sha256'] = '0' * 64
                elif defect == 'extra':
                    expected['files']['missing.txt'] = expected['files']['nested/proof.txt']
                else:
                    with archive.open('ab') as stream:
                        stream.write(b'changed opaque artifact')
                metadata.write_text(json.dumps(api))
                destination = root / 'delivery'
                with self.assertRaises(ValueError):
                    admit_delivery(archive, metadata, expected, destination)
                self.assertFalse(destination.exists())

    def test_even_a_frozen_parent_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, metadata, expected = self.fixture(root)
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('../escape.txt', b'known construction proof\n')
            expected['files'] = {'../escape.txt': expected['files']['nested/proof.txt']}
            expected['zip'] = {'bytes': archive.stat().st_size, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
            api = json.loads(metadata.read_text())
            api.update(size_in_bytes=expected['zip']['bytes'], digest='sha256:' + expected['zip']['sha256'])
            metadata.write_text(json.dumps(api))
            with self.assertRaises(ValueError):
                admit_delivery(archive, metadata, expected, root / 'delivery')
            self.assertFalse((root / 'delivery').exists())
            self.assertFalse((root / 'escape.txt').exists())

    @unittest.skipIf(os.name == 'nt', 'Mac receiver destination symlink guard')
    def test_existing_dangling_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, metadata, expected = self.fixture(root)
            destination = root / 'delivery'
            destination.symlink_to(root / 'absent')
            with self.assertRaises(FileExistsError):
                admit_delivery(archive, metadata, expected, destination)
            self.assertTrue(destination.is_symlink())
            self.assertFalse((root / 'absent').exists())


if __name__ == '__main__':
    unittest.main()
