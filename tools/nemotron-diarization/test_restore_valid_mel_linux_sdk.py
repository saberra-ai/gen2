"""A CI handoff must retain executable modes, aliases and the qualified archive."""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from restore_valid_mel_linux_sdk import restore
from sdk_archive import inventory, write_archive
import test_platform_sdk_receipt as receipt_fixtures


class RestorePlatformSdkTests(unittest.TestCase):
    def fixture(self, root):
        sdk = root / 'sdk'; (sdk / 'bin').mkdir(parents=True); (sdk / 'lib').mkdir()
        executable = sdk / 'bin/probe'; executable.write_bytes(b'model-free archive fixture'); executable.chmod(0o755)
        library = sdk / 'lib/native.so.1'; library.write_bytes(b'model-free library fixture')
        (sdk / 'lib/native.so').symlink_to(library.name)
        evidence = root / 'evidence'; evidence.mkdir()
        archive = write_archive(sdk, evidence)
        build = receipt_fixtures.PlatformSdkReceiptTests().fixture()
        build.update(platform='Linux-model-free-fixture', machine='x86_64', development_sdk_archive=archive)
        (evidence / 'receipt.json').write_text(json.dumps(build))
        return sdk, evidence, build, archive

    def test_modes_aliases_and_all_delivered_bytes_are_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdk, evidence, _, _ = self.fixture(root)
            restored = restore(evidence, root / 'delivered SDK ü')
            self.assertEqual(inventory(restored), inventory(sdk))
            self.assertEqual((restored / 'bin/probe').stat().st_mode & 0o777, 0o755)
            self.assertTrue((restored / 'lib/native.so').is_symlink())
            with self.assertRaises(FileExistsError):
                restore(evidence, restored)

    def test_cli_refuses_dangling_output_symlink_without_creating_its_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, evidence, _, _ = self.fixture(root)
            target = root / 'absent-target'
            output = root / 'existing-link'; output.symlink_to(target, target_is_directory=True)
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('restore_valid_mel_linux_sdk.py')),
                                     '--native-evidence', str(evidence), '--output', str(output)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('refusing to replace an extracted snapshot', result.stderr)
            self.assertTrue(output.is_symlink())
            self.assertFalse(target.exists())

    def test_another_archive_manifest_cannot_replace_the_receipt_bound_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, evidence, _, archive = self.fixture(root)
            changed = copy.deepcopy(archive); changed['archive_sha256'] = '0' * 64
            (evidence / 'development-sdk-files.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'archive identity differs'):
                restore(evidence, root / 'rejected')
            self.assertFalse((root / 'rejected').exists())


if __name__ == '__main__':
    unittest.main()
