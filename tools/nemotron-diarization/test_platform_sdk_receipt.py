"""Package admission rejects stale, partial and altered platform receipts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from build_valid_mel_candidate import descriptor
from platform_sdk_receipt import copy_provenance, validate_native_receipt, verify_provenance
from runtime_profiles import load_profile, profile_provenance


class PlatformSdkReceiptTests(unittest.TestCase):
    def fixture(self):
        tools = Path(__file__).resolve().parent
        lock_path, lock = load_profile()
        protocol = json.loads((tools / 'valid-mel-platform-native.protocol.json').read_text())
        phases = ['sdk-build', 'sdk-install', 'model-free-state-test', 'version-only',
                  'native-inference', 'public-inference', 'native-lifecycle-list']
        phases += ['native-lifecycle-' + str(i) for i in range(7)]
        return {**profile_provenance(lock_path, lock), 'schema_version': 2, 'backend': 'cpu',
                'development_native_gates_passed': True,
                'actual_reported_version': lock['runtime']['reported_version'],
                'protocol_sha256': descriptor(tools / 'valid-mel-platform-native.protocol.json')['sha256'],
                'model_free_state_test_completed': True,
                'completed_lifecycle_tests': protocol['required_lifecycle_tests'],
                'source_commit': '1' * 40, 'commands': [{'name': n, 'returncode': 0} for n in phases],
                'compiler_audit': {'bytes': 2, 'sha256': '0' * 64}}

    def test_historical_or_partial_runtime_cannot_be_relabelled_as_current(self):
        fixture = self.fixture()
        validate_native_receipt(fixture)
        for delta in [{'schema_version': 1}, {'actual_reported_version': 'nemo-speech-asr 0.2.0'},
                      {'lock_sha256': '0' * 64}, {'protocol_sha256': '0' * 64},
                      {'model_free_state_test_completed': False}, {'completed_lifecycle_tests': []},
                      {'source_commit': ''}, {'development_native_gates_passed': False}]:
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                validate_native_receipt({**fixture, **delta})

    def test_missing_failed_or_duplicate_commands_cannot_establish_completion(self):
        fixture = self.fixture()
        variants = [fixture['commands'][:-1], fixture['commands'] + [fixture['commands'][0]],
                    [{**c, 'returncode': 1} if c['name'] == 'native-inference' else c for c in fixture['commands']],
                    [{k: v for k, v in c.items() if k != 'returncode'} for c in fixture['commands']]]
        for commands in variants:
            with self.assertRaisesRegex(ValueError, 'command/source-audit'):
                validate_native_receipt({**fixture, 'commands': commands})

    def test_package_retains_exact_receipt_patch_profile_protocol_and_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build = self.fixture()
            audit = root / 'platform-llama-materialized-audit.json'
            audit.write_text('{}')
            build['compiler_audit'] = descriptor(audit)
            receipt = root / 'receipt.json'; receipt.write_text(json.dumps(build))
            package = root / 'package'; package.mkdir()
            copy_provenance(receipt, package / 'provenance')
            manifest = {**copy.deepcopy(build), 'build_receipt_sha256': descriptor(receipt)['sha256']}
            verify_provenance(package, manifest)
            for name in ['native-receipt.json', 'compiler-audit.json', 'runtime-profile.json',
                         'native-protocol.json', 'inference.patch']:
                path = package / 'provenance' / name
                original = path.read_bytes(); path.write_bytes(original + b'changed')
                with self.subTest(name=name), self.assertRaises((ValueError, json.JSONDecodeError)):
                    verify_provenance(package, manifest)
                path.write_bytes(original)
            (package / 'provenance/unrecorded').write_text('extra')
            with self.assertRaisesRegex(ValueError, 'unexpected platform provenance'):
                verify_provenance(package, manifest)


if __name__ == '__main__':
    unittest.main()
