import copy
import json
from pathlib import Path
import platform
import struct
import subprocess
import tempfile
import unittest

from qualify_macos_portable_runtime import policy
from qualify_pooled_native import OLD_VERSION, NEW_VERSION, compare_history, derive_helper, result_policy


class PooledNativeChecks(unittest.TestCase):
    def fixture(self):
        reference = {'audio_samples': 320, 'audio_duration_seconds': .02, 'requested_gpu_index': -1,
                     'native_frame_count': 2, 'retained_first_frame': 0, 'retained_rows': 2,
                     'input_repeats': 1, 'geometry': {'chunk': 9, 'right': 4},
                     'input_pcm_f32le_sha256': 'fixture', 'runtime_version': OLD_VERSION,
                     'turns': [{'speaker': 1, 'start_seconds': 0., 'end_seconds': .02}]}
        actual = copy.deepcopy(reference)
        actual['runtime_version'] = NEW_VERSION
        return actual, reference, struct.pack('<16f', *([.5] * 16))

    def test_complete_history_and_turns_match(self):
        actual, reference, raw = self.fixture()
        comparison = compare_history(actual, raw, reference, raw)
        self.assertTrue(comparison['bit_exact'])
        self.assertEqual(comparison['values_compared'], 16)
        self.assertEqual(comparison['turns'], 1)

    def test_last_probability_and_incomplete_history_are_rejected(self):
        actual, reference, raw = self.fixture()
        for changed in [raw[:-4], raw[:-4] + struct.pack('<f', .51), raw[:-4] + struct.pack('<f', float('nan'))]:
            with self.assertRaises(ValueError):
                compare_history(actual, changed, reference, raw)

    def test_wrong_input_identity_frame_history_and_turns_are_rejected(self):
        actual, reference, raw = self.fixture()
        for name, value in [('input_pcm_f32le_sha256', 'other'), ('native_frame_count', 1),
                            ('retained_first_frame', 1), ('requested_gpu_index', 0),
                            ('runtime_version', NEW_VERSION + '.extra')]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                compare_history({**actual, name: value}, raw, reference, raw)
        for changed in [[{'speaker': 2, 'start_seconds': 0., 'end_seconds': .02}],
                        [{'speaker': 1, 'start_seconds': 0., 'end_seconds': .01}], []]:
            with self.assertRaises(ValueError):
                compare_history({**actual, 'turns': changed}, raw, reference, raw)

    def test_derivation_changes_one_exact_literal_and_rejects_missing_or_duplicate_matches(self):
        original = 'if (strcmp(version, ' + json.dumps(OLD_VERSION) + ')) fail();\n'
        self.assertEqual(derive_helper(original), original.replace(OLD_VERSION, NEW_VERSION))
        for text in ['', original + original]:
            with self.assertRaises(ValueError):
                derive_helper(text)

    @unittest.skipUnless(platform.system() == 'Darwin', 'actual macOS metadata/data sandbox boundary')
    def test_native_can_stat_and_publish_outputs_but_cannot_read_prior_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / 'results'
            results.mkdir()
            source = results / 'prior.json'
            source.write_bytes(b'not a model')
            sandbox = root / 'runtime.sb'
            sandbox.write_text(result_policy(policy(denied_reads=[root / 'source'], readonly=[root / 'assets']), results))
            code = '''from pathlib import Path
import sys
source=Path(sys.argv[1])
assert source.exists()
try:
 source.read_bytes()
except PermissionError:
 pass
else:
 raise RuntimeError('prior payload unexpectedly readable')
(source.parent/'published.json').write_text('{}')
'''
            observed = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(sandbox), '/usr/bin/python3',
                '-c', code, str(source)], capture_output=True, text=True, timeout=10)
            self.assertEqual(observed.returncode, 0, observed.stderr)
            self.assertEqual((results / 'published.json').read_text(), '{}')
