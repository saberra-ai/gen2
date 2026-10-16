import json
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tempfile
import unittest

from compare_macos_cpu_pools import compare_probabilities, model_free_policy
from qualify_macos_portable_runtime import CONTROL, policy


class PoolComparisonTests(unittest.TestCase):
    def test_all_probabilities_including_last_channel_are_compared(self):
        reference = struct.pack('<9600f', *([.5] * 9600))
        self.assertTrue(compare_probabilities(reference, reference)['bit_exact'])
        actual = reference[:-4] + struct.pack('<f', .51)
        with self.assertRaisesRegex(ValueError, 'parity failed'):
            compare_probabilities(actual, reference)

    def test_established_tolerance_allows_float_rounding(self):
        reference = struct.pack('<9600f', *([.5] * 9600))
        actual = reference[:-4] + struct.pack('<f', .5000005)
        result = compare_probabilities(actual, reference)
        self.assertEqual(result['values_compared'], 9600)
        self.assertFalse(result['bit_exact'])

    def test_truncation_and_nonfinite_or_unbounded_values_rejected(self):
        valid = struct.pack('<9600f', *([.5] * 9600))
        for raw in [valid[:-4], valid + b'\0', *[valid[:-4] + struct.pack('<f', v)
                    for v in [float('nan'), float('inf'), -1, 1.01]]]:
            for actual, reference in [(raw, valid), (valid, raw)]:
                with self.assertRaises(ValueError):
                    compare_probabilities(actual, reference)

    @unittest.skipUnless(platform.system() == 'Darwin', 'actual macOS sandbox')
    def test_actual_model_free_policy_denies_all_weight_formats_and_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probes = [root / ('denied.' + suffix)
                      for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
            for probe in probes:
                probe.write_bytes(b'not a model')
                self.assertEqual(probe.read_bytes(), b'not a model')
            sandbox = root / 'model-free.sb'
            sandbox.write_text(model_free_policy(policy(
                denied_reads=[root / 'source'], readonly=[root / 'protected'])))
            observed = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(sandbox), sys.executable,
                '-c', CONTROL, json.dumps({'reads': list(map(str, probes)), 'writes': []})],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(observed.returncode, 0, observed.stderr)


if __name__ == '__main__':
    unittest.main()
