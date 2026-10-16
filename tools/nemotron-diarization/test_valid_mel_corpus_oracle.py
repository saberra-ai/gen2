"""Cached FP32 arrays require exact historical bytes and original oracle provenance."""
import hashlib
import json
from pathlib import Path
import tempfile
import tomllib
import unittest

import numpy as np

from qualify_valid_mel_corpus import admit_reference


class CachedOracleTests(unittest.TestCase):
    def fixture(self, root):
        oracle = root / 'oracle'; oracle.mkdir()
        array = oracle / 'fixture-fp32.npy'; np.save(array, np.zeros((2, 8), dtype=np.float32))
        versions = tomllib.loads(Path(__file__).with_name('reference').joinpath('uv.lock').read_text())
        packages = {p['name']: p['version'] for p in versions['package']}
        metadata = {'script_sha256': 'script', 'lock_sha256': 'model-lock', 'environment_lock_sha256': 'env-lock',
                    'transformers_direct_url': {'vcs_info': {'commit_id': '683b721971417b3b2c64fba9001f525607547e5a'}},
                    'installed_packages': {n: packages[n] for n in ['torch', 'transformers', 'numpy', 'soundfile', 'librosa', 'scipy']}}
        meta = oracle / 'fixture-fp32.json'; meta.write_text(json.dumps(metadata))
        protocol = {'tool_sha256': {'reference.py': 'script', 'runtime.lock.json': 'model-lock', 'reference/uv.lock': 'env-lock'}}
        receipt = {'raw_artifacts': {'fixture-fp32.npy': {'bytes': array.stat().st_size, 'sha256': hashlib.sha256(array.read_bytes()).hexdigest()}},
                   'artifacts': {'fixture-fp32.json': hashlib.sha256(meta.read_bytes()).hexdigest()}}
        return array, meta, metadata, protocol, receipt

    def test_exact_reuse_and_tampering_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); array, meta, metadata, protocol, receipt = self.fixture(root)
            result, _ = admit_reference({'id': 'fixture'}, root, protocol, receipt)
            self.assertEqual(result.shape, (2, 8))
            original = array.read_bytes(); array.write_bytes(original[:-1] + b'x')
            with self.assertRaisesRegex(ValueError, 'oracle differs'):
                admit_reference({'id': 'fixture'}, root, protocol, receipt)
            array.write_bytes(original); meta.write_text(meta.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'oracle differs'):
                admit_reference({'id': 'fixture'}, root, protocol, receipt)

    def test_matching_artifact_hash_cannot_replace_original_script_or_environment(self):
        for field in ['script', 'model-lock', 'environment']:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); _, meta, metadata, protocol, receipt = self.fixture(root)
                if field == 'script':
                    metadata['script_sha256'] = 'different-script'
                elif field == 'model-lock':
                    metadata['lock_sha256'] = 'different-model-lock'
                else:
                    metadata['installed_packages']['torch'] = '0.0.0'
                meta.write_text(json.dumps(metadata)); receipt['artifacts']['fixture-fp32.json'] = hashlib.sha256(meta.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    admit_reference({'id': 'fixture'}, root, protocol, receipt)


if __name__ == '__main__':
    unittest.main()
