import copy
import hashlib
from pathlib import Path
import tomllib
import unittest

import numpy as np

from qualify_corpus import GEOMETRY, MODEL_REVISION, score_slice, validate_oracle_environment


class QualificationTests(unittest.TestCase):
    def fixture(self):
        samples = 1600
        fp = np.zeros((10, 8), dtype=np.float32)
        fp[:, 0] = 1
        raw = np.vstack([fp, fp[-1:]]).astype('<f4').tobytes()
        item = {'id': 'fixture', 'audio_samples': samples, 'uem_seconds': [0, .1], 'statistics': {'speakers': 1}}
        gen2 = {'accepted_samples': samples, 'complete': True, 'audio_duration_seconds': .1,
                'input_pcm_f32le_sha256': 'expected-pcm-hash', 'input_repeats': 1,
                'native_frame_count': 11, 'geometry': GEOMETRY, 'model_revision': MODEL_REVISION,
                'runtime_version': 'nemo-speech-asr 0.2.0', 'backend': 'cpu', 'device': None,
                'retained_probabilities': {'first_frame': 0, 'rows': 11},
                'retained_probability_sha256': hashlib.sha256(raw).hexdigest(), 'elapsed_micros': 1,
                'turns': [{'start_seconds': 0, 'end_seconds': .1, 'speaker': 1}]}
        native = {'audio_samples': samples, 'audio_duration_seconds': .1, 'native_frame_count': 11,
                  'input_pcm_f32le_sha256': 'expected-pcm-hash', 'input_repeats': 1,
                  'geometry': GEOMETRY, 'requested_gpu_index': -1, 'retained_first_frame': 0, 'retained_rows': 11,
                  'elapsed_micros': 1, 'turns': [{'start_seconds': 0, 'end_seconds': .11, 'speaker': 1}]}
        meta = {'audio_samples': samples, 'geometry': GEOMETRY, 'model_revision': MODEL_REVISION,
                'input_pcm_f32le_sha256': 'expected-pcm-hash', 'input_repeats': 1,
                'device': 'cpu', 'dtype': 'float32', 'shape': [10, 8]}
        refs = [{'start': 0, 'end': .1, 'speaker': 'reference'}]
        return [item, gen2, native, raw, raw, fp, meta, refs, 'expected-pcm-hash']

    def test_exact_shared_threshold_and_deployed_gate(self):
        result = score_slice(*self.fixture())
        self.assertTrue(all(result['gates'].values()))
        self.assertEqual(result['q8_shared_threshold']['der'], 0)
        self.assertEqual(result['shape_difference']['native_extra_terminal_frame'], 1)

    def test_poor_slice_is_reported_without_reselection(self):
        args = self.fixture()
        args[-2] = [{'start': 0, 'end': .02, 'speaker': 'reference'}]
        result = score_slice(*args)
        self.assertFalse(result['gates']['q8_production_der_at_most_0_20'])
        self.assertFalse(result['gates']['q8_shared_der_at_most_0_20'])
        self.assertTrue(result['gates']['q8_degradation_at_most_0_02'])

    def test_frame_count_dtype_and_nonfinite_reference_rejected(self):
        for change in ['missing', 'float64', 'nan', 'range']:
            with self.subTest(change=change):
                args = self.fixture()
                if change == 'missing':
                    args[5] = args[5][:-1]
                elif change == 'float64':
                    args[5] = args[5].astype(np.float64)
                elif change == 'nan':
                    args[5][0, 0] = np.nan
                else:
                    args[5][0, 0] = 1.1
                with self.assertRaises(ValueError):
                    score_slice(*args)

    def test_geometry_input_identity_and_wrapper_mismatch_rejected(self):
        for index, change in [(0, {'audio_samples': 1601}), (6, {'model_revision': 'other'}),
                              (6, {'geometry': {}}), (6, {'device': 'metal'}),
                              (1, {'retained_probability_sha256': 'wrong'}),
                              (1, {'input_pcm_f32le_sha256': 'other120s-channel'}),
                              (2, {'input_pcm_f32le_sha256': 'other120s-channel'}), (2, {'input_repeats': 2})]:
            args = self.fixture()
            args[index] = {**copy.deepcopy(args[index]), **change}
            with self.subTest(index=index, change=change), self.assertRaises(ValueError):
                score_slice(*args)

    def test_actual_oracle_versions_and_source_must_match_lock(self):
        frozen = tomllib.loads(Path(__file__).with_name('reference').joinpath('uv.lock').read_text())
        packages = {p['name']: p['version'] for p in frozen['package']}
        metadata = {'transformers_direct_url': {'vcs_info': {'commit_id': '683b721971417b3b2c64fba9001f525607547e5a'}},
                    'installed_packages': {n: packages[n] for n in ['torch', 'transformers', 'numpy', 'soundfile', 'librosa', 'scipy']}}
        validate_oracle_environment(metadata)
        for change in ['version', 'commit', 'missing', 'extra']:
            bad = copy.deepcopy(metadata)
            if change == 'version':
                bad['installed_packages']['torch'] = '0.0.0'
            elif change == 'commit':
                bad['transformers_direct_url']['vcs_info']['commit_id'] = 'other'
            elif change == 'missing':
                del bad['installed_packages']['numpy']
            else:
                bad['installed_packages']['unlocked'] = '1.0'
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_oracle_environment(bad)


if __name__ == '__main__':
    unittest.main()
