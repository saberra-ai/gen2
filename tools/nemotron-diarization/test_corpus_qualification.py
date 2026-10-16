import copy
import hashlib
import unittest

import numpy as np

from qualify_corpus import GEOMETRY, MODEL_REVISION, score_slice


class QualificationTests(unittest.TestCase):
    def fixture(self):
        samples = 1600
        fp = np.zeros((10, 8), dtype=np.float32)
        fp[:, 0] = 1
        raw = np.vstack([fp, fp[-1:]]).astype('<f4').tobytes()
        item = {'id': 'fixture', 'audio_samples': samples, 'uem_seconds': [0, .1], 'statistics': {'speakers': 1}}
        gen2 = {'accepted_samples': samples, 'complete': True, 'audio_duration_seconds': .1,
                'native_frame_count': 11, 'geometry': GEOMETRY, 'model_revision': MODEL_REVISION,
                'runtime_version': 'nemo-speech-asr 0.2.0', 'backend': 'cpu', 'device': None,
                'retained_probabilities': {'first_frame': 0, 'rows': 11},
                'retained_probability_sha256': hashlib.sha256(raw).hexdigest(), 'elapsed_micros': 1,
                'turns': [{'start_seconds': 0, 'end_seconds': .1, 'speaker': 1}]}
        native = {'audio_samples': samples, 'audio_duration_seconds': .1, 'native_frame_count': 11,
                  'geometry': GEOMETRY, 'requested_gpu_index': -1, 'retained_first_frame': 0, 'retained_rows': 11,
                  'elapsed_micros': 1, 'turns': [{'start_seconds': 0, 'end_seconds': .11, 'speaker': 1}]}
        meta = {'audio_samples': samples, 'geometry': GEOMETRY, 'model_revision': MODEL_REVISION,
                'device': 'cpu', 'dtype': 'float32', 'shape': [10, 8]}
        refs = [{'start': 0, 'end': .1, 'speaker': 'reference'}]
        return [item, gen2, native, raw, raw, fp, meta, refs]

    def test_exact_shared_threshold_and_deployed_gate(self):
        result = score_slice(*self.fixture())
        self.assertTrue(all(result['gates'].values()))
        self.assertEqual(result['q8_shared_threshold']['der'], 0)
        self.assertEqual(result['shape_difference']['native_extra_terminal_frame'], 1)

    def test_poor_slice_is_reported_without_reselection(self):
        args = self.fixture()
        args[-1] = [{'start': 0, 'end': .02, 'speaker': 'reference'}]
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
                              (1, {'retained_probability_sha256': 'wrong'})]:
            args = self.fixture()
            args[index] = {**copy.deepcopy(args[index]), **change}
            with self.subTest(index=index, change=change), self.assertRaises(ValueError):
                score_slice(*args)


if __name__ == '__main__':
    unittest.main()
