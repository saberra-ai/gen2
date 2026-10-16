"""Current corpus scoring keeps the final valid row and original quality gates."""
import hashlib
import unittest

import numpy as np

import test_corpus_qualification as legacy
from score_valid_mel_corpus import score_slice


class ValidMelCorpusTests(unittest.TestCase):
    def fixture(self):
        args = legacy.QualificationTests().fixture()
        raw = args[5].astype('<f4').tobytes()
        args[3] = args[4] = raw
        args[1].update(runtime_version='nemo-speech-asr 0.2.0+gen2-valid-mel-v1', native_frame_count=10,
                       retained_probabilities={'first_frame': 0, 'rows': 10},
                       retained_probability_sha256=hashlib.sha256(raw).hexdigest())
        args[2].update(runtime_version=args[1]['runtime_version'], native_frame_count=10, retained_rows=10)
        args[2]['turns'][0]['end_seconds'] = .1
        return args

    def test_equal_floor_valid_arrays_and_unchanged_quality_failures(self):
        args = self.fixture()
        result = score_slice(*args)
        self.assertTrue(all(result['gates'].values()))
        self.assertEqual(result['shape_difference']['native_extra_terminal_frame'], 0)
        self.assertEqual(result['channel_mae_common_frames'], 0)
        args[-2] = [{'start': 0, 'end': .02, 'speaker': 'reference'}]
        failed = score_slice(*args)
        self.assertFalse(failed['gates']['q8_production_der_at_most_0_20'])
        self.assertFalse(failed['gates']['q8_shared_der_at_most_0_20'])
        self.assertTrue(failed['gates']['q8_degradation_at_most_0_02'])

    def test_last_valid_row_contributes_to_quantization_error(self):
        args = self.fixture()
        changed = args[5].copy(); changed[-1, 0] = 0
        args[3] = args[4] = changed.astype('<f4').tobytes()
        args[1]['retained_probability_sha256'] = hashlib.sha256(args[3]).hexdigest()
        result = score_slice(*args)
        self.assertEqual(result['channel_max_error_common_frames'], 1)
        self.assertAlmostEqual(result['channel_mae_common_frames'], 1 / 80)

    def test_legacy_extra_row_and_untagged_native_cannot_be_scored(self):
        for change in ['extra', 'version']:
            args = self.fixture()
            if change == 'extra':
                args[3] += np.zeros((1, 8), dtype='<f4').tobytes()
                args[2]['native_frame_count'] = args[2]['retained_rows'] = 11
            else:
                args[2]['runtime_version'] = 'nemo-speech-asr 0.2.0'
            with self.subTest(change=change), self.assertRaises(ValueError):
                score_slice(*args)


if __name__ == '__main__':
    unittest.main()
