"""Profile migration must retain old scoring failures and exact new parity."""
import copy
import hashlib
import unittest

import numpy as np

from bounded_corpus_metrics import score_slice
from bounded_cpu_metrics import VERSION
from pooled_runtime import BOUNDED_PROFILE
import test_valid_mel_corpus as valid_mel


class BoundedCorpusTests(unittest.TestCase):
    def fixture(self):
        args = valid_mel.ValidMelCorpusTests().fixture()
        for value in args[1:3]:
            value['runtime_version'] = VERSION
        return args

    def test_actual_identity_and_inputs_preserved(self):
        args = self.fixture()
        before = copy.deepcopy(args)
        result = score_slice(*args)
        self.assertTrue(all(result['gates'].values()))
        self.assertEqual(result['wrapper_parity']['runtime_profile'], BOUNDED_PROFILE)
        self.assertEqual(result['runtime_version'], VERSION)
        self.assertTrue(result['wrapper_parity']['probabilities_bit_exact'])
        np.testing.assert_array_equal(args[5], before[5])
        self.assertEqual(args[:5], before[:5])
        self.assertEqual(args[6:], before[6:])

    def test_sub_tolerance_probability_mismatch_rejected(self):
        args = self.fixture()
        native = args[5].copy()
        native[0, 0] = .5
        public = native.copy()
        public[0, 0] = np.nextafter(np.float32(.5), np.float32(1))
        args[3], args[4] = public.astype('<f4').tobytes(), native.astype('<f4').tobytes()
        args[1]['retained_probability_sha256'] = hashlib.sha256(args[3]).hexdigest()
        with self.assertRaisesRegex(ValueError, 'byte-exact'):
            score_slice(*args)

    def test_quality_failure_cannot_become_profile_pass(self):
        args = self.fixture()
        args[-2] = [{'start': 0, 'end': .02, 'speaker': 'reference'}]
        result = score_slice(*args)
        self.assertTrue(result['gates']['wrapper_parity'])
        self.assertTrue(result['gates']['q8_degradation_at_most_0_02'])
        self.assertFalse(result['gates']['q8_shared_der_at_most_0_20'])
        self.assertFalse(result['gates']['q8_production_der_at_most_0_20'])

    def test_wrong_profile_placement_or_original_oracle_rejected(self):
        for change in ('native-version', 'public-version', 'gpu', 'backend', 'oracle'):
            args = self.fixture()
            if change.endswith('-version'):
                args[2 if change == 'native-version' else 1]['runtime_version'] = 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1'
            elif change == 'gpu':
                args[2]['requested_gpu_index'] = 0
            elif change == 'backend':
                args[1]['backend'] = 'metal'
            else:
                args[6]['model_revision'] = 'changed'
            with self.subTest(change=change), self.assertRaises(ValueError):
                score_slice(*args)


if __name__ == '__main__':
    unittest.main()
