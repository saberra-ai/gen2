"""The fixed60s gate requires the complete history, even when tails agree."""
import hashlib
import struct
import unittest

import test_stream_comparison_valid_mel as fixtures
from qualify_valid_mel_macos_package import compare_complete_60s


class CompleteRecordingTests(unittest.TestCase):
    def fixture(self):
        public, native, _ = fixtures.ValidMelReference().fixture()
        raw = struct.pack('<f', .5) * 6000 * 8
        public.update(accepted_samples=960000, audio_duration_seconds=60, native_frame_count=6000,
                      backend='cpu', device=None, retained_probabilities={'first_frame': 0, 'rows': 6000},
                      retained_probability_sha256=hashlib.sha256(raw).hexdigest())
        native.update(audio_samples=960000, audio_duration_seconds=60, native_frame_count=6000,
                      requested_gpu_index=-1, retained_first_frame=0, retained_rows=6000)
        return public, native, raw

    def test_all6000rows_are_compared(self):
        public, native, raw = self.fixture()
        self.assertTrue(compare_complete_60s(public, native, raw, raw)['passed'])
        changed = raw[:-4] + struct.pack('<f', .51)
        public['retained_probability_sha256'] = hashlib.sha256(changed).hexdigest()
        with self.assertRaises(ValueError):
            compare_complete_60s(public, native, raw, changed)

    def test_matching_compacted_tails_do_not_prove_complete60s(self):
        public, native, raw = self.fixture()
        raw = raw[8 * 4:]
        public['retained_probabilities'] = {'first_frame': 1, 'rows': 5999}
        public['retained_probability_sha256'] = hashlib.sha256(raw).hexdigest()
        native.update(retained_first_frame=1, retained_rows=5999)
        with self.assertRaisesRegex(ValueError, 'complete6000row'):
            compare_complete_60s(public, native, raw, raw)

    def test_shorter_matching_recording_and_truncated_raw_files_fail(self):
        public, native, raw = fixtures.ValidMelReference().fixture()
        with self.assertRaises(ValueError):
            compare_complete_60s(public, native, raw, raw)
        public, native, raw = self.fixture()
        with self.assertRaisesRegex(ValueError, 'complete6000row'):
            compare_complete_60s(public, native, raw[:-32], raw[:-32])


if __name__ == '__main__':
    unittest.main()
