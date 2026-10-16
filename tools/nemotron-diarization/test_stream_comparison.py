import copy
import math
import struct
import unittest
from compare_stream import compare


class LongSessionReference(unittest.TestCase):
    def fixture(self):
        geometry = {'chunk': 9, 'right': 4, 'left': 0, 'fifo': 264, 'cache': 264, 'update': 222}
        gen2 = {'accepted_samples': 16000, 'complete': True, 'audio_duration_seconds': 1,
                'native_frame_count': 101, 'geometry': geometry, 'model_revision': 'f667ed73aee57d40cc39428eb768b4fd87a0a29e',
                'runtime_version': 'nemo-speech-asr 0.2.0', 'backend': 'metal', 'device': 'fixture GPU',
                'retained_probabilities': {'first_frame': 0, 'rows': 101}, 'elapsed_micros': 100,
                'turns': [{'start_seconds': .1, 'end_seconds': 1., 'speaker': 1}]}
        native = {'audio_samples': 16000, 'audio_duration_seconds': 1, 'native_frame_count': 101,
                  'geometry': geometry, 'requested_gpu_index': 0, 'retained_first_frame': 0,
                  'retained_rows': 101, 'elapsed_micros': 100,
                  'turns': [{'start_seconds': .1, 'end_seconds': 1.01, 'speaker': 1}]}
        return gen2, native, struct.pack('<f', .5) * 101 * 8

    def test_only_native_terminal_padding_is_clipped(self):
        gen2, native, raw = self.fixture()
        self.assertTrue(compare(gen2, native, raw)['passed'])
        for key, value in [('speaker', 2), ('start_seconds', .101), ('end_seconds', math.nan)]:
            changed = copy.deepcopy(native)
            changed['turns'][0][key] = value
            with self.assertRaises(ValueError):
                compare(gen2, changed, raw)

    def test_invalid_or_missing_numerical_evidence_is_rejected(self):
        gen2, native, raw = self.fixture()
        for changed in [raw[:-1], struct.pack('<f', math.nan) + raw[4:], struct.pack('<f', 1.1) + raw[4:]]:
            with self.assertRaises(ValueError):
                compare(gen2, native, changed)
        native['retained_rows'] = 100
        with self.assertRaises(ValueError):
            compare(gen2, native, raw)

    def test_incomplete_transport_and_mismatched_placement_cannot_qualify(self):
        gen2, native, raw = self.fixture()
        for key, value in [('complete', False), ('backend', 'cpu'), ('accepted_samples', 16001), ('device', None), ('native_frame_count', 100)]:
            changed = {**gen2, key: value}
            with self.assertRaises(ValueError):
                compare(changed, native, raw)
