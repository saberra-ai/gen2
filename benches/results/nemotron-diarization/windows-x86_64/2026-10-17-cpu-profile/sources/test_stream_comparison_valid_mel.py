import copy
import hashlib
import math
import struct
import unittest
from compare_stream_valid_mel import compare


class ValidMelReference(unittest.TestCase):
    def fixture(self):
        geometry = {'chunk': 9, 'right': 4, 'left': 0, 'fifo': 264, 'cache': 264, 'update': 222}
        raw = struct.pack('<f', .5) * 100 * 8
        gen2 = {'accepted_samples': 16000, 'complete': True, 'audio_duration_seconds': 1,
                'native_frame_count': 100, 'geometry': geometry, 'model_revision': 'f667ed73aee57d40cc39428eb768b4fd87a0a29e',
                'runtime_version': 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1', 'backend': 'metal', 'device': 'fixture GPU',
                'retained_probabilities': {'first_frame': 0, 'rows': 100}, 'elapsed_micros': 100,
                'retained_probability_sha256': hashlib.sha256(raw).hexdigest(),
                'turns': [{'start_seconds': .1, 'end_seconds': 1., 'speaker': 1}]}
        native = {'runtime_version': 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1', 'audio_samples': 16000, 'audio_duration_seconds': 1, 'native_frame_count': 100,
                  'geometry': geometry, 'requested_gpu_index': 0, 'retained_first_frame': 0,
                  'retained_rows': 100, 'elapsed_micros': 100,
                  'turns': [{'start_seconds': .1, 'end_seconds': 1., 'speaker': 1}]}
        return gen2, native, raw

    def test_valid_rows_and_turns_match(self):
        gen2, native, raw = self.fixture()
        self.assertTrue(compare(gen2, native, raw, raw)['passed'])
        for key, value in [('speaker', 2), ('start_seconds', .101), ('end_seconds', math.nan)]:
            changed = copy.deepcopy(native)
            changed['turns'][0][key] = value
            with self.assertRaises(ValueError):
                compare(gen2, changed, raw, raw)

    def test_excessive_padding_and_turns_outside_terminal_frame_cannot_disappear(self):
        gen2, native, raw = self.fixture()
        for end in [1.02, 1.9]:
            changed = copy.deepcopy(native)
            changed['turns'][0]['end_seconds'] = end
            with self.assertRaises(ValueError):
                compare(gen2, changed, raw, raw)
        native['turns'].append({'start_seconds': 1.1, 'end_seconds': 1.2, 'speaker': 2})
        with self.assertRaises(ValueError):
            compare(gen2, native, raw, raw)

    def test_invalid_or_missing_numerical_evidence_is_rejected(self):
        gen2, native, raw = self.fixture()
        for changed in [raw[:-1], struct.pack('<f', math.nan) + raw[4:], struct.pack('<f', 1.1) + raw[4:]]:
            with self.assertRaises(ValueError):
                compare(gen2, native, changed, raw)
        native['retained_rows'] = 99
        with self.assertRaises(ValueError):
            compare(gen2, native, raw, raw)

    def test_in_range_divergent_gen2_probabilities_cannot_qualify(self):
        gen2, native, raw = self.fixture()
        for changed in [raw[:-1], struct.pack('<f', .51) + raw[4:], struct.pack('<f', math.nan) + raw[4:]]:
            gen2['retained_probability_sha256'] = hashlib.sha256(changed).hexdigest()
            with self.assertRaises(ValueError):
                compare(gen2, native, raw, changed)
        gen2['retained_probability_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            compare(gen2, native, raw, raw)

    def test_incomplete_transport_and_mismatched_placement_cannot_qualify(self):
        gen2, native, raw = self.fixture()
        for key, value in [('complete', False), ('backend', 'cpu'), ('accepted_samples', 16001), ('device', None), ('native_frame_count', 101)]:
            changed = {**gen2, key: value}
            with self.assertRaises(ValueError):
                compare(changed, native, raw, raw)

    def test_the_last_valid_probability_is_never_discarded(self):
        gen2, native, raw = self.fixture()
        changed = raw[:-4] + struct.pack('<f', .51)
        gen2['retained_probability_sha256'] = hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError, 'probability values differ'):
            compare(gen2, native, raw, changed)

    def test_legacy_and_arbitrary_runtime_identifiers_are_rejected_on_both_sides(self):
        for version in ['nemo-speech-asr 0.2.0', 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1.extra']:
            for side in ['native', 'gen2']:
                with self.subTest(version=version, side=side):
                    gen2, native, raw = self.fixture()
                    (gen2 if side == 'gen2' else native)['runtime_version'] = version
                    with self.assertRaisesRegex(ValueError, 'runtime version'):
                        compare(gen2, native, raw, raw)

    def test_partial_hops_keep_the_same_floor_valid_count(self):
        gen2, native, raw = self.fixture()
        for value in [gen2, native]:
            value['audio_duration_seconds'] = 16017 / 16000
        gen2['accepted_samples'] = native['audio_samples'] = 16017
        self.assertEqual(compare(gen2, native, raw, raw)['native_frame_count'], 100)
