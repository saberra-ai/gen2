import copy
import struct
import unittest
from profile_macos_pooled_cpu import compare_sampled_history


class PooledSamplingParityTests(unittest.TestCase):
    def setUp(self):
        self.record = {'runtime_version':'nemo-speech-asr 0.2.0+gen2-valid-mel-pool-v1',
            'audio_samples':192000,'native_frame_count':1200,'retained_first_frame':0,'retained_rows':1200,
            'input_repeats':1,'requested_gpu_index':-1,'audio_duration_seconds':12.,
            'geometry':{'chunk':9,'right':4,'left':0,'fifo':264,'cache':264,'update':222},
            'input_pcm_f32le_sha256':'a'*64,
            'turns':[{'start_seconds':0.,'end_seconds':.1,'speaker':1}]}
        self.raw = struct.pack('<f',.5)*9600

    def test_complete_pool_to_pool_history_keeps_input_identity(self):
        expected = copy.deepcopy(self.record)
        result = compare_sampled_history(self.record,self.raw,expected,self.raw)
        self.assertTrue(result['bit_exact'])
        self.assertEqual(result['values_compared'],9600)
        self.assertEqual(expected,self.record)

    def test_old_unknown_and_suffix_versions_are_not_admitted(self):
        for version in ['nemo-speech-asr 0.2.0+gen2-valid-mel-v1',
                        self.record['runtime_version']+'-future','unknown']:
            changed = {**self.record,'runtime_version':version}
            for actual,expected in [(changed,self.record),(self.record,changed)]:
                with self.assertRaisesRegex(ValueError,'exact pooled'):
                    compare_sampled_history(actual,self.raw,expected,self.raw)

    def test_last_value_and_complete_length_remain_checked(self):
        for changed in [self.raw[:-4],self.raw[:-4]+struct.pack('<f',.7),self.raw[:-4]+struct.pack('<f',float('nan'))]:
            with self.assertRaises(ValueError):
                compare_sampled_history(self.record,changed,self.record,self.raw)
