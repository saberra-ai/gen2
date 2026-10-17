import copy
import json
import struct
import unittest
from bounded_cpu_metrics import VERSION,REFERENCE_VERSION,derive_helper,compare_bounded_history,validate_failed_loads
from qualify_pooled_native import OLD_VERSION


class BoundedCpuMetricsTests(unittest.TestCase):
    def test_exact_identity_and_full_history_before_any_timing_claim(self):
        reference = {'runtime_version':REFERENCE_VERSION,'audio_samples':320,'audio_duration_seconds':.02,
                     'native_frame_count':2,'retained_first_frame':0,'retained_rows':2,'input_repeats':1,
                     'requested_gpu_index':-1,'geometry':{'chunk':9,'right':4},'input_pcm_f32le_sha256':'fixture',
                     'turns':[{'speaker':1,'start_seconds':0.,'end_seconds':.02}]}
        actual = {**reference,'runtime_version':VERSION,'elapsed_micros':1000,'cold_native_load_micros':100}
        raw = struct.pack('<16f',*([.5]*16))
        self.assertTrue(compare_bounded_history(actual,raw,reference,raw)['bit_exact'])
        for changed in [raw[:-4],raw[:-4]+struct.pack('<f',.51),raw[:-4]+struct.pack('<f',float('nan'))]:
            with self.assertRaises(ValueError): compare_bounded_history(actual,changed,reference,raw)
        for key,value in [('runtime_version',VERSION+'.extra'),('elapsed_micros',0),('native_frame_count',1),
                          ('requested_gpu_index',0),('input_pcm_f32le_sha256','other')]:
            with self.subTest(change=key),self.assertRaises(ValueError):
                compare_bounded_history({**actual,key:value},raw,reference,raw)
        with self.assertRaises(ValueError):
            compare_bounded_history(actual,raw,{**reference,'runtime_version':VERSION},raw)
        with self.assertRaises(ValueError):
            compare_bounded_history({**actual,'turns':[]},raw,reference,raw)

    def test_one_version_literal_only(self):
        original = 'strcmp(version,'+json.dumps(OLD_VERSION)+');'
        self.assertEqual(derive_helper(original),original.replace(OLD_VERSION,VERSION))
        for text in ['',original+original]:
            with self.assertRaises(ValueError): derive_helper(text)

    def test_failed_load_control_preserves_all_observations_and_rejects_growth(self):
        value = {'failed_model_construction_attempts':101,'valid_models_loaded':0,'frontend_or_neural_calls':0,
                 'thread_growth_allowance':2,'maximum_requested_settling_wait_milliseconds':100,
                 'immediate_thread_counts':[3]*101,'threads_after_each_failure':[1]*101,
                 'requested_settling_wait_milliseconds':[10]*101}
        self.assertEqual(validate_failed_loads(value),value)
        for key,items in [('threads_after_each_failure',[1]*100+[4]),('immediate_thread_counts',[3]*100),
                          ('requested_settling_wait_milliseconds',[10]*100+[110])]:
            bad = copy.deepcopy(value); bad[key] = items
            with self.assertRaises(ValueError): validate_failed_loads(bad)
