import unittest
import numpy as np
from segmentation_ablation import CADENCE, DEFAULT, compare_native_turns, native_segments


class SegmentationAblationTests(unittest.TestCase):
    def test_only_equal_start_order_can_be_canonicalized(self):
        actual = [{'start':1.,'end':2.,'speaker':1},{'start':1.,'end':3.,'speaker':2},
                  {'start':4.,'end':5.,'speaker':1}]
        native = [{'start_seconds':s['start'],'end_seconds':s['end'],'speaker':s['speaker']} for s in actual]
        self.assertTrue(compare_native_turns(actual,[native[1],native[0],native[2]])['all_boundaries_and_speakers_exact'])
        with self.assertRaises(ValueError):
            compare_native_turns(actual,list(reversed(native)))

    def test_strict_thresholds_and_active_terminal_interval(self):
        p = np.zeros((5,8),dtype='<f4')
        p[:,0] = [.5,.6,.5,.4,.7]
        config = {**DEFAULT,'onset':.5,'offset':.5,'pad_onset':0.,'pad_offset':0.,
                  'min_duration_on':0.,'min_duration_off':0.}
        self.assertEqual(native_segments(p,config),[
            {'start':CADENCE,'end':3*CADENCE,'speaker':1},
            {'start':4*CADENCE,'end':5*CADENCE,'speaker':1}])

    def test_padding_precedes_merging_and_filtering(self):
        p = np.zeros((12,8),dtype='<f4')
        p[[2,6],0] = .9
        config = {**DEFAULT,'pad_onset':.01,'pad_offset':.01,
                  'min_duration_on':.06,'min_duration_off':.025}
        actual = native_segments(p,config)
        self.assertEqual(actual,[{'start':2*CADENCE-.01,'end':7*CADENCE+.01,'speaker':1}])
        self.assertEqual(native_segments(p,{**config,'min_duration_off':0.}),[])

    def test_overlap_and_invalid_probability_inputs(self):
        p = np.ones((80,8),dtype='<f4')
        result = native_segments(p,DEFAULT)
        self.assertEqual(len(result),8)
        self.assertTrue(all(s['start'] == 0 and s['end'] == 80*CADENCE for s in result))
        for bad in [p.astype('f8'),p[:,:7],p[:0],p*float('nan'),p*2]:
            with self.assertRaises(ValueError):
                native_segments(bad,DEFAULT)
