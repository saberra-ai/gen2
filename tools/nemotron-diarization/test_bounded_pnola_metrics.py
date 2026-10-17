"""Retain complete-app numerical/session checks behind exact bounded identity."""
import copy
import unittest

from bounded_pnola_metrics import VERSION, live_matches, native_reference, recorded_matches
from pooled_pnola_metrics import GEOMETRY
import test_pooled_pnola_metrics as pooled_tests
import test_valid_mel_pnola_complete_app as complete_tests
import test_valid_mel_pnola as timeline_tests


class BoundedPnolaMetricsTests(unittest.TestCase):
    def native(self, duration):
        value = pooled_tests.PooledAppMetricsTests().native(duration)
        value['runtime_version'] = VERSION
        return value

    def test_recorded_preserves_words_text_turns_and_input_without_mutation(self):
        timeline = timeline_tests.PnolaTimelineTest().reference()
        timeline['runtime_version'] = VERSION
        actual = {'audio_sha256': 'wav', 'text': 'exact Bark', 'asr_error': None,
                  'attribution': {'status': 'complete', 'complete': True, 'words_complete': True,
                                  'words': [{}] * 117, 'timeline': timeline}}
        native = self.native(60.)
        saved = copy.deepcopy((actual, native))
        self.assertEqual(recorded_matches(actual, native, 'exact Bark', 'wav')['frames'], 6000)
        self.assertEqual((actual, native), saved)
        for mutation in ['words', 'text', 'turn', 'sha', 'profile', 'device']:
            a = copy.deepcopy(actual)
            if mutation == 'words':
                a['attribution']['words'].pop()
            elif mutation == 'text':
                a['text'] = 'changed'
            elif mutation == 'turn':
                a['attribution']['timeline']['turns'][0]['end_seconds'] += .001
            elif mutation == 'sha':
                a['audio_sha256'] = 'changed'
            elif mutation == 'profile':
                a['attribution']['timeline']['runtime_version'] += '.extra'
            else:
                a['attribution']['timeline']['device'] = 'unexpected GPU'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                recorded_matches(a, native, 'exact Bark', 'wav')

    def test_live_requires_complete_session_and_independent_final_turns(self):
        actual = complete_tests.CompleteAppTests().live()
        actual['speakers']['timeline']['runtime_version'] = VERSION
        native = self.native(80.04)
        saved = copy.deepcopy((actual, native))
        self.assertEqual(live_matches(actual, native)['native_reference_frame_count'], 8004)
        self.assertEqual((actual, native), saved)
        for mutation in ['turn', 'coverage', 'revision', 'export', 'duration', 'profile', 'backend']:
            a, n = copy.deepcopy(actual), copy.deepcopy(native)
            if mutation == 'turn':
                n['turns'][0]['end_seconds'] += .001
            elif mutation == 'coverage':
                n['retained_first_frame'] = 1
            elif mutation == 'revision':
                a['provisional_revisions'] *= 2
            elif mutation == 'export':
                a['export_bytes'] = 0
            elif mutation == 'duration':
                a['speakers']['processed_audio_seconds'] = 0
            elif mutation == 'profile':
                n['runtime_version'] += '.extra'
            else:
                a['speakers']['timeline']['backend'] = 'metal'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                live_matches(a, n)

    def test_native_rejects_identity_placement_geometry_and_coverage_drift(self):
        native = self.native(60.)
        native_reference(native, 60.)
        for mutation in ['profile', 'gpu', 'geometry', 'coverage']:
            n = copy.deepcopy(native)
            if mutation == 'profile':
                n['runtime_version'] += '.extra'
            elif mutation == 'gpu':
                n['requested_gpu_index'] = 0
            elif mutation == 'geometry':
                n['geometry'] = {**GEOMETRY, 'fifo': 265}
            else:
                n['retained_rows'] -= 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                native_reference(n, 60.)
