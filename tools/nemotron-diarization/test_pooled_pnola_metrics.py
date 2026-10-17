"""Reject profile drift, missing native coverage and changed app output."""
import copy
import unittest

from pooled_pnola_metrics import GEOMETRY, NEW_VERSION, live_matches, recorded_matches
import test_valid_mel_pnola_complete_app as complete_app_tests
import test_valid_mel_pnola as timeline_tests


class PooledAppMetricsTests(unittest.TestCase):
    def native(self, duration):
        value = timeline_tests.PnolaTimelineTest().reference()
        value.update(runtime_version=NEW_VERSION, requested_gpu_index=-1, geometry=GEOMETRY,
                     input_repeats=1, audio_samples=round(duration * 16000), audio_duration_seconds=duration,
                     native_frame_count=round(duration * 100), retained_first_frame=0, retained_rows=round(duration * 100))
        return value

    def test_live_requires_independent_native_turns_and_complete_session(self):
        actual = complete_app_tests.CompleteAppTests().live()
        actual['speakers']['timeline']['runtime_version'] = NEW_VERSION
        native = self.native(80.04)
        self.assertEqual(live_matches(actual, native)['native_reference_frame_count'], 8004)
        for mutation in ['native_turn', 'profile', 'coverage', 'export', 'duplicate_revision', 'device']:
            a, n = copy.deepcopy(actual), copy.deepcopy(native)
            if mutation == 'native_turn':
                n['turns'][0]['end_seconds'] += .001
            elif mutation == 'profile':
                n['runtime_version'] += '-future'
            elif mutation == 'coverage':
                n['retained_first_frame'] = 1
            elif mutation == 'export':
                a['export_bytes'] = 0
            elif mutation == 'duplicate_revision':
                a['provisional_revisions'] *= 2
            else:
                a['speakers']['timeline']['device'] = 'unexpected GPU'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                live_matches(a, n)

    def test_recorded_preserves_bark_word_and_native_frame_gates(self):
        timeline = timeline_tests.PnolaTimelineTest().reference()
        timeline['runtime_version'] = NEW_VERSION
        actual = {'audio_sha256': 'wav', 'text': 'exact Bark', 'asr_error': None,
                  'attribution': {'status': 'complete', 'complete': True, 'words_complete': True,
                                  'words': [{}] * 117, 'timeline': timeline}}
        native = self.native(60.)
        saved = copy.deepcopy(actual)
        self.assertEqual(recorded_matches(actual, native, 'exact Bark', 'wav')['frames'], 6000)
        self.assertEqual(actual, saved)
        for mutation in ['words', 'profile', 'turn', 'text']:
            a = copy.deepcopy(actual)
            if mutation == 'words':
                a['attribution']['words'].pop()
            elif mutation == 'profile':
                a['attribution']['timeline']['runtime_version'] += '-future'
            elif mutation == 'turn':
                a['attribution']['timeline']['turns'][0]['end_seconds'] += .001
            else:
                a['text'] = 'changed'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                recorded_matches(a, native, 'exact Bark', 'wav')
