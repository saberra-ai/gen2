"""Complete-app result admission without executing apps or models."""
import copy
import unittest

from qualify_valid_mel_pnola_app import live_matches, recorded_matches
import test_valid_mel_pnola as adapter_tests


class CompleteAppTests(unittest.TestCase):
    def live(self):
        timeline = adapter_tests.PnolaTimelineTest().reference()
        timeline.update(audio_duration_seconds=80.04, geometry=[9, 4, 0, 264, 264, 222], device=None)
        del timeline['native_frame_count']
        return {'passed': True, 'microphone_used': False, 'speaker_qualification': True,
                'sections': 11, 'queued': 0, 'words': 152, 'export_bytes': 803, 'duration_seconds': 80.04,
                'provisional_revisions': [{'revision': 1, 'elapsed_seconds': 5., 'processed_audio_seconds': 4.32}],
                'speakers': {'status': 'complete', 'processed_audio_seconds': 80.04, 'timeline': timeline}}

    def test_live_requires_complete_bounded_identity_and_turn_evidence(self):
        expected = self.live()
        self.assertFalse(live_matches(copy.deepcopy(expected), expected, 'cpu')['native_frame_count_observed'])
        for change in ['turn', 'device', 'nonfinite', 'missing', 'duplicate', 'unbounded']:
            actual = copy.deepcopy(expected)
            if change == 'turn':
                actual['speakers']['timeline']['turns'][0]['end_seconds'] += .001
            elif change == 'device':
                actual['speakers']['timeline']['device'] = 'unexpected GPU'
            elif change == 'nonfinite':
                actual['speakers']['processed_audio_seconds'] = float('nan')
            elif change == 'missing':
                actual['sections'] = 10
            elif change == 'duplicate':
                actual['provisional_revisions'] *= 2
            else:
                actual['provisional_revisions'] *= 18
            with self.subTest(change=change), self.assertRaises(ValueError):
                live_matches(actual, expected, 'cpu')

    def test_recording_and_explicit_metal_failure_preserve_bark_and_cause(self):
        reference = adapter_tests.PnolaTimelineTest().reference()
        value = {'audio_sha256': 'wav', 'text': 'exact Bark', 'asr_error': None,
                 'attribution': {'status': 'complete', 'complete': True, 'words_complete': True,
                                 'words': [{}] * 117, 'timeline': reference}}
        self.assertEqual(recorded_matches(value, reference, 'exact Bark', 'cpu', 'wav')['frames'], 6000)
        value['attribution'] = {'status': 'unavailable',
                                'reason': 'diarization backend unavailable: requested Metal device is unavailable in the native SDK'}
        self.assertTrue(recorded_matches(value, reference, 'exact Bark', 'cpu', 'wav', unavailable=True)['unsupported_metal_preserves_exact_bark_text'])
        for change in ['cause', 'text', 'wav']:
            actual = copy.deepcopy(value)
            if change == 'cause':
                actual['attribution']['reason'] = 'other failure'
            elif change == 'text':
                actual['text'] = 'changed'
            else:
                actual['audio_sha256'] = 'other input'
            with self.subTest(change=change), self.assertRaises(ValueError):
                recorded_matches(actual, reference, 'exact Bark', 'cpu', 'wav', unavailable=True)


if __name__ == '__main__':
    unittest.main()
