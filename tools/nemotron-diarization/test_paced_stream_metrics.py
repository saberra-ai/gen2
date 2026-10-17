import copy
import unittest

from paced_stream_metrics import paced_timing


def trace():
    session = '00000000-0000-4000-8000-000000000001'
    pushes = [{'endpoint_samples': i * 320, 'nominal_endpoint_micros': i * 20000,
               'ready_micros': i * 20000, 'accepted_micros': i * 20000 + 10, 'busy_retries': 0}
              for i in range(1, 4003)]
    turn = {'start_seconds': 0., 'end_seconds': 1., 'speaker': 1}
    observations = [{'session_id': session, 'revision': i, 'complete': False,
        'poll_started_micros': i * 5000000 + 100, 'observed_micros': i * 5000000 + 101,
        'accepted_samples': i * 80000, 'processed_audio_seconds': i * 5. - .4,
        'native_compute_micros': i * 50000, 'turns': [turn]} for i in range(1, 17)]
    observations.append({'session_id': session, 'revision': 17, 'complete': True,
        'poll_started_micros': 80061000, 'observed_micros': 80061001, 'accepted_samples': 1280640,
        'processed_audio_seconds': 80.04, 'native_compute_micros': 900000, 'turns': [turn]})
    return {'session_id': session, 'complete': True, 'repeat_finish_cached': True, 'accepted_samples': 1280640,
        'audio_duration_seconds': 80.04, 'native_frame_count': 8004, 'pushes': pushes, 'observations': observations,
        'turns': [turn], 'finish_requested_micros': 80041000, 'finish_returned_micros': 80061000,
        'elapsed_micros': 80061000, 'finish_micros': 20000, 'native_compute_micros': 900000,
        'poll_count': 4003, 'poll_micros': 500, 'cold_load_micros': 1000, 'stream_open_micros': 100}


class PacedStreamMetricsTests(unittest.TestCase):
    def test_finish_coalescing_is_explicit_and_timing_is_not_called_label_latency(self):
        value = trace()
        report = paced_timing(value)
        self.assertEqual(report['public_finish_drain_flush_delivery_seconds'], .02)
        self.assertEqual(report['native_compute_seconds'], .9)
        self.assertFalse(report['actual_speaker_or_word_label_latency_measured'])
        self.assertEqual(report['unobserved_revision_before_final'], 0)
        value['observations'].pop(-2)
        self.assertEqual(paced_timing(value)['unobserved_revision_before_final'], 1)

    def test_invalid_input_finalization_and_observed_timelines_cannot_be_sanitized(self):
        def change(value, kind):
            if kind == 'future': value['pushes'][0]['ready_micros'] -= 1
            if kind == 'missing_push': value['pushes'].pop()
            if kind == 'missing_revision': value['observations'].pop(3)
            if kind == 'late_provisional': value['observations'][-2]['observed_micros'] = 90000000
            if kind == 'wrong_final_turns': value['observations'][-1]['turns'] = []
            if kind == 'bad_turn': value['observations'][0]['turns'][0]['end_seconds'] = 900.
            if kind == 'nan': value['observations'][0]['processed_audio_seconds'] = float('nan')
            if kind == 'finish_before_input': value['finish_requested_micros'] = 10
            if kind == 'wrong_finish_duration': value['finish_micros'] += 1
            if kind == 'poll_accounting': value['poll_count'] -= 1
            if kind == 'wrong_session': value['observations'][0]['session_id'] = 'different'
        for kind in ['future', 'missing_push', 'missing_revision', 'late_provisional', 'wrong_final_turns',
                     'bad_turn', 'nan', 'finish_before_input', 'wrong_finish_duration', 'poll_accounting', 'wrong_session']:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                value = copy.deepcopy(trace())
                change(value, kind)
                paced_timing(value)


if __name__ == '__main__':
    unittest.main()
