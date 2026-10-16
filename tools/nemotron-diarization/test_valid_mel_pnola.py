import copy
import os
import subprocess
import sys
import unittest
from unittest.mock import Mock

from qualify_valid_mel_pnola import execute_guarded, timeline_matches


class PnolaTimelineTest(unittest.TestCase):
    def reference(self):
        return {'runtime_version': 'nemo-speech-asr 0.2.0+gen2-valid-mel-v1',
                'model_revision': 'f667ed73aee57d40cc39428eb768b4fd87a0a29e',
                'backend': 'cpu', 'audio_duration_seconds': 60., 'native_frame_count': 6000,
                'geometry': {'chunk': 9, 'right': 4, 'left': 0, 'fifo': 264, 'cache': 264, 'update': 222},
                'turns': [{'start_seconds': 0., 'end_seconds': 1., 'speaker': 1}]}

    def test_persisted_turn_rounding_accepts_only_declared_tolerance(self):
        expected = self.reference(); actual = copy.deepcopy(expected)
        actual['turns'][0]['end_seconds'] += 0.0000001
        self.assertLess(timeline_matches(actual, expected, 60., 'cpu')['maximum_turn_error_seconds'], 1e-6)
        actual['turns'][0]['end_seconds'] += 0.00001
        with self.assertRaises(ValueError):
            timeline_matches(actual, expected, 60., 'cpu')

    def test_compact_live_schema_preserves_turn_checks_without_inventing_frames(self):
        expected = self.reference(); actual = copy.deepcopy(expected)
        actual['geometry'] = [9, 4, 0, 264, 264, 222]
        del actual['native_frame_count']
        result = timeline_matches(actual, expected, 60., 'cpu', compact=True)
        self.assertIsNone(result['frames'])
        self.assertFalse(result['native_frame_count_observed'])
        actual['geometry'][0] = 8
        with self.assertRaises(ValueError):
            timeline_matches(actual, expected, 60., 'cpu', compact=True)

    def test_bad_identity_placement_and_nonfinite_outputs_rejected(self):
        expected = self.reference()
        for field, value in [('runtime_version', 'nemo-speech-asr 0.2.0'), ('native_frame_count', 6001),
                             ('audio_duration_seconds', float('nan')), ('backend', 'metal'), ('device', 'GPU')]:
            actual = copy.deepcopy(expected); actual[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                timeline_matches(actual, expected, 60., 'cpu')
        for field in ['start_seconds', 'end_seconds']:
            actual = copy.deepcopy(expected); actual['turns'][0][field] = float('nan')
            with self.assertRaises(ValueError):
                timeline_matches(actual, expected, 60., 'cpu')


class PnolaProcessCleanupTest(unittest.TestCase):
    def test_pid_persistence_failure_terminates_group_before_releasing_reserve(self):
        guard = Mock(); record = {}; reaped_before_release = []
        def check_reaped():
            with self.assertRaises(ProcessLookupError):
                os.kill(record['pid'], 0)
            reaped_before_release.append(True)
        guard.release.side_effect = check_reaped
        with self.assertRaises(OSError):
            execute_guarded([sys.executable, '-c', 'import time; time.sleep(30)'],
                            cwd='.', environment=dict(os.environ), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, guard=guard, record=record, timeout=1,
                            persist=Mock(side_effect=OSError('simulated metadata disk full')))
        guard.wait.assert_not_called()
        self.assertEqual(reaped_before_release, [True])
        self.assertEqual(record['returncode'], -9)
        self.assertTrue(record['forced_process_group_termination'])


if __name__ == '__main__':
    unittest.main()
