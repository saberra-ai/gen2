import copy
import unittest

from live_frontier_metrics import frontier_observations


class LiveFrontierMetricsTests(unittest.TestCase):
    def test_frontier_age_is_separate_from_publication_gap_and_label_latency(self):
        trace = [{'revision': 1, 'elapsed_seconds': 5.25, 'processed_audio_seconds': 4.5},
                 {'revision': 2, 'elapsed_seconds': 10.5, 'processed_audio_seconds': 9.5}]
        result = frontier_observations(trace, 12.)
        self.assertEqual(result['frontier_age_at_observation_seconds']['median'], .875)
        self.assertEqual(result['inter_observation_seconds']['maximum'], 5.25)
        self.assertEqual(result['previous_observed_frontier_age_at_next_observation_seconds']['maximum'], 6.)
        self.assertEqual(result['first_observed_revision_elapsed_seconds'], 5.25)
        for field in ['actual_speaker_or_word_label_latency_measured', 'gui_delivery_measured',
                      'stable_prefix_or_label_finality_inferred', 'finalization_delay_measured']:
            self.assertFalse(result[field])
        # The loop presents a20ms block before sleeping: retain its signed age.
        trace[0].update(elapsed_seconds=4.49)
        self.assertAlmostEqual(frontier_observations(trace, 12.)['observations'][0]['signed_frontier_age_seconds'], -.01)

    def test_missing_revisions_and_impossible_or_nonmonotonic_clocks_are_rejected(self):
        original = [{'revision': 1, 'elapsed_seconds': 5.25, 'processed_audio_seconds': 4.5},
                    {'revision': 2, 'elapsed_seconds': 10.5, 'processed_audio_seconds': 9.5}]
        for field, value in [('revision', 3), ('revision', True), ('elapsed_seconds', 4.),
                             ('processed_audio_seconds', 4.), ('elapsed_seconds', float('nan')),
                             ('processed_audio_seconds', 11.), ('processed_audio_seconds', 13.)]:
            trace = copy.deepcopy(original)
            trace[1][field] = value
            with self.assertRaises(ValueError):
                frontier_observations(trace, 12.)
        with self.assertRaises(ValueError):
            frontier_observations(original[:1], 12.)


if __name__ == '__main__':
    unittest.main()
