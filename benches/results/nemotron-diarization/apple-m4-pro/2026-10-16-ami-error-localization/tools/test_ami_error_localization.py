import unittest
from types import SimpleNamespace

from diagnose_ami_errors import partition


def accumulate(stats, segments, _words, mapping, _info):
    # Analytic synthetic oracle: test partitions and mapping independently of
    # the pinned official scorer used in real diagnostic execution.
    for row in segments:
        duration, ref, hyp = row['TDUR'], row['REF'], row['SYS']
        contributions = {'SCORED_TIME': duration, 'SCORED_SPEAKER': duration * len(ref),
                         'MISSED_SPEAKER': duration * max(len(ref) - len(hyp), 0),
                         'FALARM_SPEAKER': duration * max(len(hyp) - len(ref), 0),
                         'SPEAKER_ERROR': duration * (min(len(ref), len(hyp)) - sum(mapping.get(s) in hyp for s in ref))}
        for key, value in contributions.items():
            stats[key] = stats.get(key, 0) + value


ENGINE = SimpleNamespace(_score_speaker_segments=accumulate)


class ErrorLocalizationTests(unittest.TestCase):
    def test_global_mapping_retains_confusion_and_silence_false_alarm(self):
        reference = [{'start': 0, 'end': 60, 'speaker': 'A'}, {'start': 60, 'end': 90, 'speaker': 'B'}]
        timeline = [{'TBEG': a, 'TEND': b, 'REF': ref, 'SYS': {'1': 1}} for a, b, ref in
                    [(0, 60, {'A': 1}), (60, 90, {'B': 1}), (90, 120, {})]]
        result = partition(ENGINE, timeline, reference, {'A': '1'}, 120)
        self.assertEqual(result['total']['SPEAKER_ERROR'], 30)
        self.assertEqual(result['partitions']['occupancy']['silence']['FALARM_SPEAKER'], 30)
        self.assertIsNone(result['partitions']['occupancy']['silence']['der'])
        self.assertEqual(result['partitions']['crop_time']['first_30_seconds']['SCORED_TIME'], 30)
        self.assertEqual(result['partitions']['crop_time']['remaining_90_seconds']['SCORED_TIME'], 90)
        self.assertEqual(result['partitions']['reference_span']['short_only']['SCORED_TIME'], 0)

    def test_short_and_mixed_overlap_bins_are_exhaustive(self):
        reference = [{'start': 0, 'end': 1, 'speaker': 'A'}, {'start': .5, 'end': 2, 'speaker': 'B'}]
        timeline = [{'TBEG': a, 'TEND': b, 'REF': ref, 'SYS': {}} for a, b, ref in
                    [(0, .5, {'A': 1}), (.5, 1, {'A': 1, 'B': 1}), (1, 2, {'B': 1}), (2, 120, {})]]
        result = partition(ENGINE, timeline, reference, {}, 120)
        self.assertEqual(result['total']['MISSED_SPEAKER'], 2.5)
        self.assertEqual(result['partitions']['reference_span']['short_only']['SCORED_TIME'], .5)
        self.assertEqual(result['partitions']['reference_span']['mixed']['MISSED_SPEAKER'], 1)
        self.assertEqual(result['partitions']['reference_span']['long_only']['SCORED_TIME'], 1)
        self.assertEqual(result['partitions']['occupancy']['overlap']['SCORED_SPEAKER'], 1)

    def test_reference_disagreement_and_wrong_uem_are_rejected(self):
        with self.assertRaises(ValueError):
            partition(ENGINE, [{'TBEG': 0, 'TEND': 120, 'REF': {'A': 1}, 'SYS': {}}], [], {}, 120)
        with self.assertRaises(ValueError):
            partition(ENGINE, [], [], {}, 60)


if __name__ == '__main__':
    unittest.main()
