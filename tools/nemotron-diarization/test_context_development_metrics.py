import unittest

from context_development_metrics import production_turns


class ContextDevelopmentMetricsTests(unittest.TestCase):
    def test_large_context_bad_timeline_is_rejected_before_full_recording_scoring(self):
        def turn(start, end, speaker=1):
            return {'start_seconds': start, 'end_seconds': end, 'speaker': speaker}

        for invalid in [turn(1., 900.), turn(400., 500.), turn(-.01, 1.),
                        turn(0., float('nan')), turn(0., 1., True), turn(0., 1., 9)]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                production_turns([invalid])
        self.assertEqual(production_turns([turn(299., 300.00005), turn(20., 20.)]),
                         [{'start': 299., 'end': 300., 'speaker': 1}])


if __name__ == '__main__':
    unittest.main()
