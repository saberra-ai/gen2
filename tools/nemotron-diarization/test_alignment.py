import unittest
from qualify_alignment import evaluate, token_words


class AlignmentTest(unittest.TestCase):
    def test_subword_grouping_preserves_measured_intervals(self):
        self.assertEqual(token_words([
            {"text": "Hel", "startms": 80, "endms": 160},
            {"text": "lo world!", "startms": 160, "endms": 400},
        ]), [("hello", .08, .4), ("world", .16, .4)])

    def test_deleted_words_are_not_hidden_by_timing_matches(self):
        result = evaluate([("a", 0., .1), ("short", .2, .3), ("reply", .4, .5)],
                          [("a", .08, .18), ("reply", .48, .58)])
        self.assertEqual(result["deletions"], 1)
        self.assertEqual(result["matched_words"], 2)
        self.assertAlmostEqual(result["chronological_merged_wer"], 1/3)
        self.assertAlmostEqual(result["matched_start_error_median_seconds"], .08)


if __name__ == "__main__":
    unittest.main()
