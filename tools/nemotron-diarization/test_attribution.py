import unittest
from qualify_attribution import score


class AttributionTest(unittest.TestCase):
    def test_unknown_words_cannot_be_permuted_into_a_known_person(self):
        reference = [{"text": "yes", "speaker": "A", "start_seconds": 0, "end_seconds": .2}, {"text": "no", "speaker": "B", "start_seconds": 1, "end_seconds": 1.2}]
        result = {"status": "complete", "complete": True, "words": [{"text": "yes", "attribution": {"kind": "speaker", "speaker": 2}}, {"text": "no", "attribution": {"kind": "unknown", "reason": "timing_gap"}}]}
        metrics = score(reference, result)
        self.assertEqual(metrics["chronological_merged_wer"], 0)
        self.assertEqual(metrics["conservative_cpwer"], 1)
        self.assertEqual(metrics["speaker_tagged_word_error_rate"], .5)
        self.assertEqual(metrics["unknown_words"], 1)
        self.assertEqual(metrics["speaker_mapping"], {"2": "A"})


if __name__ == "__main__":
    unittest.main()
