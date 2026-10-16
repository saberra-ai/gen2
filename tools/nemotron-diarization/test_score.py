"""Independently worked DER examples, including overlap and speaker permutation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DerScoring(unittest.TestCase):
    def score(self, reference, segments):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "ref.rttm").write_text(reference)
            (root / "hyp.json").write_text(json.dumps({"segments": segments}))
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("score.py")),
                                     str(root / "ref.rttm"), str(root / "hyp.json"),
                                     "--duration", "2"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def test_label_permutation_is_not_an_error(self):
        result = self.score("SPEAKER x 1 0 1 <NA> <NA> alice <NA> <NA>\n"
                            "SPEAKER x 1 1 1 <NA> <NA> bob <NA> <NA>\n",
                            [{"start": 0, "end": 1, "speaker": 2},
                             {"start": 1, "end": 2, "speaker": 1}])
        self.assertEqual(result["der"], 0)

    def test_missing_overlap_is_a_missed_speaker_not_confusion(self):
        result = self.score("SPEAKER x 1 0 2 <NA> <NA> alice <NA> <NA>\n"
                            "SPEAKER x 1 0 2 <NA> <NA> bob <NA> <NA>\n",
                            [{"start": 0, "end": 2, "speaker": 1}])
        self.assertEqual(result["der"], .5)
        self.assertEqual(result["missed_speaker_seconds"], 2)
        self.assertEqual(result["confusion_speaker_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
