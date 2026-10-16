"""Regression: different cache policies cannot qualify as quantization parity."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


class MatchedReference(unittest.TestCase):
    def test_geometry_mismatch_is_rejected_before_quality_is_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            probabilities = np.zeros((100, 8), dtype=np.float32)
            probabilities[:, 0] = .9
            np.save(root / "reference.npy", probabilities)
            geometry = {"chunk": 9, "right": 4, "left": 0, "fifo": 264, "cache": 264, "update": 222}
            (root / "reference.json").write_text(json.dumps({"audio_samples": 16000, "geometry": geometry}))
            np.r_[probabilities, probabilities[-1:]].tofile(root / "native.json.f32")
            native = {"frames": 101, "audio_seconds": 1, "geometry": geometry,
                      "segments": [{"start": 0, "end": 1, "speaker": 1}]}
            path = root / "native.json"
            path.write_text(json.dumps(native))
            (root / "ref.rttm").write_text("SPEAKER x 1 0 1 <NA> <NA> alice <NA> <NA>\n")
            command = [sys.executable, str(Path(__file__).with_name("compare.py")),
                       "--native", str(path), "--reference", str(root / "reference"),
                       "--rttm", str(root / "ref.rttm"), "--output", str(root / "comparison.json")]
            good = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            native["geometry"] = {**geometry, "fifo": 80}
            path.write_text(json.dumps(native))
            bad = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("streaming geometry differs", bad.stderr)


if __name__ == "__main__":
    unittest.main()
