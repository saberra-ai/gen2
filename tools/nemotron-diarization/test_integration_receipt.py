from pathlib import Path
import tempfile
import unittest
from capture_integration import require_pass, validate_soak


class ReceiptTest(unittest.TestCase):
    def test_ignored_test_cannot_count_as_executed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.log"
            path.write_text("test actual_model ... ignored\ntest result: ok. 0 passed; 0 failed; 1 ignored\n")
            with self.assertRaises(AssertionError):
                require_pass(path, ["actual_model"])

    def test_incomplete_soak_cannot_count_as_long_session(self):
        with self.assertRaises(AssertionError):
            validate_soak({"complete": False}, "248889344 maximum resident set size", 3600)


if __name__ == "__main__":
    unittest.main()
