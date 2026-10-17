"""Admit exactly one actually passed CTest case across supported summary formats."""
import re


def require_single_test_pass(output, name):
    rows = [line.strip() for line in output.splitlines()
            if re.match(r"\d+/\d+ Test #", line.strip())]
    pattern = r"1/1 Test #\d+: " + re.escape(name) + r"\s+\.+\s+Passed\s+\d+(?:\.\d+)? sec"
    summaries = [line.strip() for line in output.splitlines()
                 if re.match(r"\d+% tests passed", line.strip())]
    if (len(rows) != 1 or re.fullmatch(pattern, rows[0]) is None
            or len(summaries) != 1
            or re.fullmatch(r"100% tests passed(?:, 0 tests failed)? out of 1", summaries[0]) is None):
        raise ValueError("required model-free state test did not execute and pass")
