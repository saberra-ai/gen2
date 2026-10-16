"""A successful CTest exit must also prove the one requested case ran."""
import unittest
from ctest_output import require_single_test_pass


class CtestOutput(unittest.TestCase):
    row = '1/1 Test #7: diar_state .......................   Passed    0.28 sec\n'

    def test_both_observed_ctest_summary_formats(self):
        for summary in ['100% tests passed, 0 tests failed out of 1', '100% tests passed out of 1']:
            require_single_test_pass(self.row + '\n' + summary + '\n', 'diar_state')

    def test_missing_wrong_failed_skipped_or_multiple_tests_are_rejected(self):
        summary = '100% tests passed out of 1\n'
        for output in ['No tests were found!!!', summary, self.row,
                       self.row.replace('diar_state', 'other') + summary,
                       self.row.replace('Passed', 'Skipped') + summary,
                       self.row.replace('Passed', 'Failed') + summary,
                       self.row + summary.replace('100%', '0%'),
                       self.row + summary.replace('of 1', 'of 0'),
                       self.row + self.row + summary, self.row + summary + summary,
                       self.row.replace('1/1', '1/2') + '2/2 Test #8: other .... Passed 0.1 sec\n100% tests passed out of 2\n']:
            with self.subTest(output=output), self.assertRaises(ValueError):
                require_single_test_pass(output, 'diar_state')


if __name__ == '__main__':
    unittest.main()
