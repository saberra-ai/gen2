import unittest

from loading_info_json import jsonable_loading_info


class LoadingInfoJsonTests(unittest.TestCase):
    def test_empty_actual_loading_report_types_preserved_without_mutation(self):
        report = {'missing_keys': set(), 'unexpected_keys': set(), 'mismatched_keys': set(), 'error_msgs': []}
        self.assertEqual(jsonable_loading_info(report), {key: [] for key in report})
        self.assertIsInstance(report['missing_keys'], set)

    def test_nonempty_diagnostics_and_shape_tuples_survive_in_stable_order(self):
        report = {'missing_keys': {'z', 'a'}, 'mismatched_keys': {('weight', (2, 3), (3, 2))},
                  'error_msgs': ['load failed'], 'extra': {'conversion_error': 'retained'}}
        self.assertEqual(jsonable_loading_info(report), {'missing_keys': ['a', 'z'],
                         'mismatched_keys': [['weight', [2, 3], [3, 2]]], 'error_msgs': ['load failed'],
                         'extra': {'conversion_error': 'retained'}})

    def test_unsupported_values_raise_instead_of_losing_the_diagnostic(self):
        with self.assertRaises(TypeError):
            jsonable_loading_info({'unexpected': object()})


if __name__ == '__main__':
    unittest.main()
