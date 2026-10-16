import unittest

from diagnose_ami_reference import crop_reference


class ReferenceCropTests(unittest.TestCase):
    def test_preserves_overlap_and_clips_both_crop_edges(self):
        rows = '\n'.join('SPEAKER ES2004a 1 ' + value + ' <NA> <NA> ' + speaker + ' <NA> <NA>'
                         for value, speaker in [('299 2', 'A'), ('300.5 2', 'B'), ('419 2', 'A'), ('420 1', 'C')])
        self.assertEqual(crop_reference(rows, 'ES2004a', 300, 420), [
            {'start': 0, 'end': 1, 'speaker': 'A'},
            {'start': .5, 'end': 2.5, 'speaker': 'B'},
            {'start': 119, 'end': 120, 'speaker': 'A'}])

    def test_rejects_unexpected_recording_and_nonfinite_duration(self):
        for row in ['SPEAKER other 1 300 1 <NA> <NA> A <NA> <NA>',
                    'SPEAKER ES2004a 1 300 nan <NA> <NA> A <NA> <NA>']:
            with self.assertRaises(ValueError):
                crop_reference(row, 'ES2004a', 300, 420)


if __name__ == '__main__':
    unittest.main()
