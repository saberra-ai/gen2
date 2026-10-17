import io
import unittest
import zipfile

from context_development_reference import published_reference, transcriber_reference


class ContextDevelopmentReferenceTests(unittest.TestCase):
    def test_lexical_segments_preserve_pauses_overlap_zero_words_and_source_bounds(self):
        archive = io.BytesIO()
        meeting = 'ES9000a'
        with zipfile.ZipFile(archive, 'w') as zipped:
            for speaker in 'ABCD':
                words = (f'<root xmlns:n="http://nite.sourceforge.net/">'
                         f'<w n:id="{speaker}1" starttime="0.2" endtime="0.4">hello</w>'
                         f'<w n:id="{speaker}2" starttime="0.8" endtime="0.8">zero</w>'
                         f'<w n:id="{speaker}3" punc="true" starttime="1" endtime="1">.</w>'
                         f'<nonvocalsound n:id="{speaker}4" starttime="0.5" endtime="0.7">cough</nonvocalsound></root>')
                segments = (f'<root xmlns:n="http://nite.sourceforge.net/">'
                            f'<segment transcriber_start="0" transcriber_end="1.2">'
                            f'<n:child href="{meeting}.{speaker}.words.xml#id({speaker}1)..id({speaker}3)"/>'
                            f'</segment><segment transcriber_start="0.5" transcriber_end="0.7">'
                            f'<n:child href="{meeting}.{speaker}.words.xml#id({speaker}4)"/></segment></root>')
                zipped.writestr(f'words/{meeting}.{speaker}.words.xml', words)
                zipped.writestr(f'segments/{meeting}.{speaker}.segments.xml', segments)
        references, words, sources, zero = transcriber_reference(archive.getvalue(), meeting, [0., 1.])
        self.assertEqual(references, [{'start': 0., 'end': 1., 'speaker': s} for s in 'ABCD'])
        self.assertEqual(len(words), 4)
        self.assertTrue(all(row['start'] == .2 and row['end'] == .4 for row in words))
        self.assertEqual(len(zero), 4)
        self.assertEqual(len(sources), 8)

    def test_published_rttm_is_cropped_without_becoming_the_manual_word_reference(self):
        text = 'SPEAKER ES9000a 1 0.2 1.2 <NA> <NA> PERSON <NA> <NA>\n'
        self.assertEqual(published_reference(text, 'ES9000a', [.5, 1.]),
                         [{'start': 0., 'end': .5, 'speaker': 'PERSON'}])
        with self.assertRaises(ValueError):
            published_reference(text.replace('ES9000a 1', 'ES9000a 2'), 'ES9000a', [.5, 1.])
        with self.assertRaises(ValueError):
            published_reference(text, 'ES9000b', [.5, 1.])
        for bad in [[1., .5], [float('nan'), 1.]]:
            with self.assertRaises(ValueError):
                published_reference(text, 'ES9000a', bad)


if __name__ == '__main__':
    unittest.main()
