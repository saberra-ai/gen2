"""Real PCM conversion and source-unit uncertainty must not manufacture gold."""
import io
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
import zipfile

import numpy as np

from prepare_word_timing_review import canonical_audio, descriptor, read_pinned, review_template, source_units, wav_bytes


class WordTimingReviewTests(unittest.TestCase):
    def test_wrong_large_or_nonregular_annotation_input_is_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'annotations.zip'
            path.write_bytes(b'zip fixture')
            expected = descriptor(path.read_bytes())
            self.assertEqual(read_pinned(path, expected), b'zip fixture')
            with path.open('wb') as stream:
                stream.truncate(64 * 1024 * 1024)
            with patch.object(Path, 'open', side_effect=AssertionError('read before size/type admission')):
                with self.assertRaisesRegex(ValueError, 'exact bounded size'):
                    read_pinned(path, expected)
                with self.assertRaisesRegex(ValueError, 'regular file'):
                    read_pinned(Path(temporary), expected)

    def header(self):
        # Source declares the full recording; acquisition retains only64headerbytes.
        raw = bytearray(wav_bytes(b'\0' * 20, 2, 20000))
        raw[4:8] = (21207164 - 8).to_bytes(4, 'little')
        raw[40:44] = (5301780 * 4).to_bytes(4, 'little')
        return bytes(raw)

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'SciPy required for actual PCM resampling')
    def test_opposite_stereo_channels_cancel_without_invented_activity(self):
        stereo = np.empty((1200000, 2), dtype='<i2')
        stereo[:, 0] = 1234
        stereo[:, 1] = -1234
        header = self.header()
        original, mono = canonical_audio(header, stereo.tobytes(), {'header': descriptor(header)})
        with wave.open(io.BytesIO(mono)) as wav:
            self.assertEqual((wav.getnchannels(), wav.getframerate(), wav.getnframes()), (1, 16000, 960000))
            self.assertEqual(set(wav.readframes(960000)), {0})
        with wave.open(io.BytesIO(original)) as wav:
            self.assertEqual(wav.readframes(1200000), stereo.tobytes())
        with self.assertRaisesRegex(ValueError, 'header identity'):
            canonical_audio(header + b'x', stereo.tobytes(), {'header': descriptor(header)})
        with self.assertRaisesRegex(ValueError, 'geometry'):
            canonical_audio(header, stereo.tobytes()[:-4], {'header': descriptor(header)})

    def test_multiword_and_crop_crossing_units_remain_unreviewed(self):
        archive = io.BytesIO()
        expected = {}
        with zipfile.ZipFile(archive, 'w') as zipped:
            name = 'maptaskv2-1/Data/corpus-resources/maptask-corpus.xml'
            metadata = (b'<root><conv id="q1ec1"><pointer role="g" href="maptask-participants.xml#id(q1eta1)"/>'
                        b'<pointer role="f" href="maptask-participants.xml#id(q1etb1)"/></conv></root>')
            zipped.writestr(name, metadata)
            expected[name] = descriptor(metadata)
            for role, text, start, end in [('g', 'going to', 59.5, 60.2), ('f', 'yes', 4., 4.2)]:
                name = f'maptaskv2-1/Data/timed-units/q1ec1.{role}.timed-units.xml'
                data = f'<root><tu id="{role}1" start="{start}" end="{end}">{text}</tu></root>'.encode()
                zipped.writestr(name, data)
                expected[name] = descriptor(data)
        units = source_units(archive.getvalue(), expected)
        self.assertEqual([u['text'] for u in units], ['yes', 'going to'])
        self.assertTrue(units[1]['crosses_crop_edge'])
        self.assertEqual(units[1]['source_end_seconds'], 60.2)
        self.assertTrue(units[1]['contains_multiple_space_separated_words'])
        self.assertTrue(all(u['human_acoustic_boundary_review_verified'] is False for u in units))
        template = review_template({'sha256': 'audio'}, units)
        self.assertEqual(template['words'], [])
        self.assertEqual(template['annotators'], [])
        self.assertEqual(template['status'], 'unreviewed')
        self.assertFalse(template['gold_admitted'])
        self.assertFalse(template['release_qualified'])


if __name__ == '__main__':
    unittest.main()
