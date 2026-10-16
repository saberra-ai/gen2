"""Downloader bounds and human-reference semantics, without network/models."""
import io
import struct
import unittest
from unittest.mock import patch
import wave
import zipfile

import corpus


class Response(io.BytesIO):
    def __init__(self, data, status=206, **headers):
        super().__init__(data)
        self.status = status
        self.headers = headers
        self.read_called = False

    def read(self, *args):
        self.read_called = True
        return super().read(*args)


def archive(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buffer.getvalue()


class CorpusTests(unittest.TestCase):
    def test_range_rejects_whole_archive_before_reading(self):
        response = Response(b'large', status=200)
        with patch('corpus.urllib.request.urlopen', return_value=response):
            with self.assertRaisesRegex(ValueError, 'exact frozen range'):
                corpus.fetch('https://example.invalid/audio', 4, start=4, end=7, total=100, etag='"pin"')
        self.assertFalse(response.read_called)

    def test_range_requires_exact_extent_etag_and_body(self):
        good = {'Content-Range': 'bytes 4-7/100', 'ETag': '"pin"', 'Content-Length': '4'}
        for updates, body in [({'ETag': '"new"'}, b'abcd'), ({'Content-Range': 'bytes 0-3/100'}, b'abcd'),
                              ({'Content-Length': '5000000'}, b'abcd'), ({}, b'abc')]:
            with self.subTest(updates=updates, body=body):
                response = Response(body, **{**good, **updates})
                with patch('corpus.urllib.request.urlopen', return_value=response):
                    with self.assertRaises(ValueError):
                        corpus.fetch('https://example.invalid/audio', 4, start=4, end=7, total=100, etag='"pin"')
        response = Response(b'abcd', **good)
        with patch('corpus.urllib.request.urlopen', return_value=response) as open_url:
            self.assertEqual(corpus.fetch('https://example.invalid/audio', 4, start=4, end=7, total=100, etag='"pin"'), b'abcd')
        self.assertEqual(open_url.call_args.args[0].get_header('If-match'), '"pin"')

    def test_deflate_member_exact_size_crc_path_and_no_trailing_payload(self):
        data = b'audio-fixture' * 20
        raw = archive({'voxconverse_test_wav/test.wav': data})
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            info = z.getinfo('voxconverse_test_wav/test.wav')
        member = {'id': 'test', 'offset': 0, 'bytes': len(data), 'compressed_bytes': info.compress_size,
                  'crc32': f'{info.CRC:08x}'}
        source = {'url': 'https://example.invalid/source.zip', 'bytes': len(raw), 'etag': '"pin"'}

        def read(url, limit, **kwargs):
            self.assertEqual(kwargs['etag'], '"pin"')
            return raw[kwargs['start']:kwargs['end'] + 1]

        with patch('corpus.fetch', side_effect=read):
            self.assertEqual(corpus.zip_audio(source, member), data)
            for changes in [{'id': 'other'}, {'bytes': len(data) - 1}, {'crc32': '00000000'},
                            {'bytes': corpus.MAX_MEMBER + 1}, {'compressed_bytes': info.compress_size + 1}]:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    corpus.zip_audio(source, {**member, **changes})
        descriptor = bytearray(raw)
        struct.pack_into('<H', descriptor, 6, 8)
        struct.pack_into('<II', descriptor, 14, 0, 0)
        raw = bytes(descriptor)
        with patch('corpus.fetch', side_effect=read):
            self.assertEqual(corpus.zip_audio(source, member), data)
        # Preserve matching header metadata while corrupting the payload itself.
        damaged = bytearray(raw)
        offset = 30 + len(info.filename)
        damaged[offset + info.compress_size // 2] ^= 1
        raw = bytes(damaged)
        with patch('corpus.fetch', side_effect=read), self.assertRaises((ValueError, corpus.zlib.error)):
            corpus.zip_audio(source, member)

    def test_pcm_header_supports_extra_chunks_and_rejects_wrong_geometry(self):
        out = io.BytesIO()
        with wave.open(out, 'wb') as wav:
            wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            wav.writeframes(b'\0\0' * 320)
        original = out.getvalue()
        extra = b'JUNK' + struct.pack('<I', 3) + b'abc\0'
        raw = original[:12] + extra + original[12:]
        raw = raw[:4] + struct.pack('<I', len(raw) - 8) + raw[8:]
        self.assertEqual(corpus.pcm_header(raw[:100], len(raw)), (56, 320))
        for change in [b'RF64' + raw[4:], raw[:4] + b'\0'*4 + raw[8:], raw[:36]]:
            with self.assertRaises(ValueError):
                corpus.pcm_header(change, len(raw))
        changed = bytearray(original)
        struct.pack_into('<I', changed, 24, 44100)
        with self.assertRaisesRegex(ValueError, 'PCM16'):
            corpus.pcm_header(changed, len(changed))

    def test_ami_transcriber_reference_clips_full_segment_not_word_union(self):
        files = {}
        for speaker in 'ABCD':
            word_name = f'ES2004a.{speaker}.words.xml'
            files['words/' + word_name] = f'''<root xmlns:nite="http://nite.sourceforge.net/">
              <vocalsound nite:id="noise" starttime="2" endtime="3"/>
              <w nite:id="word" starttime="4" endtime="5">Yes</w>
              <w nite:id="dot" starttime="5" endtime="5" punc="true">.</w></root>'''
            files[f'segments/ES2004a.{speaker}.segments.xml'] = f'''<root xmlns:nite="http://nite.sourceforge.net/">
              <segment transcriber_start="1" transcriber_end="3"><nite:child href="{word_name}#id(noise)"/></segment>
              <segment transcriber_start="3" transcriber_end="8"><nite:child href="{word_name}#id(word)..id(dot)"/></segment></root>'''
        refs, words, _ = corpus.ami_reference(archive(files), [4.5, 7])
        self.assertEqual(len(refs), 4)
        self.assertEqual(refs[0], {'start': 0., 'end': 2.5, 'speaker': 'A'})
        self.assertEqual(words[0]['end'], .5)
        files['segments/ES2004a.A.segments.xml'] = files['segments/ES2004a.A.segments.xml'].replace('id(word)..id(dot)', 'id(dot)..id(word)')
        with self.assertRaisesRegex(ValueError, 'reversed'):
            corpus.ami_reference(archive(files), [4.5, 7])

    def test_icsi_noise_attributes_are_not_transcribed_words(self):
        xml = b'''<!DOCTYPE Meeting SYSTEM "http://www.icsi.berkeley.edu/Speech/mr/Meeting1.dtd">
        <Meeting><Segment Participant="a" StartTime="10" EndTime="12"><NonVocalSound Description="noise"/></Segment>
        <Segment Participant="b" StartTime="12" EndTime="14">Yes <Emphasis>one</Emphasis> two.</Segment>
        <Segment Participant="c" StartTime="14" EndTime="14">zero</Segment></Meeting>'''
        refs, texts, _, noises = corpus.icsi_reference(archive({'transcripts/Bmr005.mrt': xml}), [11, 13])
        self.assertEqual(refs, [{'start': 1., 'end': 2, 'speaker': 'b'}])
        self.assertEqual(texts[0]['text'], 'Yes one two.')
        self.assertEqual(len(noises), 1)
        with self.assertRaises(ValueError):
            corpus.xml_root(b'<!DOCTYPE x [<!ENTITY x "boom">]><x>&x;</x>')

    def test_reference_union_and_overlap_count_distinct_speakers(self):
        values = [{'start': 0, 'end': 2, 'speaker': 'a'}, {'start': 1, 'end': 3, 'speaker': 'a'},
                  {'start': 2, 'end': 4, 'speaker': 'b'}]
        stats = corpus.summarize(values)
        self.assertEqual(stats['speaker_seconds'], {'a': 3, 'b': 2})
        self.assertEqual(stats['speech_union_seconds'], 4)
        self.assertEqual(stats['overlap_seconds'], 1)
        self.assertEqual(corpus.rttm_reference(b'SPEAKER test 1 1.0 2.0 <NA> <NA> a <NA> <NA>', 'test', 2),
                         [{'start': 1., 'end': 2, 'speaker': 'a'}])
        with self.assertRaises(ValueError):
            corpus.rttm_reference(b'SPEAKER test 1 1.0 nan <NA> <NA> a <NA> <NA>', 'test', 2)


if __name__ == '__main__':
    unittest.main()
