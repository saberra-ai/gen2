"""Corrupt/oversized/redirected/deadline bodies must never become an admitted IPSW."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from macos_restore_download import acquire_restore


class RestoreDownloadTests(unittest.TestCase):
    def artifact(self, data):
        return {'name': 'fixed.ipsw', 'url': 'https://updates.cdn-apple.com/fixed.ipsw',
                'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    def response(self, data, artifact):
        response = io.BytesIO(data)
        response.status = 200
        response.headers = {}
        response.geturl = lambda: artifact['url']
        return response

    def run_download(self, directory, artifact, response, record=None):
        guard = Mock(running_minimum=1)
        guard.check.return_value = 2
        opener = Mock()
        opener.open.return_value = response
        with patch('macos_restore_download.urllib.request.build_opener', return_value=opener):
            target = acquire_restore(artifact, directory, guard=guard, record={} if record is None else record,
                persist=Mock(), deadline_seconds=10, read_timeout_seconds=1)
        return target

    def test_exact_body_publishes_without_partial_or_extra_link(self):
        data = b'complete pinned image'
        artifact = self.artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            record = {}
            target = self.run_download(directory, artifact, self.response(data, artifact), record)
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(target.stat().st_nlink, 1)
            self.assertFalse(target.with_suffix('.ipsw.partial').exists())
            self.assertTrue(record['complete'])

    def test_bad_body_or_response_is_retained_as_failure_and_never_published(self):
        artifact = self.artifact(b'correct body')
        for mutation in ['hash', 'short', 'oversize', 'redirect', 'status', 'length', 'encoding']:
            data = {'hash': b'incorrect!!!', 'short': b'a', 'oversize': b'correct body extra'}.get(mutation, b'correct body')
            response = self.response(data, artifact)
            if mutation == 'redirect':
                response.geturl = lambda: 'https://other.example/fixed.ipsw'
            elif mutation == 'status':
                response.status = 206
            elif mutation == 'length':
                response.headers = {'Content-Length': '999'}
            elif mutation == 'encoding':
                response.headers = {'Content-Encoding': 'gzip'}
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                record = {}
                with self.assertRaises(ValueError):
                    self.run_download(directory, artifact, response, record)
                self.assertFalse((Path(directory) / artifact['name']).exists())
                self.assertFalse(record['complete'])
                self.assertIn('error', record)

    def test_existing_target_is_not_overwritten_or_requested(self):
        artifact = self.artifact(b'new')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / artifact['name']
            target.write_bytes(b'original')
            with patch('macos_restore_download.urllib.request.build_opener') as opener:
                with self.assertRaises(FileExistsError):
                    acquire_restore(artifact, directory, guard=Mock(), record={}, persist=Mock(),
                        deadline_seconds=10, read_timeout_seconds=1)
            opener.assert_not_called()
            self.assertEqual(target.read_bytes(), b'original')

    def test_total_deadline_after_read_still_prevents_publication(self):
        artifact = self.artifact(b'image')
        clock = [0.]
        response = self.response(b'image', artifact)
        def delayed_read(size):
            clock[0] = 11.
            return b'image'
        response.read1 = delayed_read
        with tempfile.TemporaryDirectory() as directory, patch('macos_restore_download.time.monotonic', side_effect=lambda: clock[0]):
            with self.assertRaises(TimeoutError):
                self.run_download(directory, artifact, response)
            self.assertFalse((Path(directory) / artifact['name']).exists())
