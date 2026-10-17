from pathlib import Path
import subprocess
import tempfile
import unittest

from build_valid_mel_candidate import descriptor
from frozen_git_sources import verify_git_sources


class FrozenGitSourceTests(unittest.TestCase):
    def test_recorded_bytes_survive_checkout_changes_but_other_revisions_cannot_replace_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL).decode().strip()
            git('init')
            git('config', 'user.name', 'Model-free fixture')
            git('config', 'user.email', 'fixture@example.invalid')
            source = root / 'input.txt'
            source.write_bytes(b'original')
            expected = {'input.txt': descriptor(source)}
            git('add', 'input.txt')
            git('commit', '-m', 'original')
            original = git('rev-parse', 'HEAD')
            source.write_bytes(b'changed!')  # Same size: a digest check is required.
            git('add', 'input.txt')
            git('commit', '-m', 'changed')
            changed = git('rev-parse', 'HEAD')
            verify_git_sources(root, original, expected)
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                verify_git_sources(root, changed, expected)
            source.write_bytes(b'uncommitted checkout data')
            verify_git_sources(root, original, expected)
            for commit, pins in [('0' * 40, expected), (original, {'missing': expected['input.txt']}),
                                 (original, {'input.txt': {**expected['input.txt'], 'bytes': 1}})]:
                with self.subTest(commit=commit, pins=pins), self.assertRaises(ValueError):
                    verify_git_sources(root, commit, pins)
            git('replace', changed, original)
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                verify_git_sources(root, changed, expected)

    def test_promisor_repository_cannot_fetch_missing_payloads_during_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server, consumer = root / 'server', root / 'consumer'
            server.mkdir()
            def git(*args):
                return subprocess.check_output(['git', '-C', str(server), *args], stderr=subprocess.DEVNULL).decode().strip()
            git('init')
            git('config', 'user.name', 'Model-free fixture')
            git('config', 'user.email', 'fixture@example.invalid')
            git('config', 'uploadpack.allowFilter', 'true')
            source = server / 'input.txt'
            source.write_bytes(b'promisor payload')
            expected = {'input.txt': descriptor(source)}
            git('add', 'input.txt')
            git('commit', '-m', 'source')
            commit = git('rev-parse', 'HEAD')
            subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', server.as_uri(), str(consumer)],
                           check=True, capture_output=True)
            with self.assertRaisesRegex(ValueError, 'size/type differs'):
                verify_git_sources(consumer, commit, expected)
