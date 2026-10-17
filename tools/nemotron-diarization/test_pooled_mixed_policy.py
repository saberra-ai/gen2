import json
from pathlib import Path
import platform
import subprocess
import tempfile
import unittest

from qualify_macos_pooled_mixed import admit_builtin_mel_table, compiler_policy, mixed_policy, mixed_runtime_policy, verify_vendor_worktree
from build_valid_mel_candidate import descriptor
from qualify_macos_portable_runtime import CONTROL


class PooledMixedPolicy(unittest.TestCase):
    def test_real_mel_table_requires_a_frozen_pin_before_any_preparation(self):
        repo = Path(__file__).resolve().parents[2]
        name = 'assets/bark/nemo128.f32'
        original = json.loads(Path(__file__).with_name('macos-pooled-mixed.protocol.json').read_text())
        self.assertNotIn(name, original['repository_files'])
        with self.assertRaisesRegex(ValueError, 'coefficient source pin'):
            admit_builtin_mel_table(repo, original)
        pinned = {'repository_files': {name: descriptor(repo / name)}}
        self.assertEqual(admit_builtin_mel_table(repo, pinned), repo / name)
        pinned['repository_files'][name] = {**pinned['repository_files'][name], 'sha256': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'coefficient source pin'):
            admit_builtin_mel_table(repo, pinned)

    def test_ignored_cpp_extra_is_rejected_even_when_git_status_is_clean(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE)
            git('init', '-q')
            (root / 'baseline.cpp').write_text('int original;\n')
            git('add', 'baseline.cpp')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'source')
            revision = git('rev-parse', 'HEAD').decode().strip()
            verify_vendor_worktree(root, revision)
            (root / '.git/info/exclude').write_text('extra.cpp\n')
            extra = root / 'extra.cpp'
            extra.write_text('int extra;\n')
            self.assertEqual(git('status', '--porcelain=v1', '--untracked-files=all'), b'')
            with self.assertRaisesRegex(ValueError, 'vendor'):
                verify_vendor_worktree(root, revision)
            extra.unlink()
            verify_vendor_worktree(root, revision)

    @unittest.skipUnless(platform.system() == 'Darwin', 'actual macOS sandbox controls')
    def test_only_exact_builtin_mel_table_is_readable_during_compilation(self):
        with tempfile.TemporaryDirectory(prefix='mixed coefficients ü ') as temporary:
            root = Path(temporary).resolve()
            models, source = root / 'models', root / 'source'
            models.mkdir()
            source.mkdir()
            mel, other = source / 'nemo128.f32', root / 'nemo128.f32'
            probes = [root / ('weights.' + suffix) for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
            for file in [mel, other, models / 'model.data', *probes]:
                file.write_bytes(b'not a model')
                self.assertEqual(file.read_bytes(), b'not a model')
            sandbox = root / 'compile.sb'
            sandbox.write_text(compiler_policy(mixed_policy(denied_reads=[models], readonly=[source, models]), mel))
            result = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(sandbox), '/usr/bin/python3', '-c',
                CONTROL, json.dumps({'reads': list(map(str, [other, models / 'model.data', *probes])),
                                     'writes': [str(mel), str(models / 'model.data')]})],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(sandbox), '/usr/bin/python3', '-c',
                'from pathlib import Path;import sys;assert Path(sys.argv[1]).read_bytes()==b"not a model"', str(mel)],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(platform.system() == 'Darwin', 'actual macOS sandbox controls')
    def test_unicode_runtime_paths_deny_prior_payloads_but_allow_output_and_metadata(self):
        with tempfile.TemporaryDirectory(prefix='mixed outputs ü ') as temporary:
            root = Path(temporary).resolve()
            results, source, models = [root / name for name in ['results', 'source', 'models']]
            for directory in [results, source, models]:
                directory.mkdir()
            prior, hidden, model = results / 'prior.json', source / 'Cargo.toml', models / 'model.gguf'
            for file in [prior, hidden, model]:
                file.write_bytes(b'not a model')
            sandbox = root / 'runtime.sb'
            sandbox.write_text(mixed_runtime_policy([source], [source, models], results))
            code = '''from pathlib import Path
import json,socket,sys
def denied(fn):
 try: fn()
 except PermissionError: return
 raise RuntimeError('operation unexpectedly permitted')
p=json.loads(sys.argv[1])
assert Path(p['prior']).exists()
assert Path(p['model']).read_bytes()==b'not a model'
for name in ['prior','hidden']: denied(lambda name=name: Path(p[name]).read_bytes())
denied(lambda: Path(p['model']).write_bytes(b'changed'))
denied(lambda: socket.create_connection(('127.0.0.1',9),timeout=1))
Path(p['output']).write_bytes(b'published')
'''
            published = results / 'published.json'
            data = {'prior': str(prior), 'hidden': str(hidden), 'model': str(model), 'output': str(published)}
            result = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(sandbox), '/usr/bin/python3', '-c',
                code, json.dumps(data)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(published.read_bytes(), b'published')
