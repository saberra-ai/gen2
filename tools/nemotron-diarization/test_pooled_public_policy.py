import json
from pathlib import Path
import platform
import subprocess
import tempfile
import unittest

from qualify_macos_pooled_public import runtime_policies


class PooledPublicPolicy(unittest.TestCase):
    @unittest.skipUnless(platform.system() == 'Darwin', 'actual macOS sandbox controls')
    def test_only_lifecycle_can_read_assertion_copies_and_premodel_cannot_read_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)  # Exercise /var -> /private/var canonicalization.
            baseline, reference, results, assets = [root / name for name in ['baseline', 'reference', 'results', 'assets']]
            for directory in [baseline, reference, results, assets]:
                directory.mkdir()
            original, assertion, prior, model = baseline / 'native.json', reference / 'expected.f32', results / 'prior.json', assets / 'model.gguf'
            for file in [original, assertion, prior, model]:
                file.write_bytes(b'not a model')
            policies = runtime_policies([baseline], [baseline, reference, assets], reference, results)
            for name, text in policies.items():
                with self.subTest(profile=name):
                    sandbox = root / (name + '.sb')
                    sandbox.write_text(text)
                    allowed = ([assertion] if name == 'lifecycle' else []) + ([model] if name != 'premodel' else [])
                    denied = [original, prior] + ([assertion] if name != 'lifecycle' else []) + ([model] if name == 'premodel' else [])
                    code = '''from pathlib import Path
import json,socket,sys
def denied(operation):
 try: operation()
 except PermissionError: return
 raise RuntimeError('operation unexpectedly permitted')
p=json.loads(sys.argv[1])
assert Path(p['metadata']).exists()
for file in p['allowed']: assert Path(file).read_bytes()==b'not a model'
for file in p['denied']: denied(lambda file=file: Path(file).read_bytes())
for file in p['readonly']: denied(lambda file=file: Path(file).write_bytes(b'changed'))
denied(lambda: socket.create_connection(('127.0.0.1',9),timeout=1))
Path(p['output']).write_bytes(b'published')
'''
                    published = results / (name + '.json')
                    data = {'metadata': str(prior), 'allowed': list(map(str, allowed)),
                            'denied': list(map(str, denied)), 'readonly': list(map(str, [original, assertion, model])),
                            'output': str(published)}
                    observed = subprocess.run(['/usr/bin/sandbox-exec', '-f', str(sandbox), '/usr/bin/python3',
                        '-c', code, json.dumps(data)], capture_output=True, text=True, timeout=10)
                    self.assertEqual(observed.returncode, 0, observed.stderr)
                    self.assertEqual(published.read_bytes(), b'published')
