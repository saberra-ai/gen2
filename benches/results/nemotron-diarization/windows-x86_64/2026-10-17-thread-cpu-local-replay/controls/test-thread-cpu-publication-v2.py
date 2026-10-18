"""Publication input and final-tree admission controls; no project imports."""
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
source = (cache / 'publish-thread-cpu-local-proof-v2.py').read_bytes()
assert hashlib.sha256(source).hexdigest() == '0da5c88a8c9bfe1475a1adf230452d4d46d55c7285e0a8c35efc3f5dfe98ce46'
nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name != 'main']
scope = {'hashlib': hashlib, 'stat': stat, 'Path': Path}
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'admitted-publication-primitives', 'exec'), scope)
passed = []


def reject(label, function, *args):
    try:
        function(*args)
    except ValueError:
        passed.append(label)
    else:
        raise AssertionError('negative control accepted: ' + label)


with tempfile.TemporaryDirectory(prefix='thread-cpu-publication-controls-', dir=cache) as temporary:
    root = Path(temporary)
    evidence = root / 'input'
    evidence.write_bytes(b'first-bound-evidence')
    first = scope['identity'](evidence.read_bytes())
    assert scope['capture'](evidence, first) == b'first-bound-evidence'
    passed.append('first-bound-byte-positive')
    evidence.write_bytes(b'changed-evidence')
    reject('changed-first-bound-bytes', scope['capture'], evidence, first)
    tree = root / 'output'
    tree.mkdir()
    (tree / 'a').mkdir()
    (tree / 'a/file').write_bytes(b'proof')
    files = {'a/file': scope['identity'](b'proof')}
    scope['verify_tree'](tree, files)
    passed.append('closed-tree-positive')
    for label, target in [('broken-symlink', root / 'missing'), ('directory-symlink', tree / 'a'),
                          ('regular-file-symlink', tree / 'a/file')]:
        link = tree / 'extra'
        link.symlink_to(target)
        reject(label, scope['verify_tree'], tree, files)
        link.unlink()
    special = tree / 'extra'
    os.mkfifo(special)
    reject('special-fifo', scope['verify_tree'], tree, files)
    special.unlink()
    special.mkdir()
    reject('unlisted-empty-directory', scope['verify_tree'], tree, files)
    special.rmdir()
    (tree / 'a/file').unlink()
    reject('missing-member', scope['verify_tree'], tree, files)
    for names in [('A/file','a/other'), ('a','A'), ('pre-execution.json','PRE-EXECUTION.JSON')]:
        reject('case-collision:' + repr(names), scope['unique_names'], names)
    for name in ('.','../escape','/absolute','a/../escape'):
        reject('unsafe-path:' + name, scope['relative_name'], name)
assert len(passed) == 16
raw = Path(__file__).read_bytes()
report = {'schema_version': 1, 'checks': passed, 'checks_passed': len(passed),
          'publisher_sha256': hashlib.sha256(source).hexdigest(),
          'test_source': {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()},
          'project_imports_or_compiler_sdk_model_calls': 0}
with (cache / 'thread-cpu-publication-controls-v2.json').open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps(report))
