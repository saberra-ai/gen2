"""Actual returned AST positives and mutations; stdlib only, no project imports."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
protocol = cache / 'thread-cpu-returned-evidence-replay-v3.pre-execution.json'
raw = protocol.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'e06cf7d972026d7966c8416c75d1e9916ef6235e65db214d037d68a6ef39b331'
pre = json.loads(raw)
script = cache / 'replay-thread-cpu-returned-evidence-v3.py'
source = script.read_bytes()
assert {'bytes': len(source), 'sha256': hashlib.sha256(source).hexdigest()} == pre['files'][str(script)]
functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
             and node.name in ('require', 'reviewed_body')]
assert len(functions) == 2
scope = {'ast': ast}
exec(compile(ast.Module(body=functions, type_ignores=[]), 'admitted-pure-AST-comparison', 'exec'), scope)
delivered = Path(pre['local']) / 'materialized' / pre['result_path']
positives, rejected = [], []
with tempfile.TemporaryDirectory(prefix='thread-cpu-ast-bindings-', dir=cache) as temporary:
    for name, relative in pre['reviewed_originals'].items():
        original = Path(pre['repo']) / relative
        original_raw = original.read_bytes()
        assert {'bytes': len(original_raw), 'sha256': hashlib.sha256(original_raw).hexdigest()} == pre['files'][str(original)]
        returned = delivered / name
        observed = scope['reviewed_body'](original, returned, pre['literal_relocation'])
        assert ast.dump(ast.Module(body=observed, type_ignores=[]), include_attributes=False) == ast.dump(ast.parse(original_raw), include_attributes=False)
        positives.append(name)
        actual = returned.read_text()
        mutations = {
            'appended-statement': actual + '\nraise RuntimeError("must reject before execution")\n',
            'changed-assertion': actual.replace("q['source_commit'] == source", "q['source_commit'] != source", 1),
            'wrong-path-relocation': actual.replace('/home/runner/work/gen2/gen2', '/home/runner/work/other/repo', 1),
        }
        for label, changed in mutations.items():
            assert changed != actual
            path = Path(temporary) / (name + '-' + label)
            path.write_text(changed)
            try:
                scope['reviewed_body'](original, path, pre['literal_relocation'])
            except ValueError as error:
                assert str(error) == 'returned validation AST differs from reviewed original'
                rejected.append({'script': name, 'mutation': label})
            else:
                raise AssertionError('mutated returned AST accepted')
assert len(positives) == 2 and len(rejected) == 6
report = {'schema_version': 1, 'positive_actual_returned_asts': positives, 'rejected_ast_mutations': rejected,
          'project_imports_or_returned_code_execution': 0,
          'test_source': {'bytes': len(Path(__file__).read_bytes()),
                          'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
with (cache / 'thread-cpu-replay-ast-binding-controls-v1.json').open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
print(json.dumps(report))
