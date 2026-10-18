"""Exercise actual publisher validation-to-copy seams using pure AST fragments."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

parser = argparse.ArgumentParser()
parser.add_argument('--publisher', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
original = args.publisher.read_bytes()
tree = ast.parse(original)
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'desc', 'bind', 'read'}]
inputs, = [node for node in tree.body if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == 'inputs' for target in node.targets)]
copy_assert, = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)
    and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Dict)
    and ast.unparse(node.test.comparators[0]) == 'inputs[str(path)]']
object_binding, = [node for node in tree.body if isinstance(node, ast.For)
    and ast.unparse(node.iter) == "inspection['inputs'].items()"]


def fragment(nodes): return compile(ast.Module(body=nodes, type_ignores=[]), '<actual publisher seam>', 'exec')


def scope():
    result = {'hashlib': hashlib, 'json': json, 'Path': Path, 'bound': {}}
    exec(fragment(functions), result)
    return result


def copy_check(env, path):
    env['files'] = {'payload': path}
    exec(fragment([inputs]), env)
    env.update(path=path, raw=path.read_bytes())
    exec(fragment([copy_assert]), env)


class PublicationSeamTest(unittest.TestCase):
    def test_unchanged_validated_metadata_copies(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'metadata.json'; p.write_text('{"role":"control"}')
            env = scope(); self.assertEqual(env['read'](p)['role'], 'control')
            copy_check(env, p)

    def test_change_after_metadata_validation_before_copy_binding_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'metadata.json'; p.write_text('{"role":"control"}')
            env = scope(); self.assertEqual(env['read'](p)['role'], 'control')
            p.write_text('{"role":"candidate"}')
            with self.assertRaises(AssertionError): copy_check(env, p)

    def test_change_after_inspected_object_admission_before_copy_binding_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'kernel.obj'; p.write_bytes(b'inspected object')
            env = scope(); expected = env['desc'](p)
            env['inspection'] = {'inputs': {str(p): expected}}
            exec(fragment([object_binding]), env)
            p.write_bytes(b'different object')
            with self.assertRaises(AssertionError): copy_check(env, p)


result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PublicationSeamTest))
assert args.publisher.read_bytes() == original
report = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
    'passed': result.wasSuccessful(), 'publisher': {'bytes': len(original), 'sha256': hashlib.sha256(original).hexdigest()},
    'scope': 'Actual publisher read/admitted-object/input-binding/raw-copy assertion AST only; temporary fixtures; no publication, artifact, compiler or native call.'}
with args.report.open('x') as stream: stream.write(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
if not result.wasSuccessful(): raise SystemExit(1)
