"""Actual metadata/member/source validation-to-copy AST seams; fixtures only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

parser=argparse.ArgumentParser()
parser.add_argument('--publisher',type=Path,required=True)
parser.add_argument('--report',type=Path,required=True)
args=parser.parse_args()
original=args.publisher.read_bytes();tree=ast.parse(original)
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'desc','read','remember','bind'}]
inputs,=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='inputs' for t in n.targets)]
copy_assert,=[n for n in ast.walk(tree) if isinstance(n,ast.Assert) and isinstance(n.test,ast.Compare)
    and isinstance(n.test.left,ast.Dict) and ast.unparse(n.test.comparators[0])=='inputs[str(path)]']
member_binding,=[n for n in tree.body if isinstance(n,ast.For) and ast.unparse(n.iter)=='members.items()']
source_binding,=[n for n in tree.body if isinstance(n,ast.For) and ast.unparse(n.iter)=="p['tools'].items()"]


def fragment(nodes):return compile(ast.Module(body=nodes,type_ignores=[]),'<actual publisher seam>','exec')


def scope():
    result={'hashlib':hashlib,'json':json,'Path':Path,'bound':{}}
    exec(fragment(functions),result)
    return result


def copy_check(env,path):
    env['files']={'payload':path};exec(fragment([inputs]),env)
    raw=path.read_bytes();env.update(path=path,value=raw,raw=raw)
    exec(fragment([copy_assert]),env)


class PublicationSeamTest(unittest.TestCase):
    def test_unchanged_validated_metadata_copies(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'metadata.json';path.write_text('{"role":"control"}')
            env=scope();self.assertEqual(env['read'](path)['role'],'control')
            copy_check(env,path)

    def test_metadata_changed_after_validation_before_copy_binding_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'metadata.json';path.write_text('{"role":"control"}')
            env=scope();self.assertEqual(env['read'](path)['role'],'control')
            path.write_text('{"role":"candidate"}')
            with self.assertRaises(AssertionError):copy_check(env,path)

    def test_vector_changed_after_member_admission_before_copy_binding_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'vector.f32';path.write_bytes(bytes.fromhex('0000803e'))
            env=scope();env.update(delivered=Path(folder),members={path.name:env['desc'](path)})
            exec(fragment([member_binding]),env)
            path.write_bytes(bytes.fromhex('0000003f'))
            with self.assertRaises(AssertionError):copy_check(env,path)

    def test_tool_changed_after_frozen_source_admission_before_copy_binding_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            repo=Path(folder);path=repo/'tools/nemotron-diarization/fixture.py'
            path.parent.mkdir(parents=True);path.write_bytes(b'original source')
            env=scope();env.update(repo=repo,p={'tools':{path.name:env['desc'](path)}},files={})
            exec(fragment([source_binding]),env)
            path.write_bytes(b'changed source')
            with self.assertRaises(AssertionError):copy_check(env,path)


result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PublicationSeamTest))
assert args.publisher.read_bytes()==original
report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'passed':result.wasSuccessful(),'publisher':{'bytes':len(original),'sha256':hashlib.sha256(original).hexdigest()},
    'scope':'Actual publisher read/member/source/input-binding/raw-copy AST only; temporary fixtures; no CI artifact, compiler, model or publication invoked.'}
with args.report.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
if not result.wasSuccessful():raise SystemExit(1)
