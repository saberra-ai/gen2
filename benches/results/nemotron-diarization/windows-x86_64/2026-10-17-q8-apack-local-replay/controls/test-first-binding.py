"""Exercise actual publisher copy/final-read seams after source mutation."""
import ast
import hashlib
import json
from pathlib import Path,PurePosixPath
import tempfile
import unittest

script=Path('/Users/victor/.cache/gen2-nemotron3-n0/retain-q8-apack-local-proof-v3.py')
tree=ast.parse(script.read_bytes())
reader,=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='first_read']
copy,=[n for n in tree.body if isinstance(n,ast.For)and ast.unparse(n.iter)=='files.items()']
final,=[n for n in tree.body if isinstance(n,ast.For)and ast.unparse(n.iter)=='bound.items()']
def execute(nodes,env):exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-publisher-seam','exec'),env)
def descriptor(p):
    raw=p.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

class FirstBinding(unittest.TestCase):
    def fixture(self,root):
        source=root/'source';source.write_bytes(b'{"role":"original"}')
        target=root/'target';target.mkdir()
        env={**globals(),'bound':{},'target':target,'manifest':{},'files':{'payload':source}}
        execute([reader],env);self.assertEqual(json.loads(env['first_read'](source))['role'],'original')
        return source,env
    def test_unchanged_copies_first_identity(self):
        with tempfile.TemporaryDirectory()as directory:
            source,env=self.fixture(Path(directory));execute([copy,final],env)
            self.assertEqual(env['manifest']['payload'],descriptor(source))
    def test_mutation_between_validation_and_copy_rejects_before_write(self):
        with tempfile.TemporaryDirectory()as directory:
            source,env=self.fixture(Path(directory));source.write_bytes(b'{"role":"changed"}')
            with self.assertRaises(AssertionError):execute([copy],env)
            self.assertFalse((env['target']/'payload').exists())
    def test_mutation_after_copy_rejects_final_input_check(self):
        with tempfile.TemporaryDirectory()as directory:
            source,env=self.fixture(Path(directory));execute([copy],env);source.write_bytes(b'changed')
            with self.assertRaises(AssertionError):execute([final],env)

if __name__=='__main__':unittest.main()
