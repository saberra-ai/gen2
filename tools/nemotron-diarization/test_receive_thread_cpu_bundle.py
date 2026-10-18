"""Real quarantine-only Git delivery and rejection of mutated transport evidence."""
import copy
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

import receive_thread_cpu_bundle as receiver
from remote_q8_apack_replay import bounded_process,descriptor,verify_collection


class BundleReceiverTest(unittest.TestCase):
    def metadata(self):
        run={'id':receiver.RUN,'head_sha':receiver.VERIFIER,'status':'completed','conclusion':'success',
            'repository':{'full_name':'saberra-ai/gen2'}}
        job={'id':receiver.JOB,'run_id':receiver.RUN,'head_sha':receiver.VERIFIER,'status':'completed','conclusion':'success'}
        artifact={'id':receiver.DELIVERY['id'],'name':'nemotron-thread-cpu-replay','expired':False,
            'workflow_run':{'id':receiver.RUN,'head_sha':receiver.VERIFIER},
            'size_in_bytes':receiver.DELIVERY['bytes'],'digest':'sha256:'+receiver.DELIVERY['sha256']}
        return run,job,{'artifacts':[artifact]}

    def test_exact_terminal_source_and_frozen_delivery_identity(self):
        receiver.admit_api(*self.metadata())
        for group,key,value in [(0,'head_sha','0'*40),(0,'status','in_progress'),(0,'conclusion','failure'),
                (1,'id',receiver.JOB+1),(2,'id',receiver.DELIVERY['id']+1),(2,'expired',True),
                (2,'digest','sha256:'+'0'*64),(2,'size_in_bytes',receiver.DELIVERY['bytes']+1)]:
            values=copy.deepcopy(self.metadata());target=values[group]if group<2 else values[2]['artifacts'][0]
            target[key]=value
            with self.subTest(key=key,group=group),self.assertRaises(ValueError):receiver.admit_api(*values)

    def test_duplicate_json_and_unsafe_paths_reject(self):
        with self.assertRaises(ValueError):receiver.strict_json(b'{"a":1,"a":2}')
        for name in ['../x','/x','a//b','a/./b','a\\b','.git/config','a/.GIT/config','.', 'a\nfile']:
            with self.subTest(name=name),self.assertRaises(ValueError):receiver.safe_name(name)

    def write_zip(self,path,extra=None):
        names=['receiver-result.bundle','bundle.json','receiver-result.bundle.process.json','bundle.stderr.log',
            'storage-admission.json','commands.json','pre-execution.json','archive.stderr.log',
            'source-preparation/collection.json','source-preparation/qualification.json']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(path,'w')as out:
                for name in names:out.writestr(name,b'fixture')
                if extra is not None:
                    name,raw,mode=extra;item=zipfile.ZipInfo(name);item.external_attr=mode<<16;out.writestr(item,raw)

    def test_zip_closure_traversal_symlink_duplicate_unknown_and_expansion(self):
        for extra in [None,('../outside',b'x',stat.S_IFREG),('commands/link',b'target',stat.S_IFLNK),
                ('bundle.json',b'duplicate',stat.S_IFREG),('BUNDLE.JSON',b'case',stat.S_IFREG),
                ('unknown',b'x',stat.S_IFREG)]:
            with self.subTest(extra=extra),tempfile.TemporaryDirectory()as directory:
                root=Path(directory);archive=root/'delivery.zip';self.write_zip(archive,extra)
                if extra is None:
                    self.assertEqual(len(receiver.extract_delivery(archive,root/'out')),10)
                else:
                    with self.assertRaises(ValueError):receiver.extract_delivery(archive,root/'out')
                    self.assertFalse((root/'out').exists())
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);self.write_zip(root/'delivery.zip')
            with mock.patch.object(receiver,'EXTRACT_CAP',1),self.assertRaises(ValueError):
                receiver.extract_delivery(root/'delivery.zip',root/'out')
            self.assertFalse((root/'out').exists())

    def test_fixed_bundle_metadata_rejects_false_provenance(self):
        with tempfile.TemporaryDirectory()as directory:
            bundle=Path(directory)/'data';bundle.write_bytes(b'fixture')
            value={'verifier_source_commit':receiver.VERIFIER,'verifier_run_id':str(receiver.RUN),
                'native_source_commit':receiver.NATIVE_SOURCE,'native_run_id':receiver.NATIVE_RUN,
                'native_job_id':receiver.NATIVE_JOB,'native_archive':receiver.ARCHIVE,'result_parent':receiver.VERIFIER,
                'result_roots':list(receiver.RESULTS),'collections':{name:{}for name in receiver.RESULTS},
                'bundle':descriptor(bundle),'original_start_floor_bytes':8*1024**3,'new_native_calls':0,
                'production_or_release_qualified':False,'result_commit':'1'*40}
            with mock.patch.multiple(receiver,RESULT_COMMIT='1'*40,BUNDLE=descriptor(bundle),COLLECTION={}):
                receiver.admit_bundle_metadata(value,bundle)
            for key,bad in [('native_archive',{}),('result_parent','0'*40),('result_roots',['wrong']),
                    ('production_or_release_qualified',True),('new_native_calls',1),('bundle',{})]:
                changed={**value,key:bad}
                with self.subTest(key=key),mock.patch.multiple(receiver,RESULT_COMMIT='1'*40,BUNDLE=descriptor(bundle),COLLECTION={}),self.assertRaises(ValueError):receiver.admit_bundle_metadata(changed,bundle)

    @unittest.skipUnless(os.name=='posix'and shutil.which('git'),'real Git/owned POSIX processes')
    def test_real_bundle_quarantine_preserves_source_and_rejects_outside_link_or_corruption(self):
        for mutation in (None,'outside','symlink','corrupt','empty-tree'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory()as directory:
                root=Path(directory);producer=root/'producer';producer.mkdir();source=root/'source';source.mkdir()
                def git(cwd,*args,check=True,input=None):
                    return subprocess.run(['git','-c','user.name=Fixture','-c','user.email=fixture@invalid',
                        '-c','gc.auto=0','-c','core.hooksPath=/dev/null','-C',str(cwd),*map(str,args)],
                        capture_output=True,timeout=30,check=check,input=input)
                git(producer,'init','-q','-b','main');(producer/'base').write_text('existing')
                git(producer,'add','.');git(producer,'commit','-q','-m','base')
                base=git(producer,'rev-parse','HEAD').stdout.decode().strip()
                git(source,'init','-q','-b','main');git(source,'fetch','-q','--depth=1',producer,base)
                git(source,'checkout','-q','--detach','FETCH_HEAD')
                for name in receiver.RESULTS:
                    dest=producer/name;dest.mkdir(parents=True);(dest/'payload').write_bytes(b'verified payload')
                    manifest={'files':{'payload':descriptor(dest/'payload')}}
                    (dest/'collection.json').write_text(json.dumps(manifest)+'\n')
                collections={name:verify_collection(producer/name)for name in receiver.RESULTS}
                if mutation=='outside':(producer/'outside').write_text('wrong scope')
                if mutation=='symlink':(producer/receiver.RESULTS[0]/'link').symlink_to('payload')
                if mutation=='corrupt':(producer/receiver.RESULTS[0]/'payload').write_bytes(b'corrupt')
                git(producer,'add','.');git(producer,'commit','-q','-m','result')
                if mutation=='empty-tree':
                    empty=git(producer,'mktree',input=b'').stdout.decode().strip()
                    root_tree=git(producer,'mktree',input=git(producer,'ls-tree','HEAD').stdout+
                        f'040000 tree {empty}\tempty-outside\n'.encode()).stdout.decode().strip()
                    extra=git(producer,'commit-tree',root_tree,'-p',base,'-m','empty tree outside result scope').stdout.decode().strip()
                    git(producer,'update-ref','HEAD',extra)
                result=git(producer,'rev-parse','HEAD').stdout.decode().strip();bundle=root/'result.bundle'
                bounded_process(['git','bundle','create','-','HEAD','^'+base],bundle,root/'bundle.stderr',
                    cwd=producer,env=dict(os.environ),timeout=30,limit=128*1024)
                output=root/'output';output.mkdir();logs=output/'logs';logs.mkdir()
                def run(name,command,stdout_path=None,limit=8*1024**2):
                    path=stdout_path or logs/(name+'.out')
                    bounded_process(list(map(str,command)),path,logs/(name+'.err'),cwd=source,
                        env={**os.environ,'GIT_NO_REPLACE_OBJECTS':'1','GIT_NO_LAZY_FETCH':'1'},timeout=30,limit=limit)
                    return path.read_bytes()if stdout_path is None else b''
                metadata={'verifier_source_commit':base,'result_commit':result,'collections':collections}
                initial_index=(source/'.git/index').read_bytes()
                if mutation is None:
                    self.assertEqual(receiver.quarantine_results(bundle,metadata,source,output,run),collections)
                else:
                    with self.assertRaises(ValueError):receiver.quarantine_results(bundle,metadata,source,output,run)
                self.assertEqual(git(source,'rev-parse','HEAD').stdout.decode().strip(),base)
                self.assertEqual((source/'.git/index').read_bytes(),initial_index)
                self.assertEqual(git(source,'status','--porcelain').stdout,b'')
                self.assertNotEqual(git(source,'cat-file','-e',result+'^{commit}',check=False).returncode,0)

    def test_first_bound_helper_executes_only_exact_original_bytes(self):
        original=Path(receiver.__file__).with_name('remote_q8_apack_replay.py').read_bytes()
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);helper=root/'helper.py';helper.write_bytes(original)
            functions=receiver.frozen_process_helpers(helper)
            self.assertEqual(len(functions),3)
            self.assertEqual(functions[0](helper),descriptor(helper))
            marker=root/'executed'
            helper.write_bytes(original+('\nPath('+repr(str(marker))+').write_text("wrong")\n').encode())
            with self.assertRaises(ValueError):receiver.frozen_process_helpers(helper)
            self.assertFalse(marker.exists())

    def test_quarantine_and_header_algorithms_match_original_receiver(self):
        import ast
        current=ast.parse(Path(receiver.__file__).read_bytes())
        original=ast.parse(Path(receiver.__file__).with_name('receive_q8_apack_bundle.py').read_bytes())
        for name in ('strict_json','safe_name','bundle_header','quarantine_results'):
            a=next(n for n in current.body if isinstance(n,ast.FunctionDef)and n.name==name)
            b=next(n for n in original.body if isinstance(n,ast.FunctionDef)and n.name==name)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))


if __name__=='__main__':unittest.main()
