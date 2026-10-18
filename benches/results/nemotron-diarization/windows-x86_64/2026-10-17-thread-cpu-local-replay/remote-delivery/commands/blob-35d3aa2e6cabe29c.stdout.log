"""Frozen API mutations and exact prepared-AST relocation; no artifact/native execution."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import remote_thread_cpu_replay as target


class RemoteThreadCPUReplayTest(unittest.TestCase):
    def fixtures(self):
        run={'id':target.RUN,'head_sha':target.SOURCE,'repository':{'full_name':'saberra-ai/gen2'},
             'status':'completed','conclusion':'success'}
        job={'id':target.JOB,'run_id':target.RUN,'head_sha':target.SOURCE,'status':'completed','conclusion':'success'}
        artifact={'id':77,'name':'nemotron-windows-thread-cpu-profile','expired':False,
            'workflow_run':{'id':target.RUN,'head_sha':target.SOURCE},'size_in_bytes':123,
            'digest':'sha256:'+'a'*64}
        return run,job,{'total_count':1,'artifacts':[artifact]}

    def test_exact_terminal_api_and_all_identity_mutations(self):
        with patch.multiple(target,ARTIFACT=77,ARCHIVE={'bytes':123,'sha256':'a'*64}):
            original=self.fixtures();target.admit_api(*original)
            mutations=[(0,['id'],1),(0,['head_sha'],'b'*40),(0,['repository','full_name'],'other/repo'),
                (0,['status'],'in_progress'),(0,['conclusion'],'failure'),(1,['id'],1),
                (1,['run_id'],1),(1,['head_sha'],'b'*40),(1,['status'],'in_progress'),
                (1,['conclusion'],'failure'),(2,['total_count'],2),(2,['artifacts',0,'id'],78),
                (2,['artifacts',0,'name'],'other'),(2,['artifacts',0,'expired'],True),
                (2,['artifacts',0,'workflow_run','id'],1),(2,['artifacts',0,'workflow_run','head_sha'],'b'*40),
                (2,['artifacts',0,'size_in_bytes'],124),(2,['artifacts',0,'digest'],'sha256:'+'b'*64)]
            for index,keys,value in mutations:
                changed=copy.deepcopy(original);parent=changed[index]
                for key in keys[:-1]:parent=parent[key]
                parent[keys[-1]]=value
                with self.subTest(keys=keys),self.assertRaises(ValueError):target.admit_api(*changed)
            for artifacts in ([],[original[2]['artifacts'][0]]*2):
                with self.assertRaises(ValueError):target.admit_api(original[0],original[1],{'total_count':1,'artifacts':artifacts})

    def test_literal_relocation_preserves_calls_and_other_literals(self):
        original=("from pathlib import Path\nrepo=Path("+repr(target.OLD_REPO)+")\n"
            "cache=Path("+repr(target.OLD_CACHE+'/data')+")\nvalue='untouched'\nprint(repo/cache)\n").encode()
        result=target.relocate(original,Path('/new/repo'),Path('/new/cache'))
        expected=original.decode().replace(target.OLD_REPO,'/new/repo').replace(target.OLD_CACHE,'/new/cache')
        self.assertEqual(ast.dump(ast.parse(result),include_attributes=False),ast.dump(ast.parse(expected),include_attributes=False))
        self.assertNotIn(target.OLD_REPO.encode(),result);self.assertNotIn(target.OLD_CACHE.encode(),result)

    def test_all_actual_prepared_receivers_relocate_only_literal_prefixes(self):
        root=Path(__file__).resolve().parents[2]/target.PARENT
        for name,path in [(n,root/'prepared'/n)for n in target.EXECUTED]+[
                ('source',root/'prepare-thread-cpu-source-v3.py'),('controls',root/'test-thread-cpu-receiver-bindings-v1.py')]:
            raw=path.read_bytes();changed=target.relocate(raw,Path('/fixture/repo'),Path('/fixture/cache'))
            old_calls=[ast.dump(n,include_attributes=False)for n in ast.walk(ast.parse(raw))if isinstance(n,ast.Call)]
            new_calls=[ast.dump(n,include_attributes=False)for n in ast.walk(ast.parse(changed))if isinstance(n,ast.Call)]
            # Compare the complete expected relocated AST, including every call.
            expected=raw.decode().replace(target.OLD_REPO,'/fixture/repo').replace(target.OLD_CACHE,'/fixture/cache')
            with self.subTest(name=name):
                self.assertEqual(ast.dump(ast.parse(changed),include_attributes=False),ast.dump(ast.parse(expected),include_attributes=False))
                self.assertEqual(len(old_calls),len(new_calls))

    def test_streamed_descriptor_matches_full_bytes_and_dump_is_exclusive(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'data';raw=b'\x00\xff\n'*400000;path.write_bytes(raw)
            self.assertEqual(target.descriptor(path),{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
            output=Path(d)/'q.json';target.dump(output,{'ok':True})
            with self.assertRaises(FileExistsError):target.dump(output,{'ok':False})

    @unittest.skipUnless(os.name=='posix','actual owned receiver preparation child')
    def test_actual_preparation_and_owned_binding_controls_before_network(self):
        from remote_q8_apack_replay import bounded_process
        tree=ast.parse(Path(target.__file__).read_bytes())
        body=next(n.body for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
        first=next(i for i,n in enumerate(body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='parent'for t in n.targets))
        last=next(i for i,n in enumerate(body[first:],first)if isinstance(n,ast.For)and isinstance(n.target,ast.Tuple)
            and [t.id for t in n.target.elts]==['name','url','revision'])
        repo_root=Path(target.__file__).resolve().parents[2]
        protocol=json.loads((repo_root/'tools/nemotron-diarization/remote-thread-cpu-replay.protocol.json').read_bytes())
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);cache=root/'cache';cache.mkdir();output=root/'output';output.mkdir();logs=output/'commands';logs.mkdir()
            repo=root/'repo';repo.symlink_to(repo_root,target_is_directory=True)
            fixed={};calls=[]
            def frozen():
                for name,wanted in fixed.items():
                    if target.descriptor(Path(name))!=wanted:raise ValueError('first-bound fixture changed')
            def run(name,command):
                self.assertEqual(name,'binding-controls');calls.append(name)
                row=bounded_process(list(map(str,command)),logs/(name+'.stdout.log'),logs/(name+'.stderr.log'),
                    cwd=repo,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},timeout=30)
                self.assertTrue(row['completed']and row['leader_reaped']and row['group_gone'])
                return(logs/(name+'.stdout.log')).read_bytes()
            scope={**vars(target),'repo':repo,'cache':cache,'output':output,'logs':logs,'protocol':protocol,
                'fixed':fixed,'frozen':frozen,'run':run,'base':'fixture'}
            exec(compile(ast.Module(body[first:last],type_ignores=[]),'actual-receiver-preparation','exec'),scope)
            self.assertEqual(calls,['binding-controls'])
            report=json.loads((cache/'thread-cpu-receiver-binding-controls-v1.json').read_bytes())
            self.assertEqual(len(report['rejected_mutations']),15)
            self.assertFalse(json.loads((output/'pre-execution.json').read_bytes())['artifact_consumed'])
            self.assertEqual(len(scope['relocated']),5)
            for path in [cache/target.EXECUTED[0],cache/'thread-cpu-receivers-prepared-v3'/target.EXECUTED[0]]:
                original=path.read_bytes();path.write_bytes(original+b'\n# changed\n')
                with self.assertRaises(ValueError):scope['prepared_frozen']()
                path.write_bytes(original);scope['prepared_frozen']()

    @unittest.skipUnless(os.name=='posix','real outer-owned stage process groups')
    def test_inherited_child_success_and_output_cap(self):
        from remote_q8_apack_replay import bounded_process
        tools=Path(target.__file__).resolve().parent
        for oversized in (False,True):
            with self.subTest(oversized=oversized),tempfile.TemporaryDirectory()as d:
                root=Path(d);stage=root/'stage.py';report=root/'inner.json'
                payload="print('ok')"if not oversized else "import os;os.write(1,b'x'*4096)"
                stage.write_text('import json,os,sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(tools))+')\n'
                    'from remote_thread_cpu_replay import inherited_process\n'
                    'row=inherited_process([sys.executable,"-c",'+repr(payload)+'],Path('+repr(str(root/'inner.out'))+'),Path('+repr(str(root/'inner.err'))+'),cwd=Path('+repr(str(root))+'),env=dict(os.environ),timeout=5,limit=1024)\n'
                    'Path('+repr(str(report))+').write_text(json.dumps(row))\n')
                args=([sys.executable,'-B',str(stage)],root/'outer.out',root/'outer.err')
                kwargs={'cwd':root,'env':{**os.environ,'GEN2_REMOTE_OWNED_STAGE':'1','PYTHONDONTWRITEBYTECODE':'1'},'timeout':10}
                if oversized:
                    with self.assertRaises(ValueError):bounded_process(*args,**kwargs)
                    row=json.loads((root/'inner.out.process.json').read_bytes())
                    self.assertTrue(row['leader_reaped']);self.assertFalse(row['completed'])
                    self.assertLessEqual(row['stdout_bytes'],1024)
                else:
                    outer=bounded_process(*args,**kwargs);row=json.loads(report.read_bytes())
                    self.assertTrue(outer['completed']and outer['leader_reaped']and outer['group_gone'])
                    self.assertTrue(row['leader_reaped']and row['completed'])
                    self.assertEqual(row['stage_group'],outer['pid'])
                    self.assertEqual((root/'inner.out').read_bytes(),b'ok\n')

    def nested_stage(self,mode):
        from remote_q8_apack_replay import bounded_process
        tools=Path(target.__file__).resolve().parent
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);pids=root/'pids.json';stage=root/'stage.py'
            payload=("import json,os,subprocess,sys,time\nchild=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\n"
                "with open(sys.argv[1]+'.tmp','w')as f:json.dump({'stage':os.getpgrp(),'inner':os.getpid(),'grandchild':child.pid,'grandchild_group':os.getpgid(child.pid)},f)\nos.replace(sys.argv[1]+'.tmp',sys.argv[1])\ntime.sleep(30)\n")
            inner_timeout=1 if mode=='inner' else 30
            stage.write_text('import os,sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(tools))+')\n'
                'from remote_thread_cpu_replay import inherited_process\n'
                'inherited_process([sys.executable,"-c",'+repr(payload)+','+repr(str(pids))+'],Path('+repr(str(root/'inner.out'))+'),Path('+repr(str(root/'inner.err'))+'),cwd=Path('+repr(str(root))+'),env=dict(os.environ),timeout='+str(inner_timeout)+')\n')
            delivered=[];timer=None;previous=None
            if mode=='cancel':
                previous=signal.signal(signal.SIGTERM,lambda signum,frame:delivered.append(signum))
                def cancel_ready():
                    until=time.monotonic()+5
                    while not pids.exists()and time.monotonic()<until:time.sleep(.01)
                    if pids.exists():os.kill(os.getpid(),signal.SIGTERM)
                timer=threading.Thread(target=cancel_ready);timer.start()
            try:
                # macOS can reject the final killpg(0) probe while a killed
                # orphan group disappears. That remains a reported failure;
                # independently require every recorded owned PID to vanish.
                with self.assertRaises((TimeoutError,ValueError,InterruptedError,PermissionError)):
                    bounded_process([sys.executable,'-B',str(stage)],root/'outer.out',root/'outer.err',
                        cwd=root,env={**os.environ,'GEN2_REMOTE_OWNED_STAGE':'1','PYTHONDONTWRITEBYTECODE':'1'},
                        timeout=1.5 if mode=='outer' else 10)
            finally:
                if timer is not None:timer.join(timeout=6)
                if previous is not None:signal.signal(signal.SIGTERM,previous)
            values=json.loads(pids.read_bytes());outer=json.loads((root/'outer.out.process.json').read_bytes())
            self.assertTrue(outer['leader_reaped'])
            self.assertEqual(values['stage'],outer['pid']);self.assertEqual(values['grandchild_group'],outer['pid'])
            if mode=='cancel':self.assertEqual(delivered,[signal.SIGTERM])
            def alive(pid):
                try:os.kill(pid,0);return True
                except ProcessLookupError:return False
            deadline=time.monotonic()+3
            tracked=[values['stage'],values['inner'],values['grandchild']]
            while any(alive(pid)for pid in tracked)and time.monotonic()<deadline:time.sleep(.01)
            survivors=[pid for pid in tracked if alive(pid)]
            try:self.assertEqual(survivors,[],'owned nested child survived '+mode)
            finally:
                for pid in survivors:
                    try:os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass

    @unittest.skipUnless(os.name=='posix','real nested stage timeout')
    def test_outer_timeout_reaps_inherited_nested_children(self):self.nested_stage('outer')

    @unittest.skipUnless(os.name=='posix','real inner deadline')
    def test_inner_timeout_reaps_inherited_nested_children(self):self.nested_stage('inner')

    @unittest.skipUnless(os.name=='posix','real nested stage cancellation')
    def test_outer_cancellation_reaps_inherited_nested_children(self):self.nested_stage('cancel')


if __name__=='__main__':unittest.main()
