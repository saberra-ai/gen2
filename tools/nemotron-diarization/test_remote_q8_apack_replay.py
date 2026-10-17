"""Real bounded children, literal-only relocation and exact artifact admission."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import shutil
import subprocess
import signal
import time
from unittest import mock

from remote_q8_apack_replay import (SOURCE,RUN,JOB,ARTIFACT,ARCHIVE,OLD_REPO,OLD_CACHE,
    relocate,bounded_process,admit_api,verify_collection,descriptor)
from remote_q8_apack_replay import PARENT
import remote_q8_apack_replay as replay


class RemoteReplayTest(unittest.TestCase):
    @unittest.skipUnless(os.name=='posix','real receiver preparation controls')
    def test_fresh_preparation_controls_and_retained_pins_before_api_access(self):
        # Execute the actual preparation stage through its first API boundary.
        # No network, artifact receiver or native command is reachable here.
        tree=ast.parse(Path(replay.__file__).read_bytes())
        body=next(node.body for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='main')
        def assignment(node,name):
            return isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in node.targets)
        start=next(i for i,node in enumerate(body)if assignment(node,'parent'))
        stop=next(i for i,node in enumerate(body)if assignment(node,'api'))
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);cache=root/'cache';cache.mkdir();logs=root/'logs';logs.mkdir()
            repo=root/'repo';repo.symlink_to(Path(replay.__file__).resolve().parents[2],target_is_directory=True)
            protocol_path=repo/'tools/nemotron-diarization/remote-q8-apack-replay.protocol.json'
            raw_protocol=protocol_path.read_bytes();protocol=json.loads(raw_protocol);commands=[]
            def run(name,command):
                commands.append(name)
                bounded_process(list(map(str,command)),logs/(name+'.stdout.log'),logs/(name+'.stderr.log'),
                    cwd=repo,env=dict(os.environ),timeout=30)
            env={**vars(replay),'repo':repo,'cache':cache,'logs':logs,'run':run,'base':'fixture',
                'protocol_path':protocol_path,'raw_protocol':raw_protocol,'protocol':protocol}
            with mock.patch.dict(os.environ,{'GITHUB_RUN_ID':'fixture'}):
                exec(compile(ast.Module(body=body[start:stop],type_ignores=[]),'actual-preparation','exec'),env)
            self.assertEqual(set(commands),{'receiver-controls','publication-controls','static-publication-controls'})
            pins=json.loads((cache/'windows-q8-apack-publication-pre-execution-v1.json').read_bytes())['files']
            for path,wanted in pins.items():self.assertEqual(descriptor(Path(path)),wanted)
            for name in ['q8-apack-loop-inspector-pre-execution-v1.json',
                    'q8-apack-static-publication-green-v1.json',
                    'q8-apack-static-publication-green-v1.stdout.log',
                    'q8-apack-static-publication-green-v1.stderr.log']:
                self.assertIn(str(cache/name),pins)

    def test_actual_eight_reviewed_receiver_asts_relocate(self):
        repo=Path(__file__).resolve().parents[2]
        parent=repo/PARENT/'prepared'
        for name in ['fetch-windows-q8-apack-artifact-v1.py','admit-windows-q8-apack-delivery-v1.py',
                'replay-windows-q8-apack-supervision-v1.py','collect-windows-q8-apack-v1.py',
                'publish-windows-q8-apack-v1.py','test-q8-apack-receiver-controls-v1.py',
                'inspect-q8-apack-loops-v1.py','publish-q8-apack-static-v1.py']:
            with self.subTest(name=name):
                raw=(parent/name).read_bytes();changed=relocate(raw,Path('/fresh/repo'),Path('/fresh/cache'))
                self.assertNotIn(OLD_REPO.encode(),changed);self.assertNotIn(OLD_CACHE.encode(),changed)
                self.assertEqual(relocate(changed,Path('/fresh/repo'),Path('/fresh/cache')),changed)

    def test_receiver_relocation_changes_only_literal_paths(self):
        raw=(f"from pathlib import Path\nroot=Path('{OLD_CACHE}/data')\nrepo=Path('{OLD_REPO}')\n"
            "def gate(x):\n    assert x>=8*1024**3\n    return ['control','apack','apack','control']\n").encode()
        changed=relocate(raw,Path('/receiver/repo'),Path('/receiver/cache'))
        env={};exec(changed,env)
        self.assertEqual(env['root'],Path('/receiver/cache/data'))
        self.assertEqual(env['repo'],Path('/receiver/repo'))
        with self.assertRaises(AssertionError):env['gate'](8*1024**3-1)
        self.assertEqual(env['gate'](8*1024**3),['control','apack','apack','control'])
        functions=lambda data:[ast.dump(n,include_attributes=False)for n in ast.parse(data).body if isinstance(n,ast.FunctionDef)]
        self.assertEqual(functions(raw),functions(changed))

    def metadata(self):
        run={'id':RUN,'head_sha':SOURCE,'repository':{'full_name':'saberra-ai/gen2'},'status':'completed','conclusion':'success'}
        job={'id':JOB,'run_id':RUN,'head_sha':SOURCE,'status':'completed','conclusion':'success'}
        artifact={'id':ARTIFACT,'name':'nemotron-windows-q8-apack','expired':False,
            'workflow_run':{'id':RUN,'head_sha':SOURCE},'size_in_bytes':ARCHIVE['bytes'],'digest':'sha256:'+ARCHIVE['sha256']}
        return run,job,{'artifacts':[artifact]}

    def test_changed_source_job_digest_size_status_or_artifact_rejects(self):
        admit_api(*self.metadata())
        for group,key,value in [(0,'head_sha','0'*40),(0,'status','in_progress'),(1,'id',JOB+1),
                (2,'digest','sha256:'+'0'*64),(2,'size_in_bytes',ARCHIVE['bytes']-1),(2,'expired',True),(2,'id',ARTIFACT+1)]:
            values=copy.deepcopy(self.metadata());target=values[group]if group!=2 else values[2]['artifacts'][0]
            target[key]=value
            with self.subTest(group=group,key=key),self.assertRaises(ValueError):admit_api(*values)

    @unittest.skipUnless(os.name=='posix','real POSIX process-group controls')
    def test_real_child_output_exit_and_reap(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            result=bounded_process([sys.executable,'-B','-c',"import sys;sys.stdout.write('ok');sys.stderr.write('err')"],
                root/'out',root/'err',cwd=root,env=dict(os.environ),timeout=10,limit=64)
            self.assertEqual((root/'out').read_bytes(),b'ok')
            self.assertEqual((root/'err').read_bytes(),b'err')
            self.assertTrue(result['completed']and result['leader_reaped']and result['group_gone'])

    @unittest.skipUnless(os.name=='posix','real POSIX process-group controls')
    def test_real_overflow_failure_and_deadline_leave_bounded_logs_and_reaped_leader(self):
        for script,timeout,limit,exception in [("import sys;sys.stdout.write('x'*1000)",10,32,ValueError),
                ("import sys;sys.stderr.write('x'*(1024**2+1))",10,32,ValueError),
                ('raise SystemExit(7)',10,32,ValueError),('import time;time.sleep(30)',.1,32,TimeoutError)]:
            with self.subTest(script=script),tempfile.TemporaryDirectory()as directory:
                root=Path(directory)
                with self.assertRaises(exception):bounded_process([sys.executable,'-B','-c',script],
                    root/'out',root/'err',cwd=root,env=dict(os.environ),timeout=timeout,limit=limit)
                report=json.loads((root/'out.process.json').read_bytes())
                self.assertTrue(report['leader_reaped']);self.assertFalse(report['completed'])
                self.assertLessEqual((root/'out').stat().st_size,limit)
                self.assertTrue(report['group_gone'])

    @unittest.skipUnless(os.name=='posix','real POSIX signals')
    def test_post_spawn_signal_is_deferred_until_child_is_owned(self):
        real_popen=subprocess.Popen
        for signum in (signal.SIGINT,signal.SIGTERM):
            with self.subTest(signal=signum),tempfile.TemporaryDirectory()as directory:
                root=Path(directory);delivered=[]
                handler=lambda sig,frame:delivered.append(sig)
                previous=signal.signal(signum,handler)
                others={sig:signal.getsignal(sig)for sig in (signal.SIGINT,signal.SIGTERM)}
                def spawn(*args,**kwargs):
                    child=real_popen(*args,**kwargs)
                    os.kill(os.getpid(),signum)
                    return child
                try:
                    with mock.patch('remote_q8_apack_replay.subprocess.Popen',side_effect=spawn):
                        with self.assertRaises(InterruptedError):
                            bounded_process([sys.executable,'-B','-c','import time;time.sleep(30)'],
                                root/'out',root/'err',cwd=root,env=dict(os.environ),timeout=10)
                    report=json.loads((root/'out.process.json').read_bytes())
                    self.assertTrue(report['leader_reaped']and report['group_gone'])
                    self.assertFalse(report['completed']);self.assertEqual(delivered,[signum])
                    for sig,wanted in others.items():self.assertIs(signal.getsignal(sig),wanted)
                finally:signal.signal(signum,previous)

    @unittest.skipUnless(os.name=='posix','real POSIX signals')
    def test_sigterm_controller_reaps_child_and_writes_report_before_termination(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);ready=root/'ready'
            child_code=f'from pathlib import Path;import time;Path({str(ready)!r}).write_text("ready");time.sleep(30)'
            code=('from remote_q8_apack_replay import bounded_process;from pathlib import Path;import os,sys;'
                f'root=Path({str(root)!r});bounded_process([sys.executable,"-B","-c",{child_code!r}],'
                'root/"out",root/"err",cwd=root,env=dict(os.environ),timeout=20)')
            env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parent)}
            controller=subprocess.Popen([sys.executable,'-B','-c',code],env=env,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                deadline=time.monotonic()+10
                while not ready.exists()and controller.poll()is None and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(ready.exists(),'child readiness required before signal')
                controller.send_signal(signal.SIGTERM)
                stdout,stderr=controller.communicate(timeout=15)
                self.assertEqual(controller.returncode,-signal.SIGTERM,(stdout,stderr))
                report=json.loads((root/'out.process.json').read_bytes())
                self.assertTrue(report['leader_reaped']and report['group_gone'])
                self.assertFalse(report['completed'])
                with self.assertRaises(ProcessLookupError):os.kill(report['pid'],0)
            finally:
                if controller.poll()is None:controller.kill()
                controller.communicate(timeout=15)

    @unittest.skipUnless(os.name=='posix','real POSIX signals')
    def test_handlers_restore_after_success_and_report_failure(self):
        previous={sig:signal.getsignal(sig)for sig in (signal.SIGINT,signal.SIGTERM)}
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            bounded_process([sys.executable,'-B','-c','pass'],root/'out',root/'err',
                cwd=root,env=dict(os.environ),timeout=10)
            # Deliberate report collision exercises the outer restoration finally.
            (root/'collision.process.json').write_text('preserve')
            with self.assertRaises(FileExistsError):
                bounded_process([sys.executable,'-B','-c','pass'],root/'collision',root/'collision.err',
                    cwd=root,env=dict(os.environ),timeout=10)
            for sig,wanted in previous.items():self.assertIs(signal.getsignal(sig),wanted)

    @unittest.skipUnless(os.name=='posix' and shutil.which('git'),'real Git bundle over bounded POSIX stdout')
    def test_result_bundle_transfers_one_commit_with_base_prerequisite(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);source=root/'source';source.mkdir();dest=root/'destination';dest.mkdir()
            def git(path,*args,check=True):
                return subprocess.run(['git','-C',str(path),'-c','user.name=Fixture',
                    '-c','user.email=fixture@invalid','-c','gc.auto=0','-c','core.hooksPath=/dev/null',*args],
                    check=check,capture_output=True,timeout=30)
            git(source,'init','-q','-b','main');(source/'base').write_text('existing')
            git(source,'add','base');git(source,'commit','-q','-m','base')
            base=git(source,'rev-parse','HEAD').stdout.decode().strip()
            (source/'result').write_text('complete receipt')
            git(source,'add','result');git(source,'commit','-q','-m','result')
            result=git(source,'rev-parse','HEAD').stdout.decode().strip()
            bundle=root/'result.bundle'
            observed=bounded_process(['git','bundle','create','-','HEAD','^'+base],bundle,root/'stderr',
                cwd=source,env=dict(os.environ),timeout=30,limit=128*1024)
            self.assertTrue(observed['completed']);git(dest,'init','-q','-b','main')
            self.assertNotEqual(git(dest,'bundle','verify',str(bundle),check=False).returncode,0)
            git(dest,'fetch','-q','--depth=1',str(source),base)
            git(dest,'bundle','verify',str(bundle));git(dest,'bundle','unbundle',str(bundle))
            self.assertEqual(git(dest,'show',result+':result').stdout,b'complete receipt')
            self.assertEqual(git(dest,'rev-parse',result+'^').stdout.decode().strip(),base)
            self.assertNotEqual(git(dest,'rev-parse','--verify','HEAD',check=False).returncode,0)

    def test_closed_result_mutation_extra_file_and_symlink_reject(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);data=root/'vector.f32';data.write_bytes(b'original')
            (root/'collection.json').write_text(json.dumps({'files':{data.name:descriptor(data)}}))
            self.assertEqual(verify_collection(root)['members'],2)
            data.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_collection(root)
            data.write_bytes(b'original');(root/'extra').write_bytes(b'extra')
            with self.assertRaises(ValueError):verify_collection(root)
            (root/'extra').unlink()
            if os.name=='posix':
                (root/'link').symlink_to(data)
                with self.assertRaises(ValueError):verify_collection(root)


if __name__=='__main__':unittest.main()
