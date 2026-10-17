"""AST-isolated real-child spawn-window cleanup controls; no fixture/guest call."""
import ast,json,os,signal,subprocess,tempfile,time
from pathlib import Path
source=Path('/Users/victor/.cache/gen2-nemotron3-n0/run-macos-search-bit-control-v2.py')
tree=ast.parse(source.read_text())
functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'interrupted','run'}],type_ignores=[])
results=[]
for signum in [signal.SIGTERM,signal.SIGINT]:
    with tempfile.TemporaryDirectory() as temporary:
        out=Path(temporary); (out/'offline.sb').write_text('(version 1)(allow default)')
        owned=[]; actual=subprocess.Popen
        class Inject:
            @staticmethod
            def Popen(*args,**kwargs):
                process=actual(*args,**kwargs);owned.append(process)
                os.kill(os.getpid(),signum)
                return process
        q={'commands':[]}
        ns={'json':json,'os':os,'signal':signal,'subprocess':Inject,'time':time,'out':out,'p':{'timeout_seconds':3},'environment':{'PATH':'/usr/bin:/bin'},'q':q,'frozen':lambda:None,'persist':lambda:None,'spawning':False,'pending_signals':[]}
        exec(compile(functions,str(source),'exec'),ns)
        old={n:signal.signal(n,ns['interrupted']) for n in [signal.SIGTERM,signal.SIGINT]}
        try:
            try:ns['run']('spawn-window',['/usr/bin/python3','-B','-c','import time;time.sleep(20)'])
            except InterruptedError:pass
            else:raise AssertionError('deferred signal missing')
            row=q['commands'][0]
            assert len(owned)==1 and row['pid']==owned[0].pid
            assert row['reaped'] is True and row['returncode']==-signal.SIGKILL and owned[0].poll()==-signal.SIGKILL
            assert row['forced_process_group_termination'] is True
            results.append({'signal':int(signum),'pid':row['pid'],'returncode':row['returncode'],'reaped':True})
        finally:
            for process in owned:
                if process.poll() is None:process.kill()
                process.wait()
            for n,handler in old.items():signal.signal(n,handler)
print(json.dumps({'controls':results,'passed':True,'new_vm_operations':0,'host_fixture_invocations':0}))
