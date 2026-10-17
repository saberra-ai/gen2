"""Execute a frozen host-only fixture control; no guest resources admitted."""
import hashlib,json,os,platform,signal,subprocess,sys,time
from pathlib import Path
spawning=False
pending_signals=[]
def interrupted(number,frame):
    if spawning:
        pending_signals.append(number)
        return
    raise InterruptedError('host controller signal '+str(number))
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
tools=repo/'tools/nemotron-diarization'
protocol=tools/'macos-search-bit-control-v2.protocol.json'
raw=protocol.read_bytes(); p=json.loads(raw)
def descriptor(path):
    raw=Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def frozen():
    if protocol.read_bytes()!=raw: raise ValueError('protocol changed')
    for name,want in p['tools'].items():
        if descriptor(tools/name)!=want: raise ValueError('source changed: '+name)
    for name,want in p['external_files'].items():
        if descriptor(name)!=want: raise ValueError('interpreter changed: '+name)
frozen()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
for name,want in {'macos-search-bit-control-v2.protocol.json':descriptor(protocol),**p['tools']}.items():
    data=subprocess.check_output(['git','show',commit+':tools/nemotron-diarization/'+name],cwd=repo,env={**os.environ,'GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1'})
    if {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}!=want: raise ValueError('committed inputs required')
if platform.system()!=p['expected_host']['system'] or platform.machine()!=p['expected_host']['machine'] or os.geteuid()!=p['expected_host']['effective_uid']: raise ValueError('expected actual unprivileged host required')
cpu=subprocess.check_output(['/usr/sbin/sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
if cpu!=p['expected_host']['cpu']: raise ValueError('expected CPU required')
out=Path(p['output']); out.mkdir(mode=0o700)
(out/'protocol.json').write_bytes(raw)
(out/'runner.py').write_bytes(Path(__file__).read_bytes())
for name in p['tools']: (out/name).write_bytes((tools/name).read_bytes())
quote=lambda value:json.dumps(str(value))
policy='(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n(allow file-write* (subpath '+quote(out/'fixture')+'))\n(deny file-read* (subpath '+quote('/Users/victor/workspace/pnola')+') (subpath '+quote('/Users/victor/.cache/gen2-nemotron3-n0/macos-vm-boot-local-v1')+'))\n(deny file-read* (regex #".*\\.(nemo|gguf|safetensors|onnx|npy|f32)$"))\n'
(out/'offline.sb').write_text(policy)
(out/'denied.gguf').write_bytes(b'not a model')
q={'source_commit':commit,'scope':p['scope'],'host':{'cpu':cpu,'platform':platform.platform(),'euid':os.geteuid()},'commands':[],'host_control_qualified':False,'guest_access_qualified':False,'new_guest_operations':0,'storage':{'available_bytes':os.statvfs(out).f_bavail*os.statvfs(out).f_frsize,'vm_initial_required_bytes':120*1024**3,'vm_running_required_bytes':80*1024**3},'pins':{'protocol':descriptor(protocol),'runner':descriptor(__file__)}}
environment={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':str(out),'TMPDIR':str(out)+'/', 'LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'}
def persist(): (out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n')
def run(name,command):
    global spawning
    frozen()
    row={'name':name,'argv':['/usr/bin/sandbox-exec','-f',str(out/'offline.sb'),*command],'cwd':str(out),'environment':environment}
    q['commands'].append(row); persist(); process=None
    began=time.monotonic()
    try:
        with (out/(name+'.stdout.log')).open('xb') as stdout,(out/(name+'.stderr.log')).open('xb') as stderr:
            spawning=True
            try:
                process=subprocess.Popen(row['argv'],cwd=out,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
            finally:
                spawning=False
            row['pid']=process.pid
            if pending_signals:
                raise InterruptedError('deferred spawn signal '+str(pending_signals.pop(0)))
            persist()
            try: row['returncode']=process.wait(timeout=p['timeout_seconds'])
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGKILL)
                    row['forced_process_group_termination']=True
                row['returncode']=process.wait()
                raise
        row['reaped']=True
        row['wall_seconds']=time.monotonic()-began; persist()
        if row['returncode']!=0: raise ValueError(name+' failed')
    except BaseException as error:
        if process is not None:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGKILL)
                row['forced_process_group_termination']=True
            row['returncode']=process.wait(); row['reaped']=True
        row['error']=repr(error); persist(); raise
negative="import os,socket,sys\nfor name in sys.argv[1:]:\n try: fd=os.open(name,os.O_WRONLY|os.O_CREAT,0o600) if not name.endswith('.gguf') else os.open(name,os.O_RDONLY)\n except PermissionError: pass\n else: os.close(fd); raise RuntimeError('sandbox boundary failed')\ns=socket.socket()\ntry:s.connect(('127.0.0.1',9))\nexcept PermissionError:pass\nelse:raise RuntimeError('network allowed')\ns.close()\n"
try:
    run('offline-controls',['/usr/bin/python3','-B','-c',negative,str(out/'denied-write.bin'),str(out/'denied.gguf')])
    run('host-control',p['argv'])
    report=json.loads((out/'host-control.stdout.log').read_text())
    for name,want in p['required'].items():
        if report.get(name) is not want: raise ValueError('required observation false: '+name)
    if report['runtime']!='/Applications/Xcode.app/Contents/Developer/usr/bin/python3' or report['python_version']!=p['expected_host']['python_version']: raise ValueError('pinned actual runtime required')
    if report['before']['permissions']!=0o600 or report['during']['permissions']!=0o700 or report['restored']['permissions']!=0o600 or report['denials_before']!={'users':13,'groups':13} or set(report['children'])!={'users','groups'}: raise ValueError('full state transition required')
    if not all(report['security_'+phase]['status']=='ok' for phase in ['before','during','restored']): raise ValueError('actual security query required')
    q['report']=report
    frozen()
    q['host_control_qualified']=True
except BaseException as error:
    q['error']=repr(error)
    raise
finally:
    persist()
    names=['protocol.json','runner.py','offline.sb','denied.gguf','qualification.json',*p['tools'],*[name+'.'+kind+'.log' for name in ['offline-controls','host-control'] for kind in ['stdout','stderr'] if (out/(name+'.'+kind+'.log')).exists()]]
    (out/'collection.json').write_text(json.dumps({'files':{name:descriptor(out/name) for name in names}},indent=2)+'\n')
print(json.dumps({'output':str(out),'host_control_qualified':q['host_control_qualified'],'commands':len(q['commands'])}))
