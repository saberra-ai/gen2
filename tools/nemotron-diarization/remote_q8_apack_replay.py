"""Replay the fixed Windows artifact on a fresh Linux receiver; no kernels run."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time

SOURCE = '3fd41caa862c34134d008d18b0b6b830b736c864'
RUN, JOB, FEATURE, ARTIFACT = 39077373013, 117527216794, 39077372958, 12117422648
ARCHIVE = {'bytes': 752404322, 'sha256': 'f036e864269973d71dc069f878f986cca07ce788ca9b26317c32aadc57607269'}
OLD_REPO = '/Users/victor/workspace/gen2-nemotron3-n0'
OLD_CACHE = '/Users/victor/.cache/gen2-nemotron3-n0'
PARENT = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-apack-admission-preparation'
RESULTS = tuple('benches/results/nemotron-diarization/windows-x86_64/' + name for name in
    ('2026-10-17-q8-apack', '2026-10-17-q8-apack-loop-inspection'))


def descriptor(path):
    h=hashlib.sha256(); size=0
    with path.open('rb') as stream:
        while block:=stream.read(1024**2): size+=len(block); h.update(block)
    return {'bytes':size,'sha256':h.hexdigest()}


def relocate(raw, repo, cache):
    """Change only literal path prefixes; preserve the complete Python AST."""
    class Paths(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str):
                return ast.copy_location(ast.Constant(node.value.replace(OLD_REPO,str(repo)).replace(OLD_CACHE,str(cache))),node)
            return node
    tree=ast.parse(raw)
    changed=Paths().visit(tree)
    output=(ast.unparse(ast.fix_missing_locations(changed))+'\n').encode()
    if OLD_REPO.encode() in output or OLD_CACHE.encode() in output:
        raise ValueError('unrelocated path literal remains')
    if ast.dump(ast.parse(output),include_attributes=False)!=ast.dump(changed,include_attributes=False):
        raise ValueError('receiver AST changed beyond path literals')
    return output


def raw_api_transport(raw):
    """Opt in to faithful bytes at exactly the fetcher's two gh API lists."""
    tree=ast.parse(raw);changed=0
    for node in ast.walk(tree):
        if (isinstance(node,ast.List)and len(node.elts)>=2
                and all(isinstance(x,ast.Constant)for x in node.elts[:2])
                and [x.value for x in node.elts[:2]]==['gh','api']):
            if len(node.elts)!=3:raise ValueError('unexpected fetcher API command shape')
            node.elts.insert(2,ast.Constant('--allow-escape-sequences'));changed+=1
    if changed!=2:raise ValueError('exactly two original fetcher API commands required')
    output=(ast.unparse(ast.fix_missing_locations(tree))+'\n').encode()
    return output


def bounded_process(command, stdout_path, stderr_path, *, cwd, env, timeout=900, limit=8*1024**2):
    """Bound both output streams and own the complete POSIX process group."""
    if (os.name!='posix' or threading.current_thread()is not threading.main_thread()
            or timeout<=0 or not 0<limit<=128*1024**2):
        raise ValueError('positive bounded POSIX command required')
    report={'command':list(map(str,command)),'cwd':str(cwd),'stdout_bytes':0,'stderr_bytes':0,
        'leader_reaped':False,'group_gone':False,'completed':False}
    process=None; selector=selectors.DefaultSelector(); started=time.monotonic()
    pending=[]; previous={}; wake_read=None; wake_write=None
    def defer(signum,frame):
        pending.append(signum)
        # Wake selector.select despite Python's automatic interrupted-call retry.
        try: os.write(wake_write,b'!')
        except (BlockingIOError,OSError,TypeError): pass
    def interrupted():
        if pending: raise InterruptedError('receiver interrupted by signal '+str(pending[0]))
    try:
        wake_read,wake_write=os.pipe()
        os.set_blocking(wake_read,False);os.set_blocking(wake_write,False)
        selector.register(wake_read,selectors.EVENT_READ,None)
        for signum in (signal.SIGINT,signal.SIGTERM):
            previous[signum]=signal.signal(signum,defer)
        with stdout_path.open('xb') as out,stderr_path.open('xb') as err:
            interrupted()
            process=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0,start_new_session=True)
            report['pid']=process.pid
            # Popen can deliver Python signals after creating the child but
            # before returning its handle. Defer raising until ownership exists.
            interrupted()
            selector.register(process.stdout,selectors.EVENT_READ,('stdout',out,limit))
            selector.register(process.stderr,selectors.EVENT_READ,('stderr',err,1024**2))
            deadline=started+timeout
            while len(selector.get_map())>1:
                interrupted()
                remaining=deadline-time.monotonic()
                if remaining<=0: raise TimeoutError('owned receiver command deadline')
                events=selector.select(remaining)
                if not events: raise TimeoutError('owned receiver stream deadline')
                for key,_ in events:
                    if key.data is None:
                        os.read(wake_read,4096);interrupted();continue
                    name,dest,cap=key.data; field=name+'_bytes'
                    block=os.read(key.fd,min(1024**2,cap-report[field]+1))
                    if not block:
                        selector.unregister(key.fileobj);key.fileobj.close();continue
                    if report[field]+len(block)>cap: raise ValueError('receiver output cap: '+name)
                    dest.write(block);report[field]+=len(block)
            while True:
                interrupted()
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('owned receiver exit deadline')
                try: report['returncode']=process.wait(timeout=min(.1,remaining));break
                except subprocess.TimeoutExpired:pass
            report['leader_reaped']=True
            if report['returncode']!=0: raise ValueError('receiver command failed')
            try: os.killpg(process.pid,0)
            except ProcessLookupError: report['group_gone']=True
            if not report['group_gone']: raise ValueError('background process remains after receiver exit')
            report['completed']=True
    except BaseException as error:
        report['error']=repr(error)
        raise
    finally:
        try:
            try:
                if process is not None:
                    if not report['group_gone']:
                        try: os.killpg(process.pid,signal.SIGKILL)
                        except ProcessLookupError: pass
                    report['returncode']=process.wait(timeout=10);report['leader_reaped']=True
                    for stream in (process.stdout,process.stderr):
                        if not stream.closed: stream.close()
                    try: os.killpg(process.pid,0)
                    except ProcessLookupError: report['group_gone']=True
            finally:
                selector.close();report['elapsed_seconds']=time.monotonic()-started
                with stdout_path.with_suffix(stdout_path.suffix+'.process.json').open('x')as stream:
                    json.dump(report,stream,indent=2);stream.write('\n')
        finally:
            # Keep signals deferred through cleanup and report creation, even
            # when those fail. Restore both prior handlers before redelivery.
            try:
                for fd in (wake_read,wake_write):
                    if fd is not None:os.close(fd)
            finally:
                try:
                    if signal.SIGINT in previous:signal.signal(signal.SIGINT,previous[signal.SIGINT])
                finally:
                    if signal.SIGTERM in previous:signal.signal(signal.SIGTERM,previous[signal.SIGTERM])
            for signum in dict.fromkeys(pending):signal.raise_signal(signum)
    return report


def admit_api(run,job,artifacts):
    if (run['id']!=RUN or run['head_sha']!=SOURCE or run['repository']['full_name']!='saberra-ai/gen2'
            or run['status']!='completed' or run['conclusion']!='success'):
        raise ValueError('exact terminal native producer required')
    if (job['id']!=JOB or job['run_id']!=RUN or job['head_sha']!=SOURCE
            or job['status']!='completed' or job['conclusion']!='success'):
        raise ValueError('exact native job required')
    artifact,=artifacts['artifacts']
    if (artifact['id']!=ARTIFACT or artifact['name']!='nemotron-windows-q8-apack' or artifact['expired']is not False
            or artifact['workflow_run']['id']!=RUN or artifact['workflow_run']['head_sha']!=SOURCE
            or artifact['size_in_bytes']!=ARCHIVE['bytes'] or artifact['digest']!='sha256:'+ARCHIVE['sha256']):
        raise ValueError('exact frozen archive API identity required')


def verify_collection(root):
    raw=(root/'collection.json').read_bytes(); manifest=json.loads(raw)['files']
    observed=set(); folded=set(); total=0
    for path in root.rglob('*'):
        if path.is_symlink(): raise ValueError('symlink in result')
        if path.is_dir(): continue
        name=path.relative_to(root).as_posix()
        if not path.is_file() or path.stat().st_nlink!=1 or name.casefold()in folded:
            raise ValueError('regular unique result members required')
        folded.add(name.casefold());observed.add(name)
        if name=='collection.json':continue
        if name not in manifest or descriptor(path)!=manifest[name]:raise ValueError('result member changed')
        total+=manifest[name]['bytes']
    if observed!=set(manifest)|{'collection.json'} or len(observed)>10000 or total>1024**3:
        raise ValueError('closed bounded result required')
    return {'members':len(observed),'bytes_without_manifest':total,'manifest':descriptor(root/'collection.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if sys.platform!='linux':parser.error('fresh Linux CI receiver required')
    tools=Path(__file__).resolve().parent;repo=tools.parents[1]
    cache=args.cache.resolve();output=args.output.resolve()
    if (cache.exists() or output.exists() or cache.is_relative_to(repo) or output.is_relative_to(repo)
            or cache.is_relative_to(output) or output.is_relative_to(cache)):
        parser.error('fresh disjoint external cache/output required')
    output.mkdir()
    free=shutil.disk_usage(cache.parent).free
    with(output/'storage-admission.json').open('x')as f:
        json.dump({'free_bytes':free,'minimum_start_bytes':8*1024**3,'admitted':free>=8*1024**3,
            'archive_consumed':False,'new_compiler_sdk_model_calls':0},f,indent=2);f.write('\n')
    if free<8*1024**3:parser.error('original 8GiB start floor unmet')
    protocol_path=tools/'remote-q8-apack-replay.protocol.json'
    raw_protocol=protocol_path.read_bytes(); protocol=json.loads(raw_protocol)
    if (protocol['schema_version']!=2 or protocol['native_source']!=SOURCE or protocol['native_run_id']!=RUN
            or protocol['native_job_id']!=JOB or protocol['native_feature_run_id']!=FEATURE
            or protocol['artifact_id']!=ARTIFACT or protocol['archive']!=ARCHIVE
            or protocol['original_start_floor_bytes']!=8*1024**3 or protocol['archive_cap_bytes']!=1024**3
            or protocol['bundle_cap_bytes']!=128*1024**2
            or protocol['raw_api_transport']!={'flag':'--allow-escape-sequences','when':'advertised-by-gh-api-help',
                'exact_fetcher_command_lists':2,'raw_bytes_only':True}):raise ValueError('exact unchanged replay protocol required')
    native_path=tools/'windows-q8-apack.protocol.json'
    if descriptor(native_path)!=protocol['native_protocol']:raise ValueError('native protocol changed')
    native=json.loads(native_path.read_bytes())
    for group,base in [('tools',tools),('repository_files',repo)]:
        for name,wanted in native[group].items():
            if descriptor(base/name)!=wanted:raise ValueError('native frozen input changed: '+name)
    for name,wanted in protocol['files'].items():
        if descriptor(repo/name)!=wanted:raise ValueError('receiver frozen input changed: '+name)
    cache.mkdir();logs=output/'commands';logs.mkdir()
    environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1'}
    commands=[]
    def run(name,command,timeout=900,limit=8*1024**2):
        row=bounded_process(list(map(str,command)),logs/(name+'.stdout.log'),logs/(name+'.stderr.log'),
            cwd=repo,env=environment,timeout=timeout,limit=limit)
        commands.append(row)
        with(output/'commands.json').open('w')as f:json.dump(commands,f,indent=2);f.write('\n')
        return (logs/(name+'.stdout.log')).read_bytes()
    base=run('base',['git','rev-parse','HEAD']).decode().strip()
    if base!=os.environ.get('GITHUB_SHA'):raise ValueError('committed CI verifier source required')
    if run('clean',['git','status','--porcelain']).strip():raise ValueError('clean verifier tree required')
    if run('committed-protocol',['git','show',base+':tools/nemotron-diarization/'+protocol_path.name])!=raw_protocol:
        raise ValueError('committed receiver protocol required')
    for name in protocol['files']:
        if run('blob-'+hashlib.sha256(name.encode()).hexdigest()[:16],['git','show',base+':'+name])!=(repo/name).read_bytes():
            raise ValueError('uncommitted receiver source: '+name)
    run('native-git-source',['git','cat-file','-e',SOURCE+'^{commit}'])
    parent=repo/PARENT
    run('gh-version',['gh','version'])
    raw_api_flag_supported=b'--allow-escape-sequences'in run('gh-api-help',['gh','api','--help'])
    if descriptor(parent/'collection.json')!=protocol['parent_manifest']:raise ValueError('reviewed preparation changed')
    for name,wanted in json.loads((parent/'collection.json').read_bytes())['files'].items():
        if descriptor(parent/name)!=wanted:raise ValueError('preparation member changed')
        if name.startswith('prepared/') and name.endswith('.py'):
            (cache/Path(name).name).write_bytes((parent/name).read_bytes())
    executed=['fetch-windows-q8-apack-artifact-v1.py','admit-windows-q8-apack-delivery-v1.py',
        'replay-windows-q8-apack-supervision-v1.py','collect-windows-q8-apack-v1.py',
        'publish-windows-q8-apack-v1.py','test-q8-apack-receiver-controls-v1.py',
        'inspect-q8-apack-loops-v1.py','publish-q8-apack-static-v1.py']
    for name in executed:(cache/name).write_bytes(relocate((cache/name).read_bytes(),repo,cache))
    if raw_api_flag_supported:
        fetcher=cache/executed[0];fetcher.write_bytes(raw_api_transport(fetcher.read_bytes()))
    (cache/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (cache/protocol_path.name).write_bytes(raw_protocol)
    def dump(path,value):
        with path.open('w')as f:json.dump(value,f,indent=2);f.write('\n')
    preparation={'source_commit':SOURCE,'run_id':RUN,'job_id':JOB,'verifier_source_commit':base,
        'verifier_run_id':os.environ['GITHUB_RUN_ID'],'preparer':descriptor(Path(__file__)),
        'parent_manifest':protocol['parent_manifest'],'outputs':{n:descriptor(cache/n)for n in executed},
        'raw_api_transport':{'flag_supported':raw_api_flag_supported,'changed_command_lists':2 if raw_api_flag_supported else 0,
            'scope':'Only raw log/ZIP transport opt-in; complete bytes, all original admission gates retained'},
        'receivers_invoked':False,'artifact_consumed':False,'new_compiler_sdk_model_calls':0,
        'scope':'Literal path prefixes relocated; two raw-transport arguments added only if advertised by CLI help. Original validation AST/gates retained. Original Mac preparations preserved in Git.'}
    dump(cache/'windows-q8-apack-receivers-preparation-v1.json',preparation)
    python=[sys.executable,'-B']
    run('receiver-controls',python+[cache/'test-q8-apack-receiver-controls-v1.py'])
    run('publication-controls',python+[cache/'test-q8-apack-publication-v1.py','--publisher',
        cache/'publish-windows-q8-apack-v1.py','--report',cache/'q8-apack-publication-green-v1.json'])
    run('static-publication-controls',python+[cache/'test-q8-scale-static-publication-v1.py','--publisher',
        cache/'publish-q8-apack-static-v1.py','--report',cache/'q8-apack-static-publication-green-v1.json'])
    # The original publisher retains these conventional log paths as well.
    for stem,command in [('q8-apack-receiver-controls-local-v1','receiver-controls'),
            ('q8-apack-publication-green-v1','publication-controls'),
            ('q8-apack-static-publication-green-v1','static-publication-controls'),
            ('raw-api-cli-version','gh-version'),('raw-api-cli-help','gh-api-help')]:
        for channel in ['stdout','stderr']:
            shutil.copyfile(logs/(command+'.'+channel+'.log'),cache/(stem+'.'+channel+'.log'))
    receiver_names=executed[:6]+['prepare-windows-q8-apack-receivers-v1.py',
        'windows-q8-apack-receivers-preparation-v1.json','q8-apack-receiver-controls-local-v1.json',
        'q8-apack-receiver-controls-local-v1.stdout.log','q8-apack-receiver-controls-local-v1.stderr.log']
    dump(cache/'windows-q8-apack-receivers-pre-execution-v1.json',{'files':{n:descriptor(cache/n)for n in receiver_names},
        'source_commit':SOURCE,'run_id':RUN,'job_id':JOB,'artifact_consumed':False,'receivers_invoked':False})
    inspector_files=[cache/'inspect-q8-apack-loops-v1.py',cache/'inspect-q8-row-loops-v1.py',
        cache/'windows-q8-apack-receivers-pre-execution-v1.json']
    dump(cache/'q8-apack-loop-inspector-pre-execution-v1.json',{'files':{str(p):descriptor(p)for p in inspector_files},
        'parent_parser_ast_equal':True,'inspector_invoked':False,'new_native_calls':0})
    publication_names=receiver_names+['windows-q8-apack-receivers-pre-execution-v1.json',
        'test-q8-apack-publication-v1.py','q8-apack-publication-green-v1.json',
        'q8-apack-publication-green-v1.stdout.log','q8-apack-publication-green-v1.stderr.log',
        'inspect-q8-apack-loops-v1.py','inspect-q8-row-loops-v1.py','q8-apack-loop-inspector-pre-execution-v1.json',
        'publish-q8-apack-static-v1.py','test-q8-scale-static-publication-v1.py',
        'q8-apack-static-publication-green-v1.json','q8-apack-static-publication-green-v1.stdout.log',
        'q8-apack-static-publication-green-v1.stderr.log','raw-api-cli-version.stdout.log',
        'raw-api-cli-version.stderr.log','raw-api-cli-help.stdout.log','raw-api-cli-help.stderr.log',
        Path(__file__).name,protocol_path.name]
    dump(cache/'windows-q8-apack-publication-pre-execution-v1.json',{'files':{str(cache/n):descriptor(cache/n)for n in publication_names},
        'source_commit':SOURCE,'run_id':RUN,'job_id':JOB,'artifact_consumed':False,'publisher_invoked':False})
    api={}
    for key,endpoint in [('run',f'runs/{RUN}'),('job',f'jobs/{JOB}'),('artifacts',f'runs/{RUN}/artifacts')]:
        api[key]=json.loads(run('api-'+key,['gh','api','repos/saberra-ai/gen2/actions/'+endpoint]))
    admit_api(api['run'],api['job'],api['artifacts'])
    data=cache/'windows-q8-apack-ci-v1';data.mkdir()
    for name,script in [('fetch','fetch-windows-q8-apack-artifact-v1.py'),('delivery','admit-windows-q8-apack-delivery-v1.py'),
            ('supervision','replay-windows-q8-apack-supervision-v1.py'),('collection','collect-windows-q8-apack-v1.py'),
            ('publication','publish-windows-q8-apack-v1.py')]:
        run(name,python+[cache/script])
        for channel in ['stdout','stderr']:shutil.copyfile(logs/(name+'.'+channel+'.log'),data/(name+'.'+channel+'.log'))
        if name=='fetch' and descriptor(data/'artifact.zip')!=ARCHIVE:raise ValueError('frozen archive digest mismatch')
    run('inspection',python+[cache/'inspect-q8-apack-loops-v1.py'])
    for endpoint,name in [(f'runs/{FEATURE}','q8-apack-feature-run-terminal-v1.json'),
            (f'runs/{FEATURE}/jobs?per_page=100','q8-apack-feature-terminal-v1.json')]:
        raw=run(name,['gh','api','repos/saberra-ai/gen2/actions/'+endpoint]);(cache/name).write_bytes(raw)
    run('static-publication',python+[cache/'publish-q8-apack-static-v1.py'])
    collections={name:verify_collection(repo/name)for name in RESULTS}
    run('stage-results',['git','add','--',*RESULTS])
    staged=run('staged-paths',['git','diff','--cached','--name-only','-z']).decode().strip('\0').split('\0')
    if not staged or any(not any(p.startswith(root+'/')for root in RESULTS)for p in staged):
        raise ValueError('only new result directories may be bundled')
    run('commit-results',['git','-c','user.name=Gen2 evidence receiver','-c','user.email=gen2-evidence@invalid',
        '-c','gc.auto=0','-c','core.hooksPath=/dev/null','commit','-q','-m','Retain independently replayed A-packing evidence'])
    result=run('result-commit',['git','rev-parse','HEAD']).decode().strip()
    if run('result-parent',['git','rev-parse','HEAD^']).decode().strip()!=base:raise ValueError('one result-only commit required')
    bundle=output/'receiver-result.bundle'
    bounded_process(['git','bundle','create','-','HEAD','^'+base],bundle,output/'bundle.stderr.log',
        cwd=repo,env=environment,timeout=600,limit=128*1024**2)
    run('verify-bundle',['git','bundle','verify',bundle])
    if not 0<descriptor(bundle)['bytes']<=128*1024**2:raise ValueError('bounded nonempty bundle required')
    dump(output/'bundle.json',{'verifier_source_commit':base,'verifier_run_id':os.environ['GITHUB_RUN_ID'],
        'native_source_commit':SOURCE,'native_run_id':RUN,'native_job_id':JOB,'native_archive':ARCHIVE,
        'result_commit':result,'result_parent':base,'result_roots':list(RESULTS),'collections':collections,
        'bundle':descriptor(bundle),'original_start_floor_bytes':8*1024**3,'new_compiler_sdk_model_calls':0,
        'production_or_release_qualified':False,
        'scope':'Full unchanged artifact replay on independent Linux CI host; no kernel/model rerun. Git bundle contains result-only commit relative to verifier source. Local import/replay/review remains required. Raw native artifact remains in original GitHub artifact; its complete archive is not transferred via this bundle.'})
    print(json.dumps({'bundle':descriptor(bundle),'result_commit':result,'collections':collections,'new_model_calls':0}))


if __name__=='__main__':main()
