"""Admit and replay one frozen Windows CPU-counter artifact on Linux; no native execution."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time

SOURCE='15c905d8815190a54f25452e7e51378c24f85914'
RUN,JOB,FEATURE=39083120058,117544022272,39083120118
ARTIFACT=12119423327
ARCHIVE={'bytes':24016727,'sha256':'27ed5316133a179a288e665d167c87a53437336fca51ed967666f844cc550618'}
NATIVE_PROTOCOL={'bytes':670257,'sha256':'e9809ffe025bbe127c27daa5f8dfa07916cab7fe24a40d074651103b8c3cfc66'}
HELPER={'bytes':22591,'sha256':'6b7f4c6d3a9a672ad0a89c790fa0d51608adad4c3667395c2da1329c68e48b08'}
PARENT='benches/results/nemotron-diarization/windows-x86_64/2026-10-17-thread-cpu-admission-preparation'
RESULT='benches/results/nemotron-diarization/windows-x86_64/2026-10-17-thread-cpu-profile'
OLD_REPO='/Users/victor/workspace/gen2-nemotron3-n0'
OLD_CACHE='/Users/victor/.cache/gen2-nemotron3-n0'
EXECUTED=('admit-windows-thread-cpu-profile-delivery-v1.py',
          'replay-windows-thread-cpu-profile-supervision-v1.py',
          'collect-windows-thread-cpu-profile-v1.py')


def descriptor(path):
    h=hashlib.sha256();n=0
    with path.open('rb')as f:
        while b:=f.read(1024**2):n+=len(b);h.update(b)
    return {'bytes':n,'sha256':h.hexdigest()}


def byte_descriptor(raw):
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def relocate(raw,repo,cache):
    """Only repository/cache string prefixes may change in prepared Python ASTs."""
    tree=ast.parse(raw)
    class Paths(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str):
                return ast.copy_location(ast.Constant(node.value.replace(OLD_REPO,str(repo)).replace(OLD_CACHE,str(cache))),node)
            return node
    tree=Paths().visit(tree)
    out=(ast.unparse(ast.fix_missing_locations(tree))+'\n').encode()
    if OLD_REPO.encode()in out or OLD_CACHE.encode()in out:
        raise ValueError('unrelocated prepared literal remains')
    if ast.dump(ast.parse(out),include_attributes=False)!=ast.dump(tree,include_attributes=False):
        raise ValueError('prepared AST changed beyond literal relocation')
    return out


def admit_api(run,job,artifacts):
    if (run['id']!=RUN or run['head_sha']!=SOURCE or run['repository']['full_name']!='saberra-ai/gen2'
            or run['status']!='completed' or run['conclusion']!='success'):
        raise ValueError('exact successful terminal native run required')
    if (job['id']!=JOB or job['run_id']!=RUN or job['head_sha']!=SOURCE
            or job['status']!='completed' or job['conclusion']!='success'):
        raise ValueError('exact successful native job required')
    if artifacts['total_count']!=1:raise ValueError('one complete native artifact required')
    a,=artifacts['artifacts']
    if (a['id']!=ARTIFACT or a['name']!='nemotron-windows-thread-cpu-profile' or a['expired']is not False
            or a['workflow_run']['id']!=RUN or a['workflow_run']['head_sha']!=SOURCE
            or a['size_in_bytes']!=ARCHIVE['bytes'] or a['digest']!='sha256:'+ARCHIVE['sha256']):
        raise ValueError('exact frozen native ZIP identity required')


def dump(path,value):
    with path.open('x')as f:json.dump(value,f,indent=2);f.write('\n')


def inherited_process(command,stdout_path,stderr_path,*,cwd,env,timeout=180,limit=8*1024**2):
    """Bound one stage child while its outer supervisor owns the entire process group."""
    if (os.name!='posix' or env.get('GEN2_REMOTE_OWNED_STAGE')!='1'
            or os.getpgrp()!=os.getpid()or timeout<=0 or not 0<limit<=8*1024**2):
        raise ValueError('registered outer stage group and bounded child required')
    started=time.monotonic();process=None;selector=selectors.DefaultSelector()
    row={'command':list(map(str,command)),'cwd':str(cwd),'stage_group':os.getpgrp(),
        'group_owner':'outer bounded stage supervisor','leader_reaped':False,'completed':False,
        'stdout_bytes':0,'stderr_bytes':0}
    try:
        with stdout_path.open('xb')as out,stderr_path.open('xb')as err:
            # Inherit the stage group. No nested session/group can escape the
            # outer timeout/cancellation owner's single kill-on-exit boundary.
            process=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
            row['pid']=process.pid
            selector.register(process.stdout,selectors.EVENT_READ,('stdout',out,limit))
            selector.register(process.stderr,selectors.EVENT_READ,('stderr',err,1024**2))
            deadline=started+timeout
            while selector.get_map():
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('inherited source command deadline')
                events=selector.select(remaining)
                if not events:raise TimeoutError('inherited source stream deadline')
                for key,_ in events:
                    name,destination,cap=key.data;field=name+'_bytes'
                    block=os.read(key.fd,min(1024**2,cap-row[field]+1))
                    if not block:
                        selector.unregister(key.fileobj);key.fileobj.close();continue
                    if row[field]+len(block)>cap:raise ValueError('inherited source output cap: '+name)
                    destination.write(block);row[field]+=len(block)
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('inherited source exit deadline')
            row['returncode']=process.wait(timeout=remaining);row['leader_reaped']=True
            if row['returncode']!=0:raise ValueError('inherited source command failed')
            row['completed']=True
    except BaseException as error:
        row['error']=repr(error);raise
    finally:
        try:
            if process is not None:
                if process.poll()is None:process.kill()
                row['returncode']=process.wait(timeout=10);row['leader_reaped']=True
        finally:
            selector.close()
            if process is not None:
                for stream in (process.stdout,process.stderr):
                    if stream is not None and not stream.closed:stream.close()
            row['elapsed_seconds']=time.monotonic()-started
            dump(stdout_path.with_suffix(stdout_path.suffix+'.process.json'),row)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if sys.platform!='linux':p.error('fresh Linux saved-data receiver required')
    tools=Path(__file__).resolve().parent;repo=tools.parents[1]
    cache=args.cache.resolve();output=args.output.resolve()
    if (cache.exists()or output.exists()or not cache.parent.is_dir()or not output.parent.is_dir()
            or cache.is_relative_to(repo)or output.is_relative_to(repo)
            or cache.is_relative_to(output)or output.is_relative_to(cache)):
        p.error('fresh disjoint external cache/output required')
    output.mkdir()
    free=shutil.disk_usage(cache.parent).free
    dump(output/'storage-admission.json',{'free_bytes':free,'minimum_start_bytes':8*1024**3,
        'admitted':free>=8*1024**3,'artifact_consumed':False,'new_native_calls':0})
    if free<8*1024**3:p.error('original8GiB native-artifact floor unmet')
    protocol_path=tools/'remote-thread-cpu-replay.protocol.json'
    protocol_raw=protocol_path.read_bytes();protocol=json.loads(protocol_raw)
    if (protocol['schema_version']!=1 or protocol['native_source']!=SOURCE
            or protocol['native_run']!=RUN or protocol['native_job']!=JOB or protocol['native_feature_run']!=FEATURE
            or protocol['artifact_id']!=ARTIFACT or ARTIFACT<=0 or protocol['archive']!=ARCHIVE
            or not 0<ARCHIVE['bytes']<=128*1024**2 or protocol['original_start_floor_bytes']!=8*1024**3
            or protocol['native_archive_cap_bytes']!=256*1024**2 or protocol['bundle_cap_bytes']!=128*1024**2):
        raise ValueError('exact pre-execution archive/protocol required')
    native_path=tools/'windows-thread-cpu-profile.protocol.json'
    if descriptor(native_path)!=NATIVE_PROTOCOL or protocol['native_protocol']!=NATIVE_PROTOCOL:
        raise ValueError('original native protocol changed')
    native=json.loads(native_path.read_bytes())
    fixed={**{str(tools/name):v for name,v in native['tools'].items()},
           **{str(repo/name):v for name,v in native['repository_files'].items()},
           **{str(repo/name):v for name,v in protocol['files'].items()},
           str(native_path):NATIVE_PROTOCOL,str(tools/'remote_q8_apack_replay.py'):HELPER}
    def frozen():
        if protocol_path.read_bytes()!=protocol_raw:raise ValueError('remote protocol changed')
        for name,wanted in fixed.items():
            if descriptor(Path(name))!=wanted:raise ValueError('first-bound replay input changed: '+name)
    frozen()
    # No project helper is imported until all native/receiver inputs are bound.
    from remote_q8_apack_replay import bounded_process,verify_collection
    from pooled_runtime import BOUNDED_PROFILE,load_profile
    _,lock=load_profile(BOUNDED_PROFILE)
    cache.mkdir();logs=output/'commands';logs.mkdir()
    environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','GIT_LFS_SKIP_SMUDGE':'1',
        'GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1'}
    commands=[]
    def run(name,command,timeout=900,limit=8*1024**2,owned_stage=False):
        frozen()
        row=bounded_process(list(map(str,command)),logs/(name+'.stdout.log'),logs/(name+'.stderr.log'),
            cwd=repo,env={**environment,**({'GEN2_REMOTE_OWNED_STAGE':'1'}if owned_stage else {})},timeout=timeout,limit=limit)
        commands.append({'name':name,'owned_stage_inherited_children':owned_stage,**row})
        (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        frozen();return (logs/(name+'.stdout.log')).read_bytes()
    base=run('base',['git','rev-parse','HEAD']).decode().strip()
    if base!=os.environ.get('GITHUB_SHA'):raise ValueError('committed CI receiver source required')
    if run('clean',['git','status','--porcelain']).strip():raise ValueError('clean receiver checkout required')
    if run('committed-protocol',['git','show',base+':tools/nemotron-diarization/'+protocol_path.name])!=protocol_raw:
        raise ValueError('committed replay protocol required')
    for name in protocol['files']:
        if run('blob-'+hashlib.sha256(name.encode()).hexdigest()[:16],['git','show',base+':'+name])!=(repo/name).read_bytes():
            raise ValueError('uncommitted replay input: '+name)
    if run('native-protocol',['git','show',SOURCE+':tools/nemotron-diarization/windows-thread-cpu-profile.protocol.json'])!=native_path.read_bytes():
        raise ValueError('committed native producer protocol changed')
    parent=repo/PARENT
    manifest=json.loads((parent/'collection.json').read_bytes())['files']
    for name,wanted in manifest.items():
        if descriptor(parent/name)!=wanted:raise ValueError('reviewed receiver preparation changed')
    prepared=cache/'thread-cpu-receivers-prepared-v3';prepared.mkdir()
    for name in ['preparation.json',*EXECUTED]:
        shutil.copyfile(parent/'prepared'/name,prepared/name)
        fixed[str(prepared/name)]=manifest['prepared/'+name]
    relocated={}
    for name,origin in [(name,parent/'prepared'/name)for name in EXECUTED]+[
            ('prepare-thread-cpu-source-v3.py',parent/'prepare-thread-cpu-source-v3.py'),
            ('test-thread-cpu-receiver-bindings-v1.py',parent/'test-thread-cpu-receiver-bindings-v1.py')]:
        raw=relocate(origin.read_bytes(),repo,cache);target=cache/name
        with target.open('xb')as f:f.write(raw)
        relocated[str(target)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    def prepared_frozen():
        frozen()
        for name,wanted in relocated.items():
            if descriptor(Path(name))!=wanted:raise ValueError('relocated receiver changed')
    dump(output/'pre-execution.json',{'source':base,'native_source':SOURCE,'native_run':RUN,'native_job':JOB,
        'native_archive':ARCHIVE,'relocated_files':relocated,'artifact_consumed':False,'new_native_calls':0})
    python=[sys.executable,'-B']
    run('binding-controls',python+[cache/'test-thread-cpu-receiver-bindings-v1.py'])
    prepared_frozen()
    for name,url,revision in [('native','https://github.com/NVIDIA/NeMo-Speech.cpp.git',lock['runtime']['revision']),
            ('llama','https://github.com/ggml-org/llama.cpp.git',lock['runtime']['llama_revision'])]:
        path=cache/'NeMo-Speech.cpp' if name=='native' else cache/'NeMo-Speech.cpp/llama.cpp'
        run('pristine-init-'+name,['git','init',path])
        run('pristine-origin-'+name,['git','-C',path,'remote','add','origin',url])
        run('pristine-fetch-'+name,['git','-C',path,'fetch','--depth=1','origin',revision],timeout=180)
        run('pristine-checkout-'+name,['git','-C',path,'checkout','--detach',revision])
    run('source-only-preparation',python+[cache/'prepare-thread-cpu-source-v3.py'],owned_stage=True)
    prepared_frozen()
    source_root=cache/'windows-thread-cpu-profile-source-admission-v1'
    source_q_raw=(source_root/'qualification.json').read_bytes();source_q=json.loads(source_q_raw)
    source_stage=next(row for row in commands if row['name']=='source-only-preparation')
    if (source_q['completed']is not True or source_q['compiled_sdk_or_model']is not False
            or len(source_q['materialized_files'])!=3425 or len(source_q['commands'])!=11
            or any(row['returncode']!=0 or row['completed']is not True or row['leader_reaped']is not True
                or row['stage_group']!=source_stage['pid']or row['group_owner']!='outer bounded stage supervisor'
                for row in source_q['commands'])):
        raise ValueError('complete source-only owned-stage proof required')
    source_copy=output/'source-preparation';source_copy.mkdir()
    source_proof={'qualification.json':byte_descriptor(source_q_raw)}
    for path in (source_root/'evidence').rglob('*'):
        if path.is_symlink():raise ValueError('regular source proof required')
        if path.is_dir():continue
        if not path.is_file():raise ValueError('regular source proof required')
        source_proof[path.relative_to(source_root).as_posix()]=descriptor(path)
    if len(source_proof)>1024 or sum(v['bytes']for v in source_proof.values())>64*1024**2:
        raise ValueError('bounded source-only proof required')
    for name,wanted in source_proof.items():
        target=source_copy/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_root/name,target)
        if descriptor(target)!=wanted or descriptor(source_root/name)!=wanted:
            raise ValueError('first-bound source proof changed')
        fixed[str(source_root/name)]=wanted;fixed[str(target)]=wanted
    dump(source_copy/'collection.json',{'files':source_proof})
    help_text=run('gh-api-help',['gh','api','--help'])
    run('gh-version',['gh','version'])
    gh=['gh','api']+(['--allow-escape-sequences']if b'--allow-escape-sequences'in help_text else [])
    api={};api_raw={}
    for name,endpoint in [('producer-run.json',f'runs/{RUN}'),('job.json',f'jobs/{JOB}'),('artifacts.json',f'runs/{RUN}/artifacts')]:
        raw=run('api-'+name,gh+['repos/saberra-ai/gen2/actions/'+endpoint]);api_raw[name]=raw;api[name]=json.loads(raw)
    admit_api(api['producer-run.json'],api['job.json'],api['artifacts.json'])
    data=cache/'windows-thread-cpu-profile-ci-v1';data.mkdir()
    api_paths={}
    for name in api:
        raw=api_raw[name]
        with(data/name).open('xb')as f:f.write(raw)
        api_paths[str(data/name)]=byte_descriptor(raw)
        api_paths[str(logs/('api-'+name+'.stdout.log'))]=byte_descriptor(raw)
    joblog=run('native-job-log',gh+[f'repos/saberra-ai/gen2/actions/jobs/{JOB}/logs'])
    with(data/'job.log').open('xb')as f:f.write(joblog)
    api_paths[str(data/'job.log')]=byte_descriptor(joblog)
    api_paths[str(logs/'native-job-log.stdout.log')]=byte_descriptor(joblog)
    fixed.update(api_paths)
    prepared_frozen()
    archive_path=data/'artifact.zip'
    # The frozen archive is below the existing supervisor's128MiB byte bound.
    bounded_process(gh+[f'repos/saberra-ai/gen2/actions/artifacts/{ARTIFACT}/zip'],archive_path,
        output/'archive.stderr.log',cwd=repo,env=environment,timeout=180,limit=ARCHIVE['bytes'])
    if descriptor(archive_path)!=ARCHIVE:raise ValueError('complete original ZIP differs from frozen API identity')
    fixed[str(archive_path)]=ARCHIVE
    for name,script in [('delivery',EXECUTED[0]),('supervision',EXECUTED[1]),('collection',EXECUTED[2])]:
        prepared_frozen()
        run(name,python+[cache/script])
        if descriptor(archive_path)!=ARCHIVE:raise ValueError('original archive changed during replay')
        produced=([data/'delivery-admission-v1/qualification.json',data/'delivery-admission-v1/artifact-members.json']
            if name=='delivery'else[data/'supervision-replay-v1.json']if name=='supervision'else[])
        for path in produced:fixed[str(path)]=descriptor(path)
    prepared_frozen()
    for name,wanted in api_paths.items():
        if descriptor(Path(name))!=wanted:raise ValueError('first-bound native metadata/log changed')
    result_root=repo/RESULT;result_manifest=(result_root/'collection.json').read_bytes()
    collection=verify_collection(result_root)
    if collection['manifest']!=byte_descriptor(result_manifest):raise ValueError('first-bound result manifest changed')
    fixed[str(result_root/'collection.json')]=byte_descriptor(result_manifest)
    for name,wanted in json.loads(result_manifest)['files'].items():fixed[str(result_root/name)]=wanted
    for name in api:
        if descriptor(result_root/name)!=byte_descriptor(api_raw[name]):raise ValueError('published native API bytes changed')
    if descriptor(result_root/'job.log')!=byte_descriptor(joblog):raise ValueError('published native job log changed')
    run('stage-result',['git','add','--',RESULT])
    staged=run('staged-paths',['git','diff','--cached','--name-only','-z']).decode().strip('\0').split('\0')
    if not staged or any(not n.startswith(RESULT+'/')for n in staged):raise ValueError('only new evidence may be bundled')
    run('commit-result',['git','-c','user.name=Gen2 evidence receiver','-c','user.email=gen2-evidence@invalid',
        '-c','gc.auto=0','-c','core.hooksPath=/dev/null','commit','-q','-m','Retain independently replayed Windows CPU counters'])
    result=run('result-commit',['git','rev-parse','HEAD']).decode().strip()
    if run('result-parent',['git','rev-parse','HEAD^']).decode().strip()!=base:raise ValueError('one result-only commit required')
    bundle=output/'receiver-result.bundle'
    bounded_process(['git','bundle','create','-','HEAD','^'+base],bundle,output/'bundle.stderr.log',
        cwd=repo,env=environment,timeout=600,limit=128*1024**2)
    run('verify-bundle',['git','bundle','verify',bundle])
    dump(output/'bundle.json',{'verifier_source_commit':base,'verifier_run_id':os.environ['GITHUB_RUN_ID'],
        'native_source_commit':SOURCE,'native_run_id':RUN,'native_job_id':JOB,'native_archive':ARCHIVE,
        'result_commit':result,'result_parent':base,'result_roots':[RESULT],'collections':{RESULT:collection},
        'bundle':descriptor(bundle),'original_start_floor_bytes':8*1024**3,'new_native_calls':0,
        'production_or_release_qualified':False,'scope':'Full original ZIP admitted/replayed on Linux without native SDK/model execution. Local bundle admission, independent replay and review remain required.'})
    print(json.dumps({'bundle':descriptor(bundle),'result_commit':result,'collection':collection,'new_native_calls':0}))


if __name__=='__main__':main()
