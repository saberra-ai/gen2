"""Admit one fixed CI result bundle into quarantine without changing source refs."""
import argparse
import json
import os
from pathlib import Path,PurePosixPath
import re
import shutil
import stat
import sys
import zipfile

from remote_q8_apack_replay import (SOURCE as NATIVE_SOURCE,RUN as NATIVE_RUN,JOB as NATIVE_JOB,
    ARCHIVE,RESULTS,descriptor,bounded_process,verify_collection)

VERIFIER='cc655ab95f2bb800696f1f5c9a27d69321627b27'
RUN,JOB=39080172371,117535343984
DELIVERY={'id':12118043328,'bytes':17392754,
    'sha256':'0a31a38d6e136bdf48a66b7b106745ab1c175e601e6f44b2af438cbe207bcd16'}
ZIP_CAP=128*1024**2
EXTRACT_CAP=192*1024**2
BUNDLE_CAP=128*1024**2
START_FLOOR=4*1024**3


def strict_json(raw):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=unique)


def safe_name(name):
    p=PurePosixPath(name)
    if (not re.fullmatch(r'[A-Za-z0-9._/-]+',name)or not p.parts or p.is_absolute()or p.as_posix()!=name
            or any(part.casefold()in ('..','.git')for part in p.parts)):
        raise ValueError('unsafe member path')
    return p


def admit_api(run,job,artifacts):
    if (run['id']!=RUN or run['head_sha']!=VERIFIER or run['status']!='completed'
            or run['conclusion']!='success' or run['repository']['full_name']!='saberra-ai/gen2'):
        raise ValueError('exact successful remote receiver required')
    if (job['id']!=JOB or job['run_id']!=RUN or job['head_sha']!=VERIFIER
            or job['status']!='completed' or job['conclusion']!='success'):
        raise ValueError('exact successful receiver job required')
    artifact,=artifacts['artifacts']
    if (artifact['name']!='nemotron-q8-apack-replay' or artifact['expired']is not False
            or artifact['workflow_run']['id']!=RUN or artifact['workflow_run']['head_sha']!=VERIFIER
            or type(artifact['id'])is not int or artifact['id']<=0
            or type(artifact['size_in_bytes'])is not int or not 0<artifact['size_in_bytes']<=ZIP_CAP
            or artifact['id']!=DELIVERY['id']or artifact['size_in_bytes']!=DELIVERY['bytes']
            or artifact['digest']!='sha256:'+DELIVERY['sha256']):
        raise ValueError('bounded exact-run result artifact required')
    return artifact


def extract_delivery(archive,dest):
    """Admit the bounded command-output package; never execute its contents."""
    root_files={'receiver-result.bundle','bundle.json','receiver-result.bundle.process.json',
        'bundle.stderr.log','storage-admission.json','commands.json'}
    if archive.stat().st_size>ZIP_CAP or dest.exists():raise ValueError('fresh bounded delivery required')
    entries=[];folded=set();total=0
    with zipfile.ZipFile(archive)as source:
        if len(source.infolist())>1000:raise ValueError('too many delivery members')
        # Complete metadata admission precedes any extracted file write.
        for item in source.infolist():
            name=item.filename.rstrip('/')if item.is_dir()else item.filename
            path=safe_name(name)
            if name.casefold()in folded or item.flag_bits&1:raise ValueError('duplicate/encrypted member')
            folded.add(name.casefold())
            mode=stat.S_IFMT(item.external_attr>>16)
            if item.is_dir():
                if name!='commands' or mode not in (0,stat.S_IFDIR):raise ValueError('unexpected directory')
                continue
            if (mode not in (0,stat.S_IFREG)or not(name in root_files
                    or(len(path.parts)==2 and path.parts[0]=='commands'))):
                raise ValueError('unexpected regular delivery member')
            cap=BUNDLE_CAP if name=='receiver-result.bundle'else 8*1024**2
            if not 0<=item.file_size<=cap:raise ValueError('oversize delivery member')
            total+=item.file_size
            if total>EXTRACT_CAP:raise ValueError('delivery expansion cap')
            entries.append((item,path))
        names={str(path)for _,path in entries}
        if not root_files<=names:raise ValueError('incomplete result delivery')
        dest.mkdir()
        manifest={}
        for item,path in entries:
            target=dest.joinpath(*path.parts);target.parent.mkdir(parents=True,exist_ok=True)
            count=0
            with source.open(item)as incoming,target.open('xb')as outgoing:
                while block:=incoming.read(1024**2):
                    count+=len(block)
                    if count>item.file_size:raise ValueError('decompressed member exceeds admitted size')
                    outgoing.write(block)
            if count!=item.file_size:raise ValueError('truncated delivery member')
            manifest[str(path)]=descriptor(target)
    return manifest


def admit_bundle_metadata(value,bundle):
    if (value['verifier_source_commit']!=VERIFIER or str(value['verifier_run_id'])!=str(RUN)
            or value['native_source_commit']!=NATIVE_SOURCE or value['native_run_id']!=NATIVE_RUN
            or value['native_job_id']!=NATIVE_JOB or value['native_archive']!=ARCHIVE
            or value['result_parent']!=VERIFIER or value['result_roots']!=list(RESULTS)
            or set(value['collections'])!=set(RESULTS)or value['bundle']!=descriptor(bundle)
            or not 0<value['bundle']['bytes']<=BUNDLE_CAP
            or value['original_start_floor_bytes']!=8*1024**3
            or value['new_compiler_sdk_model_calls']!=0 or value['production_or_release_qualified']is not False
            or not re.fullmatch(r'[0-9a-f]{40}',value['result_commit'])or value['result_commit']==VERIFIER):
        raise ValueError('exact bounded result metadata required')


def bundle_header(path,base,result):
    with path.open('rb')as f:
        if f.readline(64)!=b'# v2 git bundle\n':raise ValueError('SHA1 v2 bundle required')
        lines=[];size=0
        while True:
            line=f.readline(65537);size+=len(line)
            if not line or size>65536:raise ValueError('bounded complete bundle header required')
            if line==b'\n':break
            lines.append(line.decode('ascii').rstrip('\n'))
    if (len(lines)!=2 or not lines[0].startswith('-'+base+' ')
            or lines[1]!=result+' HEAD'):
        raise ValueError('exact one base prerequisite and one HEAD required')


def quarantine_results(bundle,metadata,source_repo,output,run):
    """Inspect and materialize result blobs using a fresh separate object store."""
    base=metadata['verifier_source_commit'];result=metadata['result_commit']
    bundle_header(bundle,base,result)
    objects=Path(run('source-objects',['git','rev-parse','--git-path','objects']).decode().strip())
    if not objects.is_absolute():objects=source_repo/objects
    objects=objects.resolve(strict=True)
    quarantine=output/'quarantine.git'
    run('init-quarantine',['git','init','--bare','--template=','-q',quarantine])
    (quarantine/'objects/info/alternates').write_text(str(objects)+'\n')
    prefix=['git','--git-dir='+str(quarantine),'-c','core.hooksPath=/dev/null','-c','gc.auto=0',
        '-c','maintenance.auto=false','-c','core.deltaBaseCacheLimit=16m','-c','pack.threads=1']
    run('verify-bundle',prefix+['bundle','verify',bundle])
    run('unbundle',prefix+['bundle','unbundle',bundle])
    parents=run('parents',prefix+['show','-s','--format=%P',result]).decode().strip()
    if parents!=base:raise ValueError('exact one result parent required')
    if run('base-roots',prefix+['ls-tree','-r','--name-only',base,'--',*RESULTS]).strip():
        raise ValueError('result roots must be new at verifier base')
    changes=run('changes',prefix+['diff-tree','--no-commit-id','--raw','--no-renames','-r','-t','-z',base,result]).decode().split('\0')
    if changes[-1]!=''or len(changes)%2!=1:raise ValueError('complete Git change list required')
    paths=set()
    for header,name in zip(changes[::2],changes[1::2]):
        safe_name(name)
        parts=header.removeprefix(':').split()
        if len(parts)!=5 or not header.startswith(':'):raise ValueError('complete raw tree change required')
        oldmode,newmode,oldoid,newoid,status=parts
        inside=any(name.startswith(root+'/')for root in RESULTS)
        ancestor=any(root.startswith(name+'/')for root in RESULTS)
        if newmode=='040000':
            if (newoid=='4b825dc642cb6eb9a060e54bf8d69288fbee4904'
                    or not(inside or ancestor or name in RESULTS)
                    or not((status=='A'and oldmode=='000000')
                        or(status=='M'and oldmode=='040000'and ancestor))):
                raise ValueError('only nonempty result trees and their ancestor updates required')
            continue
        if (status!='A'or oldmode!='000000'or newmode!='100644'or name in paths or not inside):
            raise ValueError('only additions under the two result roots required')
        paths.add(name)
    entries=run('trees',prefix+['ls-tree','-r','-l','-z',result,'--',*RESULTS]).split(b'\0')
    if entries[-1]!=b'':raise ValueError('complete Git tree required')
    rows=[];folded=set();selected={};totals={root:0 for root in RESULTS};counts={root:0 for root in RESULTS}
    for raw in entries[:-1]:
        header,name=raw.decode().split('\t');mode,kind,oid,size=header.split();safe_name(name)
        if (mode!='100644'or kind!='blob'or not re.fullmatch(r'[0-9a-f]{40}',oid)
                or name.casefold()in folded):raise ValueError('unique regular Git blobs required')
        folded.add(name.casefold());rows.append((name,oid))
        if not 0<=int(size)<=128*1024**2:raise ValueError('returned blob size cap')
        selected[oid]=int(size)
    if set(name for name,_ in rows)!=paths:raise ValueError('tree and addition paths differ')
    for name,oid in rows:
        root=next(root for root in RESULTS if name.startswith(root+'/'))
        totals[root]+=selected[oid];counts[root]+=1
        if totals[root]>1024**3 or counts[root]>10000:raise ValueError('bounded result collection required')
    materialized=output/'materialized';materialized.mkdir();stage=output/'blob-stage';stage.mkdir()
    for index,(name,oid)in enumerate(rows):
        temporary=stage/str(index);target=materialized/name;target.parent.mkdir(parents=True,exist_ok=True)
        run('blob-'+str(index),prefix+['cat-file','blob',oid],stdout_path=temporary,limit=max(1,selected[oid]))
        if temporary.stat().st_size!=selected[oid]:raise ValueError('blob size mismatch')
        temporary.rename(target)
    for name in RESULTS:strict_json((materialized/name/'collection.json').read_bytes())
    collections={name:verify_collection(materialized/name)for name in RESULTS}
    if collections!=metadata['collections']:raise ValueError('sender/local collection mismatch')
    return collections


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2];output=args.output.resolve()
    if output.exists()or output.is_relative_to(repo):parser.error('fresh external output required')
    free=shutil.disk_usage(output.parent).free
    if free<START_FLOOR:parser.error('4GiB return-delivery floor unmet; original native8GiB floor unchanged')
    output.mkdir();logs=output/'commands';logs.mkdir()
    env={k:v for k,v in os.environ.items()if not k.startswith('GIT_')}
    env.update({'PYTHONDONTWRITEBYTECODE':'1','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':os.devnull,
        'GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1','GIT_TERMINAL_PROMPT':'0'})
    commands=[]
    def run(name,command,stdout_path=None,limit=8*1024**2):
        path=stdout_path or logs/(name+'.stdout')
        row=bounded_process(list(map(str,command)),path,logs/(name+'.stderr'),cwd=repo,env=env,timeout=600,limit=limit)
        commands.append(row)
        (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        return path.read_bytes()if stdout_path is None else b''
    head=run('source-head',['git','rev-parse','HEAD']).decode().strip()
    refs=run('source-refs',['git','show-ref','--head'])
    index=Path(run('source-index',['git','rev-parse','--git-path','index']).decode().strip())
    if not index.is_absolute():index=repo/index
    index_before=descriptor(index)
    for name in (Path(__file__).name,'remote_q8_apack_replay.py','q8-apack-bundle-receive.protocol.json'):
        if run('committed-'+name,['git','show',head+':tools/nemotron-diarization/'+name])!=(repo/'tools/nemotron-diarization'/name).read_bytes():
            raise ValueError('committed reviewed local receiver dependency required')
    protocol=strict_json((Path(__file__).parent/'q8-apack-bundle-receive.protocol.json').read_bytes())
    if (protocol['schema_version']!=1 or protocol['remote_source']!=VERIFIER or protocol['remote_run_id']!=RUN
            or protocol['remote_job_id']!=JOB or protocol['delivery']!=DELIVERY or protocol['zip_cap']!=ZIP_CAP
            or protocol['extract_cap']!=EXTRACT_CAP or protocol['bundle_cap']!=BUNDLE_CAP
            or protocol['start_floor']!=START_FLOOR):raise ValueError('exact local delivery protocol required')
    for name,wanted in protocol['files'].items():
        if descriptor(repo/name)!=wanted:raise ValueError('local receiver frozen input changed: '+name)
    api={}
    for name,endpoint in [('run',f'runs/{RUN}'),('job',f'jobs/{JOB}'),('artifacts',f'runs/{RUN}/artifacts')]:
        api[name]=strict_json(run('api-'+name,['gh','api','repos/saberra-ai/gen2/actions/'+endpoint]))
    artifact=admit_api(api['run'],api['job'],api['artifacts'])
    (output/'pre-download.json').write_text(json.dumps({'artifact':artifact,'local_source_commit':head,
        'free_bytes':free,'minimum_free_bytes':START_FLOOR,'original_remote_native_floor_bytes':8*1024**3},indent=2)+'\n')
    help_raw=run('gh-help',['gh','api','--help']);raw_flag=['--allow-escape-sequences']if b'--allow-escape-sequences'in help_raw else []
    archive=output/'delivery.zip'
    run('download',['gh','api',*raw_flag,f"repos/saberra-ai/gen2/actions/artifacts/{artifact['id']}/zip"],
        stdout_path=archive,limit=artifact['size_in_bytes'])
    expected={'bytes':artifact['size_in_bytes'],'sha256':artifact['digest'][7:]}
    if descriptor(archive)!=expected:raise ValueError('result delivery digest mismatch')
    extracted=output/'delivery';files=extract_delivery(archive,extracted)
    metadata=strict_json((extracted/'bundle.json').read_bytes());bundle=extracted/'receiver-result.bundle'
    admit_bundle_metadata(metadata,bundle)
    collections=quarantine_results(bundle,metadata,repo,output,run)
    if (run('final-refs',['git','show-ref','--head'])!=refs or descriptor(index)!=index_before
            or run('final-head',['git','rev-parse','HEAD']).decode().strip()!=head):
        raise ValueError('source Git state changed during admission')
    (output/'admission.json').write_text(json.dumps({'receiver_source_commit':head,'remote_run_id':RUN,
        'remote_job_id':JOB,'remote_source_commit':VERIFIER,'remote_result_commit':metadata['result_commit'],
        'archive':expected,'delivery_members':files,'collections':collections,'source_refs_or_index_changed':False,
        'raw_numerical_timing_replay_pending':True,'production_or_release_qualified':False},indent=2)+'\n')
    print(json.dumps({'collections':collections,'raw_numerical_timing_replay_pending':True}))


if __name__=='__main__':main()
