"""Freeze the fixed bounded result return, preserving the separate native8GiB floor."""
import json
from pathlib import Path

from receive_thread_cpu_bundle import (VERIFIER,RUN,JOB,DELIVERY,ZIP_CAP,EXTRACT_CAP,BUNDLE_CAP,START_FLOOR,
    RESULT_COMMIT,BUNDLE,COLLECTION,RESULTS,descriptor)


def main():
    tools=Path(__file__).resolve().parent;repo=tools.parents[1]
    remote_path=tools/'remote-thread-cpu-replay.protocol.json'
    if descriptor(remote_path)!={'bytes':25164,'sha256':'b7614d9205a59fa79a02cfecfb14c70107a55aa6da9e3bad8325d3649de798c8'}:
        raise ValueError('original frozen remote route changed')
    remote=json.loads(remote_path.read_bytes())
    for name,wanted in remote['files'].items():
        if descriptor(repo/name)!=wanted:raise ValueError('remote dependency changed: '+name)
    names=set(remote['files'])|{'tools/nemotron-diarization/'+name for name in [
        remote_path.name,'receive_thread_cpu_bundle.py','test_receive_thread_cpu_bundle.py',
        'freeze_thread_cpu_bundle_receiver.py','receive_q8_apack_bundle.py']}
    names.add('docs/research/nemotron-windows-thread-cpu-bundle-receive.md')
    preparation='benches/results/nemotron-diarization/windows-x86_64/2026-10-17-thread-cpu-return-preparation'
    manifest=json.loads((repo/preparation/'collection.json').read_bytes())['files']
    if {p.relative_to(repo/preparation).as_posix()for p in (repo/preparation).rglob('*')if p.is_file()}!=set(manifest)|{'collection.json'}:
        raise ValueError('closed return preparation required')
    for name,wanted in manifest.items():
        if descriptor(repo/preparation/name)!=wanted:raise ValueError('return preparation changed')
    names|={preparation+'/'+name for name in manifest}|{preparation+'/collection.json'}
    protocol={'schema_version':1,'remote_source':VERIFIER,'remote_run_id':RUN,'remote_job_id':JOB,
        'delivery':DELIVERY,'zip_cap':ZIP_CAP,'extract_cap':EXTRACT_CAP,'bundle_cap':BUNDLE_CAP,
        'start_floor':START_FLOOR,'result_commit':RESULT_COMMIT,'bundle':BUNDLE,'result_roots':list(RESULTS),
        'collection':COLLECTION,'files':{name:descriptor(repo/name)for name in sorted(names)},
        'limits':['Separate bounded result return with unchanged4GiB floor; original complete-native8GiB floor remains enforced remotely.',
            'Exact terminal native/remote/archive/bundle/result identities; source-only proof and complete result collection retain separate scope.',
            'Existing bounded process helper is hashed before execution and compiled from that same captured byte string. No project helper imported before that binding.',
            'Original safe-name, JSON uniqueness, Git v2 bundle header and complete quarantine algorithms remain AST-identical. No source refs/index/object import during admission.',
            'Raw numerical/counter/timing replay and review precede explicit result import. No native SDK/model calls or production/release claim.']}
    target=tools/'thread-cpu-bundle-receive.protocol.json'
    target.write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'protocol':descriptor(target),'files':len(names),'return_consumed':False}))


if __name__=='__main__':main()
