"""Freeze the separate saved-artifact replay route without changing native gates."""
import json
from pathlib import Path

from remote_thread_cpu_replay import (SOURCE,RUN,JOB,FEATURE,ARTIFACT,ARCHIVE,NATIVE_PROTOCOL,HELPER,
    PARENT,descriptor)


def main():
    tools=Path(__file__).resolve().parent;repo=tools.parents[1]
    if ARTIFACT<=0 or set(ARCHIVE)!={'bytes','sha256'}or not 0<ARCHIVE['bytes']<=128*1024**2:
        raise ValueError('observe and freeze actual terminal native artifact identity first')
    native_path=tools/'windows-thread-cpu-profile.protocol.json'
    if descriptor(native_path)!=NATIVE_PROTOCOL or descriptor(tools/'remote_q8_apack_replay.py')!=HELPER:
        raise ValueError('unchanged native protocol and process helper required')
    native=json.loads(native_path.read_bytes())
    for group,base in [('tools',tools),('repository_files',repo)]:
        for name,wanted in native[group].items():
            if descriptor(base/name)!=wanted:raise ValueError('native frozen input changed: '+name)
    parent=repo/PARENT;members=json.loads((parent/'collection.json').read_bytes())['files']
    if {p.relative_to(parent).as_posix()for p in parent.rglob('*')if p.is_file()}!=set(members)|{'collection.json'}:
        raise ValueError('closed preparation collection required')
    for name,wanted in members.items():
        if descriptor(parent/name)!=wanted:raise ValueError('preparation changed: '+name)
    names={PARENT+'/'+name for name in members}|{PARENT+'/collection.json',
        '.github/workflows/diarization-thread-cpu-replay.yml','docs/research/nemotron-windows-thread-cpu-replay.md'}
    names|={'tools/nemotron-diarization/'+name for name in [
        'remote_thread_cpu_replay.py','test_remote_thread_cpu_replay.py','freeze_remote_thread_cpu_replay.py',
        'remote_q8_apack_replay.py','test_remote_q8_apack_replay.py']}
    previous_path=tools/'remote-q8-apack-replay.protocol.json'
    if descriptor(previous_path)!={'bytes':18183,'sha256':'6053c9cd14e21a5810f3fde0d47a9ba44da2e325e7127ace881f59b0a13c1a5d'}:
        raise ValueError('existing process/control route protocol changed')
    previous=json.loads(previous_path.read_bytes())
    for name,wanted in previous['files'].items():
        if descriptor(repo/name)!=wanted:raise ValueError('existing process/control dependency changed: '+name)
    names|=set(previous['files'])|{'tools/nemotron-diarization/remote-q8-apack-replay.protocol.json'}
    protocol={'schema_version':1,'native_source':SOURCE,'native_run':RUN,'native_job':JOB,
        'native_feature_run':FEATURE,'artifact_id':ARTIFACT,'archive':ARCHIVE,'native_protocol':NATIVE_PROTOCOL,
        'original_start_floor_bytes':8*1024**3,'native_archive_cap_bytes':256*1024**2,
        'bundle_cap_bytes':128*1024**2,'files':{name:descriptor(repo/name)for name in sorted(names)},
        'limits':['No new compiler, SDK, graph or model execution; source-only composition and saved-data replay.',
            'The unchanged original8GiB full-native start floor is enforced on independent Linux CI; actual frozen ZIP is within the existing process supervisor128MiB output bound and original256MiB native archive cap.',
            'All135 native tools and2880 repository references remain pinned and checked before project imports and every stage.',
            'Actual native source/SDK/compiler/graph/output/counter/supervision evidence must independently replay. Preserve zero counters and original observer bounds.',
            'Exact59 command sequence,12 graph processes before four model cases, original libraries/retained copies and complete model argv/cwd are mandatory.',
            'Bounded result-only Git commit/bundle is returned for separate local quarantine admission, independent replay and review. No production/main or release qualification.']}
    target=tools/'remote-thread-cpu-replay.protocol.json'
    target.write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'protocol':descriptor(target),'files':len(names),'new_native_calls':0}))


if __name__=='__main__':main()
