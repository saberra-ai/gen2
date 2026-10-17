"""Freeze the remote receiver route without changing the native experiment."""
import json
from pathlib import Path

from remote_q8_apack_replay import descriptor,PARENT,SOURCE,RUN,JOB,FEATURE,ARTIFACT,ARCHIVE


def main():
    tools=Path(__file__).resolve().parent;repo=tools.parents[1];parent=repo/PARENT
    expected={'bytes':6091,'sha256':'352e64a11fc41e5f02fc66dd802b4e70cc0db7ab8ec9815981cbdb10ee0a6b7e'}
    if descriptor(parent/'collection.json')!=expected:raise ValueError('reviewed preparation manifest required')
    names={PARENT+'/collection.json',
        'tools/nemotron-diarization/remote_q8_apack_replay.py',
        'tools/nemotron-diarization/test_remote_q8_apack_replay.py',
        'tools/nemotron-diarization/freeze_remote_q8_apack_replay.py',
        '.github/workflows/diarization-q8-apack-replay.yml',
        'docs/research/nemotron-windows-q8-apack-remote-replay.md'}
    for name,wanted in json.loads((parent/'collection.json').read_bytes())['files'].items():
        if descriptor(parent/name)!=wanted:raise ValueError('reviewed preparation changed: '+name)
        names.add(PARENT+'/'+name)
    native=tools/'windows-q8-apack.protocol.json'
    if descriptor(native)!={'bytes':885307,'sha256':'329e0afea3c8e4f05d6158e58e801519a0dfb3d76fa0e1b27afa9d00eb7e5c2f'}:
        raise ValueError('unchanged native protocol required')
    result={'schema_version':1,'native_source':SOURCE,'native_run_id':RUN,'native_job_id':JOB,
        'native_feature_run_id':FEATURE,'artifact_id':ARTIFACT,'archive':ARCHIVE,'native_protocol':descriptor(native),
        'parent_manifest':expected,'files':{n:descriptor(repo/n)for n in sorted(names)},
        'original_start_floor_bytes':8*1024**3,'archive_cap_bytes':1024**3,'bundle_cap_bytes':128*1024**2,
        'limits':['Full unchanged original archive admission, source/command/vector/half/packing/timing/observer replay on Linux receiver. Only literal path prefixes relocate; no kernel/compiler/SDK/model rerun.',
            'Original8GiBreceiver start floor and1GiBarchive bound remain in the fetched receiver itself. Remote API archive identity is fixed before download and completeZIPdigest rechecked before admission.',
            'Owned POSIX processgroups, boundedstdout/stderr and deadlines apply to every receiver command. Original95nativeWindowsJob observations remain independently replayed.',
            'Only two closed result directories are staged into one ephemeral result-only commit. Bounded incremental Gitbundle is returned; no remote branch write/push and no originalarchive ZIP copied intoGit.',
            'Bundle transfer/localimport/replay/review are separate pending gates; sourceartifact remains in originalGitHubartifact. No performance/model/production/release pass follows workflow completion alone.']}
    path=tools/'remote-q8-apack-replay.protocol.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'protocol':descriptor(path),'files':len(names),'native_protocol':result['native_protocol']}))


if __name__=='__main__':main()
