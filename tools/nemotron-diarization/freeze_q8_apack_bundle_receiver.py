"""Freeze local return admission after terminal remote APIs, before download."""
import json
from pathlib import Path

from remote_q8_apack_replay import descriptor
from receive_q8_apack_bundle import (VERIFIER,RUN,JOB,DELIVERY,ZIP_CAP,EXTRACT_CAP,BUNDLE_CAP,
    START_FLOOR,admit_api,strict_json)


def main():
    tools=Path(__file__).resolve().parent;repo=tools.parents[1]
    remote=tools/'remote-q8-apack-replay.protocol.json'
    if descriptor(remote)!={'bytes':18183,'sha256':'6053c9cd14e21a5810f3fde0d47a9ba44da2e325e7127ace881f59b0a13c1a5d'}:
        raise ValueError('fixed successful remote source protocol required')
    names=set(strict_json(remote.read_bytes())['files'])
    for name,wanted in strict_json(remote.read_bytes())['files'].items():
        if descriptor(repo/name)!=wanted:raise ValueError('remote frozen input changed: '+name)
    names.add('tools/nemotron-diarization/'+remote.name)
    parent_name='benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-apack-bundle-admission-preparation'
    parent=repo/parent_name
    if descriptor(parent/'collection.json')!={'bytes':436,'sha256':'128f0922b5ed979005caeda87376942e2abca5c210bb26682ba3b355cbc5f31b'}:
        raise ValueError('exact retained terminal API preparation required')
    names.add(parent_name+'/collection.json')
    for name,wanted in strict_json((parent/'collection.json').read_bytes())['files'].items():
        if descriptor(parent/name)!=wanted:raise ValueError('terminal API changed: '+name)
        names.add(parent_name+'/'+name)
    admit_api(*(strict_json((parent/name).read_bytes())for name in ['run.json','job.json','artifacts.json']))
    names.update('tools/nemotron-diarization/'+name for name in [Path(__file__).name,
        'receive_q8_apack_bundle.py','test_receive_q8_apack_bundle.py'])
    names.add('docs/research/nemotron-windows-q8-apack-bundle-admission.md')
    value={'schema_version':1,'remote_source':VERIFIER,'remote_run_id':RUN,'remote_job_id':JOB,'delivery':DELIVERY,
        'zip_cap':ZIP_CAP,'extract_cap':EXTRACT_CAP,'bundle_cap':BUNDLE_CAP,'start_floor':START_FLOOR,
        'files':{name:descriptor(repo/name)for name in sorted(names)},
        'scope':'Quarantine-only bounded local result delivery; original native archive floor/gates unchanged. No source refs/index/files changed; raw numerical/timing/source interpretation remains pending.'}
    output=tools/'q8-apack-bundle-receive.protocol.json';output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'protocol':descriptor(output),'files':len(names),'delivery':DELIVERY}))


if __name__=='__main__':main()
