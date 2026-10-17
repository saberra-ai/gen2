"""Freeze the original/candidate sources, complete imports and actual prior result."""
import ast
import hashlib
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from pooled_runtime import BOUNDED_PROFILE, load_profile
from q8_tile_source import ORIGINAL, PATH, transform
from windows_q8_tile_metrics import ORDER, SHAPES, PRIMARY_SHAPES, REPETITIONS


def main():
    tools=Path(__file__).resolve().parent
    repo=tools.parents[1]
    parent_path=tools/'windows-math-errno.protocol.json'
    if descriptor(parent_path)!={'bytes':319875,'sha256':'290b3f890eb8a5ce413b0d7daa798baf542bed5149d648fb6cce63eb1dd3c4af'}:
        raise ValueError('frozen math errno protocol changed')
    parent=json.loads(parent_path.read_text())
    for group,root in [('tools',tools),('repository_files',repo)]:
        for name,expected in parent[group].items():
            if descriptor(root/name)!=expected: raise ValueError('historical input changed: '+name)
    base='benches/results/nemotron-diarization/windows-x86_64/2026-10-17-math-errno'
    manifest=json.loads((repo/base/'collection.json').read_text())['files']
    for name,expected in manifest.items():
        if descriptor(repo/base/name)!=expected: raise ValueError('preceding actual receipt changed: '+name)
    audit=json.loads((repo/base/'evidence-control/platform-llama-materialized-audit.json').read_text())
    fixture=tools/'fixtures/q8-tile-sgemm.cpp'
    if descriptor(fixture)!=ORIGINAL or audit['files'][PATH]!=ORIGINAL or len(audit['files'])!=3425:
        raise ValueError('exact canonical SGEMM fixture required')
    candidate=transform(fixture.read_bytes())
    references=set(parent['repository_files'])|{base+'/'+name for name in manifest}|{base+'/collection.json',
        '.github/workflows/diarization-windows-q8-tile.yml','docs/research/nemotron-windows-q8-tile-probe.md'}
    names=set(parent['tools'])|{parent_path.name,'q8_tile_benchmark.cpp','q8_tile_benchmark.cmake','fixtures/q8-tile-sgemm.cpp'}
    pending=['windows_q8_tile_probe.py','test_windows_q8_tile.py','freeze_windows_q8_tile.py']
    visited=set()
    while pending:
        name=pending.pop()
        if name in visited: continue
        visited.add(name)
        names.add(name)
        for node in ast.walk(ast.parse((tools/name).read_bytes())):
            imports=([node.module]if isinstance(node,ast.ImportFrom)else[item.name for item in node.names]if isinstance(node,ast.Import)else[])
            for module in imports:
                local=(module or '').split('.')[0]+'.py'
                if (tools/local).is_file() and local not in visited: pending.append(local)
    _,lock=load_profile(BOUNDED_PROFILE)
    protocol={'schema_version':1,'order':ORDER,'shapes':SHAPES,'primary_shapes':PRIMARY_SHAPES,'timed_calls_per_shape':REPETITIONS,
        'native_revision':lock['runtime']['revision'],'llama_revision':lock['runtime']['llama_revision'],
        'original_source':ORIGINAL,'candidate_source':{'bytes':len(candidate),'sha256':hashlib.sha256(candidate).hexdigest()},
        'tools':{name:descriptor(tools/name)for name in sorted(names)},
        'repository_files':{name:descriptor(repo/name)for name in sorted(references)},
        'limits':['Model-free Windows Q8 kernel control only; no NeMo SDK/model/public/lifecycle/release qualification.',
            'One Q8-only Windows AVX2 tile dispatch change; all arithmetic and canonical source remain independently checked.',
            'Original exact Clang/linker, contraction off, original platform/Release flags and fixed AVX2/FMA/F16C.',
            'Twelve fresh processes and all84 finite nonzero vectors must remain byte-identical, with guarded padded strides.',
            '5-percent speedup and 10-percent drift predicates apply to all three primary synthetic shapes; retain every timing.',
            'A useful kernel still requires separate full SDK/model/numerical/performance and all original release gates.']}
    target=tools/'windows-q8-tile.protocol.json'
    target.write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'protocol':descriptor(target),'tools':len(names),'repository_files':len(references),'local_import_closure':len(visited)}))


if __name__=='__main__':main()
