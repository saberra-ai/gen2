"""Prepare saved-only packed-B-scale receivers from admitted row sources."""
import hashlib
import json
from pathlib import Path
import re

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
parent = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-row'


def desc(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


assert desc(parent / 'collection.json') == {'bytes':213899,
    'sha256':'213cd26a9737f0c7ac55a44f76114ed6a4b910a143222d0772e65a2c42b31a2a'}
manifest = json.loads((parent / 'collection.json').read_text())['files']
files = {name:name.replace('q8-row','q8-mx2') for name in [
    'fetch-windows-q8-row-artifact-v1.py', 'admit-windows-q8-row-delivery-v1.py',
    'replay-windows-q8-row-supervision-v1.py', 'collect-windows-q8-row-v1.py',
    'publish-windows-q8-row-v1.py', 'test-q8-row-receiver-controls-v1.py']}
parents = {}; prepared = {}
for old, new in files.items():
    wanted = manifest['receiver/' + old]
    assert desc(parent / 'receiver' / old) == desc(cache / old) == wanted
    parents[old] = wanted
    raw = (parent / 'receiver' / old).read_text()
    for before, after in [
            ('q8-row','q8-mx2'), ('q8_row','q8_mx2'),
            ('2dbf3a008bde6ad405dccd33bb7f015da1901b46','ebdf051a1be03df766a28e03e901e7dc3513d500'),
            ('39070438530','39075328381'), ('117506743687','117521191434'),
            ("'row1'","'mx2'"), ("else 'ROW'","else 'MX2'"),
            ('GEN2_Q8_ROW_EXPECTED','GEN2_Q8_MX2_EXPECTED'),
            ('sgemm-row1.cpp','sgemm-mx2.cpp'), ('disasm-row1','disasm-mx2'),
            ('actual_row_control_tests','actual_mx2_control_tests'),
            ('One-row source','Packed-B-scale source'),
            ('failed one-row experiment evidence','failed packed-B-scale experiment evidence'),
            ("'bytes': 866554","'bytes': 878017"),
            ('8bab9e5a60b3a1da49de1fadb4cb7050ca7eccab6a8827984c741c558af2152e',
             '5545fc259ba5ad4992517e5b6876cec3328e88ad9a32d839f75fe53a6a9b55cd')]:
        raw = raw.replace(before, after)
    raw = re.sub(r'\b140\b','149',raw)
    raw = re.sub(r'\b3758\b','3802',raw)
    assert '2dbf3a00' not in raw and '39070438530' not in raw and "'row1'" not in raw
    assert 'q8-row' not in raw and 'q8_row' not in raw
    if new == 'admit-windows-q8-mx2-delivery-v1.py':
        assert raw.count("'actual_mx2_control_tests':6") == 1
    compile(raw,new,'exec')
    prepared[new] = raw
assert all(not (cache / name).exists() for name in prepared)
outputs = {}
for name, raw in prepared.items():
    with (cache / name).open('x') as stream: stream.write(raw)
    outputs[name] = desc(cache / name)
report = {'parents':parents,'outputs':outputs,'preparer':desc(Path(__file__)),
    'run_id':39075328381,'job_id':117521191434,
    'source_commit':'ebdf051a1be03df766a28e03e901e7dc3513d500',
    'receivers_invoked':False,'artifact_consumed':False,'new_native_or_model_calls':0}
with (cache / 'windows-q8-mx2-receivers-preparation-v1.json').open('x') as stream:
    stream.write(json.dumps(report,indent=2)+'\n')
print(json.dumps({'parents_verified':len(parents),'receivers_prepared':len(outputs),'artifact_consumed':False}))
