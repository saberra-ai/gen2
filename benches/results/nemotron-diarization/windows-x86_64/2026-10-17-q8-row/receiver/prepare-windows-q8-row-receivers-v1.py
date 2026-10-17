"""Prepare saved-only one-row receivers from retained, admitted parent sources."""
import hashlib,json,re
from pathlib import Path
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0')
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0')
result=repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-pair'
def desc(path):
    raw=path.read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
assert desc(result/'collection.json')=={'bytes':212709,'sha256':'a4ed4a36afcfeac94f4ba05b68c1bf11584eb80fd21adc21a557e1b0f5827aed'}
manifest=json.loads((result/'collection.json').read_text())['files']
files={
 'fetch-windows-q8-pair-artifact-v1.py':'fetch-windows-q8-row-artifact-v1.py',
 'admit-windows-q8-pair-delivery-v1.py':'admit-windows-q8-row-delivery-v1.py',
 'replay-windows-q8-pair-supervision-v1.py':'replay-windows-q8-row-supervision-v1.py',
 'collect-windows-q8-pair-v2.py':'collect-windows-q8-row-v1.py',
 'publish-windows-q8-pair-v2.py':'publish-windows-q8-row-v1.py',
 'test-q8-pair-receiver-controls-v1.py':'test-q8-row-receiver-controls-v1.py'}
parents={}
for old,new in files.items():
    wanted=manifest['receiver/'+old]
    assert desc(cache/old)==desc(result/'receiver'/old)==wanted
    parents[old]=wanted

def derive(raw):
    for old,new in files.items():raw=raw.replace(old,new)
    for old,new in [
        ('q8-pair','q8-row'),('q8_pair','q8_row'),
        ('259966ceca113f50d7d172fa63269b0bcb17b37a','2dbf3a008bde6ad405dccd33bb7f015da1901b46'),
        ('39066832336','39070438530'),('117496146656','117506743687'),
        ("'pair'","'row1'"),('GEN2_Q8_PAIR_EXPECTED','GEN2_Q8_ROW_EXPECTED'),
        ("else 'PAIR'", "else 'ROW'"),('sgemm-pair.cpp','sgemm-row1.cpp'),('disasm-pair','disasm-row1'),
        ('actual_pair_control_tests','actual_row_control_tests'),('Two-row source','One-row source'),
        ('failed two-row experiment evidence','failed one-row experiment evidence'),
        ('collection-replay-v2.json','collection-replay-v1.json'),
        ("'bytes': 862413", "'bytes': 866554"),
        ('865d607acc7de3d9306a7874ca657ddfec1ab894c9f50558e642cff7cbcb9959','8bab9e5a60b3a1da49de1fadb4cb7050ca7eccab6a8827984c741c558af2152e')]:raw=raw.replace(old,new)
    raw=re.sub(r'\b131\b','140',raw)
    raw=re.sub(r'\b3745\b','3758',raw)
    return raw
outputs={}
for old,new in files.items():
    raw=derive((cache/old).read_text())
    if new=='publish-windows-q8-row-v1.py':
        old_list="""for name in ['fetch-windows-q8-row-artifact-v1.py','windows-q8-row-receivers-pre-execution-v1.json','windows-q8-row-receivers-pre-execution-v2.json',
        'prepare-windows-q8-row-receivers-v1.py','prepare-windows-q8-row-receivers-v2.py','test-q8-row-receiver-controls-v1.py','q8-row-receiver-controls-local-v1.json']:"""
        new_list="""for name in ['fetch-windows-q8-row-artifact-v1.py','windows-q8-row-receivers-pre-execution-v1.json','windows-q8-row-receivers-preparation-v1.json',
        'prepare-windows-q8-row-receivers-v1.py','test-q8-row-receiver-controls-v1.py','q8-row-receiver-controls-local-v1.json']:"""
        assert raw.count(old_list)==1
        raw=raw.replace(old_list,new_list,1)
    assert '259966ce' not in raw and '39066832336' not in raw and "'pair'" not in raw and 'q8-pair' not in raw and 'q8_pair' not in raw
    compile(raw,new,'exec')
    with (cache/new).open('x') as stream:stream.write(raw)
    outputs[new]=desc(cache/new)
r={'parents':parents,'outputs':outputs,'preparer':desc(Path(__file__)),'run_id':39070438530,'job_id':117506743687,'source_commit':'2dbf3a008bde6ad405dccd33bb7f015da1901b46','receivers_invoked':False,'new_native_or_model_calls':0}
with (cache/'windows-q8-row-receivers-preparation-v1.json').open('x') as stream:stream.write(json.dumps(r,indent=2)+'\n')
print(json.dumps({'parents_verified':len(parents),'receivers_prepared':len(outputs),'actual_artifact_consumed':False}))
