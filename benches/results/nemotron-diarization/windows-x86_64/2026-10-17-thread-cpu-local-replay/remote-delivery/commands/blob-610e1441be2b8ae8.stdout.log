"""Derive saved-only A-packing receivers from the admitted Mx2 receipt."""
import hashlib
import json
from pathlib import Path
import re

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
parent = repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-mx2'
source = '3fd41caa862c34134d008d18b0b6b830b736c864'


def desc(path):
    raw=path.read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def once(raw, old, new):
    assert raw.count(old)==1,old
    return raw.replace(old,new,1)


protocol_path=repo/'tools/nemotron-diarization/windows-q8-apack.protocol.json'
assert desc(protocol_path)=={'bytes':885307,'sha256':'329e0afea3c8e4f05d6158e58e801519a0dfb3d76fa0e1b27afa9d00eb7e5c2f'}
protocol=json.loads(protocol_path.read_bytes())
parents_path=repo/'tools/nemotron-diarization/q8-apack-parent-identities.json'
assert desc(parents_path)==protocol['tools'][parents_path.name]
expected=json.loads(parents_path.read_bytes())
assert desc(parent/'collection.json')==expected[str((parent/'collection.json').relative_to(repo))]
manifest=json.loads((parent/'collection.json').read_bytes())['files']
files={name:name.replace('q8-mx2','q8-apack') for name in [
    'fetch-windows-q8-mx2-artifact-v1.py','admit-windows-q8-mx2-delivery-v1.py',
    'replay-windows-q8-mx2-supervision-v1.py','collect-windows-q8-mx2-v1.py',
    'test-q8-mx2-receiver-controls-v1.py']}
files['publish-windows-q8-mx2-v2.py']='publish-windows-q8-apack-v1.py'
parents={};prepared={}
for old,new in files.items():
    wanted=manifest['receiver/'+old]
    path=parent/'receiver'/old
    assert desc(path)==desc(cache/old)==wanted
    parents[old]=wanted
    raw=path.read_text()
    for before,after in [
        ('q8-mx2','q8-apack'),('q8_mx2','q8_apack'),
        ('ebdf051a1be03df766a28e03e901e7dc3513d500',source),
        ('39075328381','39077373013'),('117521191434','117527216794'),
        ("'mx2'","'apack'"),("else 'MX2'","else 'APACK'"),
        ('GEN2_Q8_MX2_EXPECTED','GEN2_Q8_APACK_EXPECTED'),
        ('sgemm-mx2.cpp','sgemm-apack.cpp'),('disasm-mx2','disasm-apack'),
        ('actual_mx2_control_tests','actual_apack_control_tests'),
        ('Packed-B-scale source','A packing source'),
        ('failed packed-B-scale experiment evidence','failed A packing experiment evidence'),
        ("'bytes': 878017","'bytes': 885307"),
        ('5545fc259ba5ad4992517e5b6876cec3328e88ad9a32d839f75fe53a6a9b55cd',
         '329e0afea3c8e4f05d6158e58e801519a0dfb3d76fa0e1b27afa9d00eb7e5c2f')]:
        raw=raw.replace(before,after)
    raw=re.sub(r'\b149\b','161',raw)
    raw=re.sub(r'\b3802\b','3826',raw)
    if new=='admit-windows-q8-apack-delivery-v1.py':
        raw=once(raw,"log.count('Ran 7 tests') == 2","log.count('Ran 7 tests') == 3")
        raw=once(raw,"log.count('Ran 6 tests') == 3","log.count('Ran 6 tests') == 2")
        raw=once(raw,"'actual_apack_control_tests':6","'actual_apack_control_tests':7, 'actual_apack_portable_control_skipped':1")
        raw=once(raw,"assert file_inventory(delivered) == members\nassert descriptor(root / 'artifact.zip')",
            "assert log.count('OK (skipped=1)') == 1\nassert file_inventory(delivered) == members\nassert descriptor(root / 'artifact.zip')")
    if new=='collect-windows-q8-apack-v1.py':
        raw=once(raw,'from windows_q8_apack_metrics import ',
            'from q8_apack_metrics import admit_packing\nfrom windows_q8_apack_metrics import ')
        marker='identity=desc(Path(__file__))'
        helper='''def admit_packing_case(case, evidence, phase, half_raw):
    payload=(evidence/(phase+'.apack-payload.i8')).read_bytes()
    scales=(evidence/(phase+'.apack-scales.f32')).read_bytes()
    observed=admit_packing(payload,scales)
    assert scales==half_raw and observed==case['packing_control']
    return observed

'''
        raw=once(raw,marker,helper+marker)
        raw=once(raw,"assert half_raw==reference_half", "assert half_raw==reference_half\n        admit_packing_case(case,evidence,phase,half_raw)")
        raw=once(raw,"assert q['all_12_half_tables_bit_exact'] is True and q['all_36_support_rejections_verified'] is True",
            "assert q['all_12_half_tables_bit_exact'] is True and q['all_36_support_rejections_verified'] is True\n    assert q['all_12_packing_controls_verified'] is True")
        raw=once(raw,'result.update(all_1056_outputs_bit_exact=True,',
            'result.update(all_12_packing_controls_verified=True,all_1056_outputs_bit_exact=True,')
    if new=='test-q8-apack-receiver-controls-v1.py':
        raw=once(raw,'import sys\n','import sys\nimport struct\nfrom q8_apack_metrics import admit_packing\n')
        # The existing sys.path admission must precede local packing imports.
        raw=raw.replace('from q8_apack_metrics import admit_packing\n','')
        raw=once(raw,'from q8_apack_source import control_source',
            'from q8_apack_source import control_source\nfrom q8_apack_metrics import admit_packing')
        raw=once(raw,"'admit_kernel_command'}","'admit_kernel_command', 'admit_packing_case'}")
        raw=once(raw,"'control_source': control_source, 're': re}",
            "'control_source': control_source, 're': re, 'admit_packing': admit_packing}")
        marker='    def test_exact_guard_and_case_arguments_admit(self):'
        extra='''    def packing(self, path):
        phase='case-4-0'
        payload=bytes((i+17*j)&255 for i in range(65536) for j in range(32))
        scales=b''.join(struct.pack('<f',struct.unpack('<e',struct.pack('<H',i))[0]) for i in range(65536))
        (path/(phase+'.apack-payload.i8')).write_bytes(payload)
        (path/(phase+'.apack-scales.f32')).write_bytes(scales)
        return phase,payload,scales,{'packing_control':admit_packing(payload,scales)}

    def test_complete_packing_outputs_and_original_lookup_identity_admit(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);phase,payload,scales,case=self.packing(path)
            self.assertEqual(scope['admit_packing_case'](case,path,phase,scales),case['packing_control'])

    def test_packing_mutation_missing_output_metadata_and_lookup_mismatch_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);phase,payload,scales,case=self.packing(path)
            target=path/(phase+'.apack-payload.i8')
            target.write_bytes(payload[:-1])
            with self.assertRaises(ValueError):scope['admit_packing_case'](case,path,phase,scales)
            target.write_bytes(payload)
            target.unlink()
            with self.assertRaises(FileNotFoundError):scope['admit_packing_case'](case,path,phase,scales)
            target.write_bytes(payload)
            changed=copy.deepcopy(case);changed['packing_control']['blocks']=65535
            with self.assertRaises(AssertionError):scope['admit_packing_case'](changed,path,phase,scales)
            with self.assertRaises(AssertionError):scope['admit_packing_case'](case,path,phase,scales[:-4])
            (path/(phase+'.apack-scales.f32')).write_bytes(b'\\0\\0\\x80\\x3f'+scales[4:])
            with self.assertRaises(ValueError):scope['admit_packing_case'](case,path,phase,scales)

'''
        raw=once(raw,marker,extra+marker)
    if new=='publish-windows-q8-apack-v1.py':
        raw=raw.replace('windows-q8-apack-publication-pre-execution-v2.json',
            'windows-q8-apack-publication-pre-execution-v1.json')
        raw=once(raw,"assert replay['timing_comparison']==q['timing_comparison']",
            "assert replay['timing_comparison']==q['timing_comparison']\n    assert replay['all_12_packing_controls_verified'] is q['all_12_packing_controls_verified'] is True")
        raw=once(raw,"'all_12_half_tables_bit_exact':completed,",
            "'all_12_packing_controls_verified':completed,'all_12_half_tables_bit_exact':completed,")
    assert 'ebdf051a' not in raw and '39075328381' not in raw and "'mx2'" not in raw
    assert 'q8-mx2' not in raw and 'q8_mx2' not in raw
    compile(raw,new,'exec')
    prepared[new]=raw
assert all(not (cache/name).exists() for name in prepared)
outputs={}
for name,raw in prepared.items():
    with (cache/name).open('x') as f:f.write(raw)
    outputs[name]=desc(cache/name)
report={'parents':parents,'outputs':outputs,'preparer':desc(Path(__file__)),
    'source_commit':source,'run_id':39077373013,'job_id':117527216794,
    'receivers_invoked':False,'artifact_consumed':False,'new_native_calls':0}
with (cache/'windows-q8-apack-receivers-preparation-v1.json').open('x')as f:
    f.write(json.dumps(report,indent=2)+'\n')
print(json.dumps({'parents_verified':len(parents),'receivers_prepared':len(outputs),'artifact_consumed':False}))
