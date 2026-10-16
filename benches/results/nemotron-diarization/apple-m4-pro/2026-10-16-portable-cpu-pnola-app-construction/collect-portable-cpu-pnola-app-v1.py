"""Collect model-free app construction; compare bytes without loading code."""
import ast,hashlib,json,plistlib,shutil,struct,sys
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from sdk_archive import inventory
source=cache/'portable-cpu-pnola-app-build-v1';p=json.loads((source/'protocol.json').read_text());r=json.loads((source/'build.json').read_text())
assert r['app_construction_passed'] is True and r['native_version_queries_or_neural_calls']==0 and r['release_qualified'] is False
assert descriptor(cache/'build-portable-cpu-pnola-app-v1.py')==p['launcher']
for name,value in p['tools'].items():assert descriptor(tools/name)==value
for name,value in p['references'].items():assert descriptor(Path(name))==value
old=Path(p['original_app']);app=Path(r['app']);package=Path(p['portable_package'])
assert inventory(old)==p['original_inventory'] and inventory(package)==p['portable_inventory'] and inventory(app)==r['inventory']
assert r['source_commit']==p['source_commit']=='7db170d0c8f17b3aea670769577d2d3cd2deaf5b'
assert r['unsigned_executable_payload_unchanged'] and r['original_hardened_entitlements_preserved']
expected_names=['original-entitlements','original-signature','clone',*['framework-'+str(i) for i in range(93)],'bundle-sign','bundle-verify','new-entitlements','new-signature']
assert [c['name'] for c in r['commands']]==expected_names
for c in r['commands']:
 assert c['returncode']==0 and not c.get('forced_process_group_termination') and not c.get('disk_pressure_abort_free_bytes')
 assert c['argv'][:3]==['/usr/bin/sandbox-exec','-f',str(source/'construction.sb')]
 assert c['cwd']==str(source)
assert (source/'construction.sb').read_text()=='(version 1)\n(allow default)\n(deny network*)\n'
entitlements={'com.apple.security.cs.disable-library-validation':True,'com.apple.security.device.audio-input':True}
for name in ['original-entitlements.stdout.log','new-entitlements.stdout.log','Entitlements.plist']:
 assert plistlib.loads((source/name).read_bytes())==entitlements
for name in ['original-signature.stderr.log','new-signature.stderr.log']:
 assert 'runtime' in (source/name).read_text() and 'Signature=adhoc' in (source/name).read_text()
assert r['storage']['released'] and r['storage']['minimum_observed_free_bytes']>=p['storage']['running_minimum']
assert r['storage']['reserve_reported_allocated_bytes']>=p['storage']['reserve_bytes']
static=json.loads((source/'static-verification.json').read_text());assert descriptor(source/'static-verification.json')==r['static_verification']
manifest=json.loads((package/'package.json').read_text());assert static['backend']==manifest['backend']=='cpu'
assert static['logical_runtime_library_count']==len(manifest['libraries'])==92
assert static['release_qualified'] is False and static['app_structure_verified'] is True
expected=set(manifest['libraries'])|{'libnemo_speech_asr_c.dylib','libonnxruntime.dylib'}
frameworks=app/'Contents/Frameworks';assert set(f.name for f in frameworks.iterdir())==expected==set(static['frameworks'])
# Normalize only signature extent fields, then compare every other file byte.
# This does not execute codesign, otool, a loader or any native function.
def payload(path):
 raw=bytearray(path.read_bytes());magic,cpu,_,kind,count,command_bytes,_,_=struct.unpack_from('<8I',raw)
 assert magic==0xfeedfacf and cpu==0x100000c and kind in {2,6}
 cursor=32;end=32+command_bytes;signature=[];linkedit=[]
 assert end<=len(raw)
 for _ in range(count):
  command,size=struct.unpack_from('<II',raw,cursor);assert size>=8 and cursor+size<=end
  if command==0x1d:
   assert size==16;offset,length=struct.unpack_from('<II',raw,cursor+8);signature.append((cursor,offset,length))
  if command==0x19 and raw[cursor+8:cursor+24].rstrip(b'\0')==b'__LINKEDIT':
   assert size>=72;fileoff,filesize=struct.unpack_from('<QQ',raw,cursor+40);linkedit.append((cursor,fileoff,filesize))
  cursor+=size
 assert cursor==end and len(signature)==len(linkedit)==1
 sig,offset,length=signature[0];seg,fileoff,filesize=linkedit[0]
 assert end<=fileoff<=offset and offset+length==len(raw) and fileoff+filesize==len(raw)
 unsigned_length=offset-fileoff
 struct.pack_into('<I',raw,sig+12,0)
 struct.pack_into('<Q',raw,seg+48,unsigned_length)
 struct.pack_into('<Q',raw,seg+32,(unsigned_length+16383)//16384*16384)
 return bytes(raw[:offset])
def equal_payload(a,b):
 left,right=payload(a),payload(b);assert left==right,(a,b)
 return {'bytes_before_signature':len(left),'normalized_sha256':hashlib.sha256(left).hexdigest()}
proof={'main':equal_payload(old/'Contents/MacOS/pnola-desktop',app/'Contents/MacOS/pnola-desktop'),'frameworks':{}}
for name in sorted(expected):
 path=frameworks/name;assert descriptor(path)==static['frameworks'][name]
 original=app/'Contents/Resources/runtime'/name if name=='libonnxruntime.dylib' else package/'lib'/name
 proof['frameworks'][name]=equal_payload(path,original)
assert descriptor(frameworks/'libonnxruntime.dylib')==descriptor(old/'Contents/Frameworks/libonnxruntime.dylib')
changed={n for n in set(r['inventory'])|set(p['original_inventory']) if r['inventory'].get(n)!=p['original_inventory'].get(n)}
assert sorted(changed)==r['changed_inventory_entries']
assert all(n.startswith(('Contents/Resources/nemo/','Contents/Frameworks/','Contents/_CodeSignature/')) or n=='Contents/MacOS/pnola-desktop' for n in changed)
assert inventory(app/'Contents/Resources/nemo')==p['portable_inventory']
record={'schema_version':1,'source_commit':r['source_commit'],'model_free_app_construction_passed':True,'release_qualified':False,'models_or_app_executed':False,'commands_passed':100,'app_inventory_entries':len(r['inventory']),'changed_entries':len(changed),'frameworks_byte_verified_except_signature_extents':94,'old_main_payload_byte_equal_except_signature_extents':True,'signature_extent_normalization':'Only LC_CODE_SIGNATURE.datasize and __LINKEDIT.filesize/vmsize; signature payload removed; all remaining bytes must match.','hardened_adhoc_entitlements_preserved':entitlements,'storage':r['storage'],'limits':['Constructed locally on M4 from exact prior complete CPU app and qualified M1 portable package.','No fresh-machine app execution, inference, GUI, microphone, uncontended performance, accuracy, Developer ID, notarization or release qualification.','Opaque app and model bytes remain in owned cache; this receipt retains hashes, protocol, raw static/signing logs and byte comparison evidence only.','Collection executes only Python byte/hash/plist comparisons. No codesign, otool, loader, model, app or native version function is executed.']}
out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-portable-cpu-pnola-app-construction';assert not out.exists();out.mkdir()
for f in source.iterdir():
 if f.is_file():shutil.copy2(f,out/f.name)
for script in ['build-portable-cpu-pnola-app-v1.py','collect-portable-cpu-pnola-app-v1.py']:
 shutil.copy2(cache/script,out/script)
(out/'payload-byte-comparison.json').write_text(json.dumps(proof,indent=2)+'\n');(out/'qualification.json').write_text(json.dumps(record,indent=2)+'\n')
files={f.name:descriptor(f) for f in out.iterdir() if f.is_file()};(out/'collection.json').write_text(json.dumps({'files':files},indent=2)+'\n')
print(json.dumps({'members':len(files)+1,'output':str(out),'passed':True}))
