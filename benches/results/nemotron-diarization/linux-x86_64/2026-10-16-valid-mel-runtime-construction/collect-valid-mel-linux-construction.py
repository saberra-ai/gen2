"""Collect existing package construction and restore bytes only; no native code."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
repo=Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');tools=repo/'tools/nemotron-diarization';sys.path.insert(0,str(tools))
from build_valid_mel_candidate import descriptor
from sdk_archive import inventory,restore_archive
from platform_sdk_receipt import verify_provenance
from package_linux import check_source_bundle,validate_sdk_library_mapping
root=cache/'valid-mel-linux-package-ci-v1/linux-valid-mel-construction';construction=root/'construction';target=repo/'benches/results/nemotron-diarization/linux-x86_64/2026-10-16-valid-mel-runtime-construction';assert not target.exists()
q=json.loads((root/'qualification.json').read_text());r=json.loads((construction/'receipt.json').read_text());m=json.loads((construction/'package.json').read_text());p=json.loads((root/'protocol.json').read_text());f=json.loads((root/'frozen-inputs.json').read_text());ci=json.loads((cache/'valid-mel-linux-package-run.json').read_text())
assert ci['databaseId']==38942471178 and ci['status']=='completed' and ci['conclusion']=='success' and len(ci['jobs'])==1 and ci['jobs'][0]['conclusion']=='success'
assert ci['headSha']==q['source_commit']==r['source_commit']==f['source_commit']=='054cac64bd3fb6ae286bd35245d35bee3ab04115'
assert q['package_construction_passed'] is True and q['new_native_model_or_neural_calls']==0 and r['package_construction_passed'] is True and q['command']['returncode']==0
assert descriptor(root/'protocol.json')==q['protocol'] and (root/'protocol.json').read_bytes()==(tools/'valid-mel-linux-construction.protocol.json').read_bytes()
assert descriptor(construction/'receipt.json')==q['construction_receipt'] and descriptor(root/'native-receipt.json')==q['native_receipt']==p['native_receipt']
assert descriptor(construction/'runtime/package.json')['sha256']==r['package_manifest_sha256'] and m==json.loads((construction/'runtime/package.json').read_text())
for records,prefix in [(p['tools'],'tools/nemotron-diarization/'),(p['repository_files'],'')]:
 for name,expected in records.items():
  data=subprocess.check_output(['git','show',ci['headSha']+':'+prefix+name],cwd=repo)
  assert {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==expected
archive=json.loads((construction/'development-runtime-files.json').read_text());assert {k:v for k,v in archive.items()if k!='files'}==r['archive']==q['archive']
delivered=cache/'valid-mel-linux-delivered-runtime-v1';restore_archive(construction/'development-runtime.tar.gz',archive,delivered)
assert inventory(delivered)==archive['files']
actual=inventory(delivered);actual.pop('package.json');assert actual==m['files']
build=verify_provenance(delivered,m);validate_sdk_library_mapping(m,build)
assert descriptor(delivered/'provenance/native-receipt.json')==p['native_receipt']
components={}
for name,record in m['components'].items():
 b=record['source_bundle'];assert check_source_bundle(delivered/'sources'/b['source_package'],b['source_package'],b['source_version'])==b
 assert descriptor(delivered/'notices'/name/'copyright')['sha256']==record['copyright_sha256'];components[name]=b
sdk=cache/'valid-mel-platform-ci-v1/restored-linux-sdk'
for path in (sdk/'share/licenses/nemo-speech').rglob('*'):
 if path.is_file():assert path.read_bytes()==(delivered/'notices/nemo-speech'/path.relative_to(sdk/'share/licenses/nemo-speech')).read_bytes()
assert len(m['libraries'])==8 and m['format']=='gen2-nemo-linux-cpu-v2' and m['runtime_profile']=='gen2-valid-mel-v1'
assert m['entry_library']=='libgen2_nemo_linux_nemo_speech_asr_c.so.1' and (delivered/'lib/libnemo_speech_asr_c.so').readlink().as_posix()==m['entry_library']
target.mkdir(parents=True)
for rel in ['qualification.json','protocol.json','frozen-inputs.json','command.json','construct.stdout.log','construct.stderr.log','producer-run.json','native-receipt.json','construction/receipt.json','construction/package.json','construction/commands.json','construction/package-build.log','construction/development-runtime-files.json']:
 dest=target/'raw'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,dest)
for path in sorted(delivered.rglob('*')):
 if not path.is_file() or path.is_symlink() or path.is_relative_to(delivered/'lib'):continue
 if path.is_relative_to(delivered/'sources') and path.suffix!='.dsc':continue
 dest=target/'delivered-metadata'/path.relative_to(delivered);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
for name in ['valid-mel-linux-package-run.json','valid-mel-linux-package-artifacts.json']:shutil.copy2(cache/name,target/name)
execution=target/'execution-tools';execution.mkdir()
for name,expected in p['tools'].items():
 assert descriptor(tools/name)==expected
 dest=execution/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(tools/name,dest)
shutil.copy2(Path(__file__),target/Path(__file__).name)
record={'schema_version':1,'release_qualified':False,'package_construction_passed':True,'scope':'Actual Linux v2 construction plus receiver-only archive bytes/modes/aliases, complete SDK mapping, provenance and corresponding source/notice verification. No Linux or native code executed during collection.','source_commit':ci['headSha'],'run_id':ci['databaseId'],'ci_url':ci['url'],'protocol':q['protocol'],'native_receipt':p['native_receipt'],'constructor_returncode':0,'constructor_elapsed_seconds':q['command']['elapsed_seconds'],'runtime_archive':r['archive'],'runtime_manifest':descriptor(delivered/'package.json'),'libraries':list(m['libraries']),'package_inventory_entries_excluding_manifest':len(m['files']),'independently_restored_inventory_matches':True,'complete_sdk_library_source_mapping_matches':True,'packaged_provenance_matches':True,'native_notices_match_qualified_sdk':True,'source_bundles_rechecked':components,'new_native_model_or_neural_calls':0,'limits':['Actual ELF introspection/closure checks happened in the named Linux CI constructor; receiver rechecks bytes, filesystem and provenance without executing Linux tools.','Opaque runtime/library/source archives retained in CI/cache, omitted from Git with exact descriptors preserved.','Offline native/public/lifecycle, CPU chat/Bark/Laya coexistence, original AMI quality, desktop and release gates remain open.'],'files':{str(path.relative_to(target)):descriptor(path)for path in sorted(target.rglob('*'))if path.is_file()}}
(target/'qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'target':str(target),'members':len(record['files'])+1,'libraries':len(m['libraries']),'archive_bytes':r['archive']['archive_bytes']}))
