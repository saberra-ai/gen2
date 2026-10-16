"""Construct/relocate/inspect current macOS package once; no model or native load."""
import hashlib,json,pathlib,shutil,subprocess,time
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');out=cache/'valid-mel-macos-package-v1';sdk=cache/'sdk-valid-mel-tagged-cpu-v1';build=cache/'valid-mel-tagged-metadata-v1/recovered-build.json';tools=repo/'tools/nemotron-diarization'
def rec(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def tree(root):return {str(p.relative_to(root)):{**rec(p),'symlink_target':str(p.readlink())if p.is_symlink()else None}for p in sorted(root.rglob('*'))if p.is_file()}
assert not out.exists(),'previous attempt must be preserved'
source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert source.startswith('0b3d0d6')
names=['package_macos.py','runtime_profiles.py','build_valid_mel_candidate.py','artifacts.py','runtime.lock.json','runtime-valid-mel.lock.json','patches/v3-finish-valid-mel.patch']
pins={n:rec(tools/n)for n in names}
for n in names:assert hashlib.sha256(subprocess.check_output(['git','show',source+':tools/nemotron-diarization/'+n],cwd=repo)).hexdigest()==pins[n]['sha256']
before=tree(sdk);metadata=rec(build)
closure=json.loads((cache/'tagged-valid-mel-probes-v1/loader-closure.json').read_text())
# Use the exact closure discovered before the fixed-case qualification.
print('closure_fields',list(closure),flush=True)
out.mkdir();(out/'source-pins.json').write_text(json.dumps({'source_commit':source,'tools':pins,'sdk_before':before,'build_receipt':metadata},indent=2)+'\n');shutil.copy2(__file__,out/pathlib.Path(__file__).name)
profile=out/'construct.sb'
no_read=[cache/'assets',cache/'native-f32-conversion-v1',cache/'tail-framing-v1',cache/'tagged-valid-mel-probes-v1/calibration.wav',cache/'tagged-valid-mel-public-inputs-v1']
profile.write_text('(version 1)\n(allow default)\n(deny network*)\n(deny file-read* '+' '.join('(subpath '+json.dumps(str(p))+')'for p in no_read)+')\n(deny file-write* '+' '.join('(subpath '+json.dumps(str(p))+')'for p in [sdk,cache/'NeMo-Speech-valid-mel-tagged-v1',pathlib.Path('/opt/homebrew')])+')\n')
commands=[];env={'PATH':'/usr/bin:/bin','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'};python=cache/'evidence-tools-env/bin/python'
def run(name,args,sb=profile):
 argv=['/usr/bin/sandbox-exec','-f',str(sb),str(python),str(tools/'package_macos.py'),*map(str,args)]
 record={'name':name,'argv':argv,'environment':env};commands.append(record);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');start=time.monotonic()
 with(out/(name+'.stdout.log')).open('xb')as stdout,(out/(name+'.stderr.log')).open('xb')as stderr:r=subprocess.run(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr)
 record.update(returncode=r.returncode,elapsed_seconds=time.monotonic()-start);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');print(json.dumps(record),flush=True);r.check_returncode()
run('construct',['--sdk-prefix',sdk,'--build-receipt',build,'--output',out/'package'])
shutil.copytree(out/'package',out/'relocated/nemo',symlinks=True)
# Deny original binary identities and source SDK during relocated structural verification.
manifest=json.loads((out/'package/package.json').read_text());denied=[sdk,cache/'NeMo-Speech-valid-mel-tagged-v1',out/'package',*[pathlib.Path(r['source'])for r in manifest['libraries'].values()]]
relocated=out/'relocated-verify.sb';relocated.write_text(profile.read_text()+'(deny file-read* '+' '.join('(subpath '+json.dumps(str(p))+')'for p in denied)+')\n')
run('relocated-verify',['--verify',out/'relocated/nemo'],relocated)
assert tree(out/'package')==tree(out/'relocated/nemo')
assert tree(sdk)==before and rec(build)==metadata
for n,pin in pins.items():assert rec(tools/n)==pin
q={'source_commit':source,'package_format':manifest['format'],'runtime_profile':manifest['runtime_profile'],'package_manifest':rec(out/'package/package.json'),'libraries':len(manifest['libraries']),'qualified_sdk_libraries':sum(r['sdk_library']is not None for r in manifest['libraries'].values()),'commands':commands,'sdk_unchanged':True,'relocated_package_bytes_and_aliases_identical':True,'relocated_structural_verification_with_original_libraries_denied':True,'neural_calls':0,'native_library_loaded':False,'production_qualified':False,'release_qualified':False,'limits':['Construction and offline structural/signature/provenance verification only. No native version query, model execution, actual loaded-path telemetry, lifecycle/coexistence, desktop or release qualification.']}
(out/'result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q),flush=True)
