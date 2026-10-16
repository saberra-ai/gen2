"""Collect model-free construction/relocation metadata; omit native binaries."""
import hashlib,json,pathlib,shutil,sys
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');src=cache/'valid-mel-metal-macos-package-v1';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-valid-mel-metal-macos-package'
def rec(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def snapshot(root):return {str(p.relative_to(root)):{**rec(p),'symlink_target':str(p.readlink())if p.is_symlink()else None}for p in sorted(root.rglob('*'))if p.is_file()}
r=json.loads((src/'result.json').read_text());pins=json.loads((src/'source-pins.json').read_text());assert r['neural_calls']==0 and not r['native_library_loaded']and not r['production_qualified']and len(r['commands'])==2 and all(c['returncode']==0 for c in r['commands'])
assert r['source_commit']==pins['source_commit']=='1ab02d21006d699286aacce2259f52abd5564e4e'
package,relocated=src/'package',src/'relocated/nemo';files=snapshot(package);assert files==snapshot(relocated)
m=json.loads((package/'package.json').read_text());assert rec(package/'package.json')==r['package_manifest']and len(m['libraries'])==93 and sum(v['sdk_library']is not None for v in m['libraries'].values())==7
sys.path.insert(0,str(repo/'tools/nemotron-diarization'));from package_macos import verify_files,verify_package_provenance
verify_files(package,m['files']);verify_package_provenance(package,m)
assert snapshot(cache/'sdk-valid-mel-tagged-metal-v1')==pins['sdk_before']
assert rec(cache/'valid-mel-tagged-metal-build-v1/build/build.json')==pins['build_receipt']
assert not out.exists();out.mkdir()
for p in src.iterdir():
 if p.is_file():shutil.copy2(p,out/p.name)
for p in package.rglob('*'):
 if p.is_file()and not p.is_relative_to(package/'lib'):
  dest=out/'package'/p.relative_to(package);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
(out/'package-files.json').write_text(json.dumps(files,indent=2)+'\n')
shutil.copy2(__file__,out/pathlib.Path(__file__).name)
q={**r,'test_limits':'Construction and saved-byte verification only. No native binaries, model weights or audio are included in this receipt.'}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');members={str(p.relative_to(out)):rec(p)for p in sorted(out.rglob('*'))if p.is_file()};(out/'collection.json').write_text(json.dumps({'files':members},indent=2)+'\n');print(json.dumps({'members':len(members),'libraries':93,'sdk_libraries':7,'neural_calls':0}))
