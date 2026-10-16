import hashlib,json,pathlib,shutil,zipfile
repo=pathlib.Path('/Users/victor/workspace/gen2-nemotron3-n0');cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');work=cache/'original-nemo-env-v1';source=work/'Speech-218d0b443441b7d8aa1c832f47df310f186cf0a4';out=repo/'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-original-nemo-environment'
def desc(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
preparation=json.loads((work/'preparation.json').read_text());assert desc(cache/'research/original-nemo-tail/speech-218d0b4.tar.gz')==preparation['archive'];assert desc(work/'source-inventory.json')==preparation['source_inventory']
files=json.loads((work/'source-inventory.json').read_text());assert len(files)==2606
for name,expected in files.items():assert desc(source/name)==expected,name
for name,target in preparation['internal_source_symlinks'].items():assert str((source/name).readlink())==target
wheel=work/'wheels/nemo_toolkit-3.1.0-py3-none-any.whl';mapping=json.loads((work/'wheel-source-mapping.json').read_text());assert len(mapping)==774
with zipfile.ZipFile(wheel)as z:
 assert {i.filename for i in z.infolist()if i.filename.startswith('nemo/')and not i.is_dir()}==set(mapping)
 for name,expected in mapping.items():
  b=z.read(name);assert expected=={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} and desc(source/name)==expected and desc(work/'venv/lib/python3.12/site-packages'/name)==expected
install=json.loads((work/'installation.json').read_text());assert install['passed'] is True and len(install['commands'])==4 and all(c['returncode']==0 for c in install['commands'])and install['storage']['released'] is True
for path,expected in install['inputs'].items():assert desc(pathlib.Path(path))==expected
command=json.loads((work/'import-command.json').read_text());result=json.loads((work/'import-smoke-result.json').read_text());assert command['returncode']==0 and command['passed'] is True and command['model_execution_authorized'] is False and command['storage']['released'] is True
assert result['passed']is True and result['source_files_verified']==774 and all(result[k]is False for k in ['model_constructed','model_restored','model_executed','frontend_executed','torch_cuda_available'])
assert result['packages']['torch']=='2.8.0' and result['packages']['numpy']=='2.2.6' and result['packages']['nemo-toolkit']=='3.1.0'
assert 'All installed packages are compatible' in (work/'check.stdout.log').read_text()+(work/'check.stderr.log').read_text()
assert not out.exists();out.mkdir()
for p in work.iterdir():
 if p.is_file()and p.suffix in ['.json','.log','.lock','.in','.sb','.py']:shutil.copy2(p,out/p.name)
for name in ['prepare-original-nemo-env-v1.py','install-original-nemo-env-v1.py','run-original-nemo-import-v1.py','collect-original-nemo-env-v1.py']:shutil.copy2(cache/name,out/name)
q={'schema_version':1,'source_commit':'218d0b443441b7d8aa1c832f47df310f186cf0a4','original_source_files_verified':len(files),'wheel_and_installed_source_files_equal':len(mapping),'wheel_outside_git':desc(wheel),'resolved_hash_locked_environment_installed':True,'model_free_class_import_passed':True,'torch_cpu_available':True,'actual_cuda_available':False,'model_or_frontend_execution':False,'original_model_parity_qualified':False,'release_qualified':False,'limits':['Independent NeMo3.1.0 source at the pinned later official V3 commit, not a reproduction of the checkpoint training environment (declared3.0.0).','The actual142-package macOS/Python3.12 resolution pins Torch2.8.0 and NumPy2.2.6; this is a separately resolved diagnostic environment, not the upstream complete uv.lock resolution.','Four environment construction/check commands and one offline model-class/frontend-class import only. No checkpoint restore, model/frontend construction or inference.','Imports run with network and weight-format reads denied, private HOME/TMP, and source/environment writes denied. Normal CLI test success alone is not app, quality or performance qualification.','The first archive policy rejected three known internal relative symlinks before extraction. Those exact three links were then explicitly admitted and retained.','Source archive, wheel, installed environment and dependencies remain in cache; their hashes and original source mapping are retained here. Production Rust/native runtime and original HF reference environment were not changed.']}
(out/'qualification.json').write_text(json.dumps(q,indent=2)+'\n');records={p.name:desc(p)for p in out.iterdir()if p.is_file()};(out/'collection.json').write_text(json.dumps({'files':records},indent=2)+'\n');print(json.dumps({'members':len(records)+1,'wheel_files':len(mapping),'original_source_files':len(files)}))
