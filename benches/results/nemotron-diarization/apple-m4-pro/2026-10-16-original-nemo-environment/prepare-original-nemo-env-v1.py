import hashlib,json,pathlib,shutil,tarfile
cache=pathlib.Path('/Users/victor/.cache/gen2-nemotron3-n0');research=cache/'research/original-nemo-tail';out=cache/'original-nemo-env-v1';assert not any(out.iterdir())
(out/'preparation-first-admission-failure.json').write_text(json.dumps({'reason':'archive contains three relative internal symlinks; initial no-symlinks policy rejected before extraction','model_imported':False,'model_executed':False},indent=2)+'\n')
archive=research/'speech-218d0b4.tar.gz';rootname='Speech-218d0b443441b7d8aa1c832f47df310f186cf0a4'
def record(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
free=shutil.disk_usage(out).free
assert free>=5*1024**3
with tarfile.open(archive,'r:gz')as t:
 members=t.getmembers();assert all(pathlib.PurePosixPath(m.name).parts[0]==rootname and '..'not in pathlib.PurePosixPath(m.name).parts and not(m.islnk()or m.isdev())for m in members)
 assert {(m.name.removeprefix(rootname+'/'),m.linkname)for m in members if m.issym()}=={('.codex/skills','../.claude/skills'),('.cursor/skills','../.claude/skills'),('CLAUDE.md','AGENTS.md')}
 assert sum(m.size for m in members)<1024**3
 t.extractall(out,filter='data')
source=out/rootname
for p in research.iterdir():
 if p.name.startswith('nemo__')or p.name=='pyproject.toml':
  relative=p.name.replace('__','/');assert (source/relative).read_bytes()==p.read_bytes(),relative
pins={p.relative_to(source).as_posix():record(p)for p in sorted(source.rglob('*'))if p.is_file()and not p.is_symlink()}
(out/'source-inventory.json').write_text(json.dumps(pins,indent=2)+'\n')
(out/'requirements.in').write_text(f'nemo-toolkit[asr] @ {source.as_uri()}\ntorch==2.8.0\nnumpy==2.2.6\n')
result={'source_commit':'218d0b443441b7d8aa1c832f47df310f186cf0a4','source_archive_url':'https://codeload.github.com/NVIDIA-NeMo/Speech/tar.gz/218d0b443441b7d8aa1c832f47df310f186cf0a4','archive':record(archive),'source_members':len(pins),'internal_source_symlinks':{m.name.removeprefix(rootname+'/'):m.linkname for m in members if m.issym()},'source_inventory':record(out/'source-inventory.json'),'retained_source_files_match':True,'free_before':free,'free_after':shutil.disk_usage(out).free,'model_imported':False,'model_executed':False,'existing_reference_environment_modified':False}
(out/'preparation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
