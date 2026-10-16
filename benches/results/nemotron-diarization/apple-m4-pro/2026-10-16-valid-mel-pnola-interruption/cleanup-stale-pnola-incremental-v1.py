"""Remove old Gen2/Pnola incremental compiler caches only; preserve final artifacts."""
import json,os,shutil,subprocess,time
from pathlib import Path
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');roots=[Path('/Users/victor/.cache/pnola-nemotron-build/debug/incremental'),Path('/Users/victor/.cache/gen2-bark-onnx/build/debug/incremental')];report=cache/'pnola-incremental-cleanup-v1.json';assert not report.exists()
record={'scope':__doc__,'free_before':shutil.disk_usage(cache).free,'removed':[],'stopped_for_active_compiler':False,'models_evidence_sdks_executables_and_archives_preserved':True}
for root in roots:
 for path in sorted(root.iterdir()):
  if not path.name.startswith(('gen2-','pnola_desktop-'))or path.is_symlink()or not path.is_dir():continue
  items=list(path.rglob('*'))
  if any(p.is_symlink()for p in items):continue
  files=[p for p in items if p.is_file()]
  if not files or time.time()-max(p.stat().st_mtime for p in files)<3600:continue
  if any(subprocess.run(['pgrep','-x',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0 for name in ['cargo','rustc']):record['stopped_for_active_compiler']=True;break
  row={'path':str(path),'files':[{'path':str(f.relative_to(path)),'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns,'inode':f.stat().st_ino,'links':f.stat().st_nlink}for f in files]}
  row['apparent_bytes']=sum(f['bytes']for f in row['files']);record['removed'].append(row)
  report.write_text(json.dumps(record,indent=2)+'\n');shutil.rmtree(path)
 if record['stopped_for_active_compiler']:break
record['free_after']=shutil.disk_usage(cache).free;record['removed_directories']=len(record['removed']);record['removed_apparent_bytes']=sum(r['apparent_bytes']for r in record['removed']);report.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items()if k!='removed'}))
