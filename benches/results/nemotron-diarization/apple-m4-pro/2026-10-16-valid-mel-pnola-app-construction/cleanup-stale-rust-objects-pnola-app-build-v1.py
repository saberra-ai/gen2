"""Remove only old single-link Mach-O Rust compiler object intermediates while idle."""
import json,os,shutil,subprocess,time
from pathlib import Path
cache=Path('/Users/victor/.cache/gen2-nemotron3-n0');base=Path('/Users/victor/.cache/pnola-nemotron-build');report=cache/'compiler-object-cleanup-pnola-app-build-v1.json';journal=cache/'compiler-object-cleanup-pnola-app-build-v1.events.jsonl'
assert not report.exists()and not journal.exists()
rows=[]
for mode in ['debug']:
 root=base/mode/'deps'
 for path in root.glob('*.rcgu.o'):
  stat=path.lstat()
  if path.is_file()and not path.is_symlink()and stat.st_nlink==1 and time.time()-stat.st_mtime>3600:
   with path.open('rb')as stream:header=stream.read(16)
   if len(header)!=16 or header[:4]!=bytes.fromhex('cffaedfe')or int.from_bytes(header[12:16],'little')!=1:continue
   rows.append({'path':str(path),'bytes':stat.st_size,'mtime':stat.st_mtime,'inode':stat.st_ino})
rows.sort(key=lambda r:r['bytes'],reverse=True);record={'scope':__doc__,'free_before':shutil.disk_usage(base).free,'target_free_bytes':7*1024**3,'eligible_files':len(rows),'eligible_bytes':sum(r['bytes']for r in rows),'removed_files':0,'removed_apparent_bytes':0,'stopped_for_active_compiler':False,'preserved_kind':'All models, evidence, SDKs, executables and Rust library archives/metadata; only stale *.rcgu.o intermediates eligible.'}
report.write_text(json.dumps(record,indent=2)+'\n')
with journal.open('x')as events:
 for row in rows:
  if shutil.disk_usage(base).free>=record['target_free_bytes']:break
  if any(subprocess.run(['pgrep','-x',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0 for name in ['cargo','rustc']):record['stopped_for_active_compiler']=True;break
  path=Path(row['path']);stat=path.lstat()
  if path.is_symlink()or stat.st_nlink!=1 or stat.st_ino!=row['inode']or stat.st_size!=row['bytes']or stat.st_mtime!=row['mtime']:continue
  path.unlink();events.write(json.dumps(row)+'\n');record['removed_files']+=1;record['removed_apparent_bytes']+=row['bytes']
  if record['removed_files']%256==0:events.flush();os.fsync(events.fileno())
 events.flush();os.fsync(events.fileno())
record['free_after']=shutil.disk_usage(base).free;report.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
