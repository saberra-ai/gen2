"""Reclaim idle, old Rust incremental caches in the two owned build targets only."""
import json
import shutil
import subprocess
import time
from pathlib import Path

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
roots = [Path('/Users/victor/.cache/gen2-bark-onnx/build/debug/incremental'),
         Path('/Users/victor/.cache/pnola-nemotron-build/debug/incremental')]
report = cache / 'owned-incremental-app-cleanup-v1.json'
assert not report.exists()
record = {'scope': __doc__, 'free_before': shutil.disk_usage(cache).free, 'target_free_bytes': 7 * 1024**3,
          'removed': [], 'stopped_for_active_compiler': False, 'preserved': 'All sources, models, evidence, SDKs, final executables and archives; only stale Rust incremental compiler caches are eligible.'}
def save():
    report.write_text(json.dumps(record, indent=2) + '\n')
def metadata(path):
    s = path.stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino, 'links': s.st_nlink}
candidates = []
for root in roots:
    for path in root.iterdir():
        if path.is_symlink() or not path.is_dir():
            continue
        items = list(path.rglob('*'))
        if any(p.is_symlink() for p in items):
            continue
        files = [p for p in items if p.is_file()]
        if (not files or any(p.suffix not in {'.o', '.bin', '.lock'} for p in files)
                or time.time() - max(p.stat().st_mtime for p in files) < 3600):
            continue
        rows = {str(f.relative_to(path)): metadata(f) for f in files}
        candidates.append((sum(r['bytes'] for r in rows.values()), path, rows))
save()
for size, path, rows in sorted(candidates, key=lambda x: x[0], reverse=True):
    if shutil.disk_usage(cache).free >= record['target_free_bytes']:
        break
    if any(subprocess.run(['pgrep', '-x', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0 for name in ['cargo', 'rustc']):
        record['stopped_for_active_compiler'] = True
        break
    actual = {str(f.relative_to(path)): metadata(f) for f in path.rglob('*') if f.is_file()}
    if actual != rows or any(p.is_symlink() for p in path.rglob('*')):
        continue
    row = {'path': str(path), 'apparent_bytes': size, 'files': rows, 'removed': False}
    record['removed'].append(row); save()
    shutil.rmtree(path)
    row['removed'] = True; save()
record['free_after'] = shutil.disk_usage(cache).free
record['removed_directories'] = sum(r['removed'] for r in record['removed'])
record['removed_apparent_bytes'] = sum(r['apparent_bytes'] for r in record['removed'] if r['removed'])
save()
print(json.dumps({k: v for k, v in record.items() if k != 'removed'}))
