"""Replace duplicate model copies with identical APFS clones; preserve all payload bytes."""
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0'); cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from build_valid_mel_candidate import descriptor
from sdk_archive import inventory

build = cache / 'valid-mel-pnola-instrumented-app-build-v1'
result = json.loads((build / 'build.json').read_text())
assert result['app_construction_passed'] and all(c['returncode'] == 0 for c in result['commands'])
source = cache / 'valid-mel-pnola-app-cpu-snapshot-v1/Pnola.app/Contents/Resources'
targets = [build / backend / 'Pnola.app/Contents/Resources' for backend in ['cpu', 'metal']]
targets += [Path('/Users/victor/.cache/pnola-nemotron-build/release/bundle/macos/Pnola.app/Contents/Resources'),
            Path('/Users/victor/workspace/pnola/.desktop-assets')]
report = cache / 'identical-app-model-reclone-v1.json'; assert not report.exists()
record = {'scope': __doc__, 'free_before': shutil.disk_usage(cache).free, 'files': [], 'native_or_model_calls': 0}
def save():
    report.write_text(json.dumps(record, indent=2) + '\n')
save()
for target in targets:
    for origin in sorted(source.rglob('*')):
        name = origin.relative_to(source)
        if (len(name.parts) != 2 or name.parts[0] not in {'bark', 'nemotron'}
                or origin.suffix not in {'.onnx', '.gguf'} or origin.is_symlink()):
            continue
        dest = target / name
        assert dest.is_file() and not dest.is_symlink()
        expected = descriptor(origin); assert descriptor(dest) == expected
        metadata = dest.stat(); mode = stat.S_IMODE(metadata.st_mode)
        temporary = dest.with_name(dest.name + '.identical-cow.next'); assert not temporary.exists()
        row = {'source': str(origin), 'target': str(dest), 'payload': expected, 'mode': mode,
               'inode_before': metadata.st_ino, 'replaced': False}
        record['files'].append(row); save()
        subprocess.run(['/bin/cp', '-c', '-p', str(origin), str(temporary)], check=True)
        os.chmod(temporary, mode)
        assert descriptor(temporary) == expected
        os.replace(temporary, dest)
        assert descriptor(dest) == expected and stat.S_IMODE(dest.stat().st_mode) == mode
        row['inode_after'] = dest.stat().st_ino; row['replaced'] = True; save()
for backend in ['cpu', 'metal']:
    app = build / backend / 'Pnola.app'
    assert inventory(app) == result['apps'][backend]['files']
    subprocess.run(['/usr/bin/codesign', '--verify', '--deep', '--strict', str(app)], check=True, capture_output=True)
record.update(free_after=shutil.disk_usage(cache).free, both_app_inventories_and_deep_signatures_unchanged=True)
save(); print(json.dumps({key: value for key, value in record.items() if key != 'files'}))
