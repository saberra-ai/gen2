"""Receiver byte verification only; Windows execution remains named CI evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from package_windows import CRT, qualified_sdk, verify
from windows_archive import byte_inventory, digest, restore_archive
from windows_naming import NAMES

base, output = map(Path, sys.argv[1:])
ci = json.loads(Path(str(base) + '-ci.json').read_text())
receipt = json.loads((base / 'receipt.json').read_text())
assert ci['conclusion'] == 'success' and ci['headSha'] == 'abcef0692c8ffd9775802b4e9b4583b0204e7731'
assert receipt['package_construction_passed'] is True
assert all(receipt[name] is False for name in ['release_qualified', 'redistribution_qualified', 'runtime_qualified'])
manifest = verify(base / 'runtime')
assert receipt['package_manifest_sha256'] == digest(base / 'runtime/package.json')
for name, expected in manifest['tools'].items():
    source = subprocess.check_output(['git', 'show', ci['headSha'] + ':tools/nemotron-diarization/' + name], cwd=repo)
    assert hashlib.sha256(source).hexdigest() == expected == digest(repo / 'tools/nemotron-diarization' / name), name
native_root = base.parent / 'windows-38882923286'
lock = json.loads((repo / 'tools/nemotron-diarization/runtime.lock.json').read_text())
native, _, records = qualified_sdk(native_root / 'nemo-sdk', native_root / 'nemo-evidence', lock)
assert digest(native_root / 'nemo-evidence/receipt.json') == receipt['native_receipt_sha256']
assert native['source_commit'] == receipt['native_source_commit'] == manifest['native_source_commit']
archive = json.loads((base / 'development-windows-runtime-files.json').read_text())
for key in ['archive', 'archive_bytes', 'archive_sha256']:
    assert archive[key] == receipt['archive'][key]
assert byte_inventory(base / 'runtime') == archive['files'] == byte_inventory(base / 'relocated runtime ü')
with tempfile.TemporaryDirectory() as temporary:
    restored = restore_archive(base / archive['archive'], archive, Path(temporary) / 'receiver relocated runtime ü')
    assert verify(restored) == manifest and byte_inventory(restored) == archive['files']
assert set(manifest['libraries']) == set(NAMES.values()) | CRT
for name, row in manifest['libraries'].items():
    assert all(row[key] == records['bin/' + name][key] for key in ['bytes', 'sha256'])
output.mkdir(parents=True, exist_ok=False)
for name in ['receipt.json', 'development-windows-runtime-files.json']:
    shutil.copy2(base / name, output / name)
shutil.copy2(base / 'runtime/package.json', output / 'package.json')
shutil.copytree(base / 'runtime/metadata', output / 'metadata')
shutil.copytree(base / 'runtime/notices', output / 'notices')
shutil.copy2(Path(str(base) + '-ci.json'), output / 'ci.json')
shutil.copy2(Path(str(base) + '-ci.log'), output / 'github-job.log')
shutil.copy2(base.parent / 'windows-package-first-ci-failure.log', output / 'first-ci-test-selection-failure.log')
shutil.copy2(__file__, output / 'collect-windows-package.py')
qualification = {'schema_version': 1, 'source_commit': ci['headSha'], 'run_url': ci['url'],
                 'ci_job_id': ci['jobs'][0]['databaseId'], 'package_construction_passed': True,
                 'release_qualified': False, 'redistribution_qualified': False, 'runtime_qualified': False,
                 'native_source_commit': native['source_commit'], 'native_receipt_sha256': receipt['native_receipt_sha256'],
                 'archive': receipt['archive'], 'archive_entries_reverified': len(archive['files']),
                 'dlls_preserved_from_qualified_sdk': 13, 'sdk_regular_files_reverified': len(native['installed_files']),
                 'receiver_archive_restore_and_pe_manifest_passed': True,
                 'packaging_host': receipt['packaging_host'],
                 'prior_failed_run': {'source_commit': 'cb46d2b72604b8e206b1165c8830a2cdaabb2485',
                     'run_url': 'https://github.com/saberra-ai/gen2/actions/runs/38887646359',
                     'reason': 'Workflow attempted POSIX/scientific tests on Windows without their dependencies; failed before artifact acquisition or package construction. Retry selects 25 Windows/portable tests. Full 78 tests passed on macOS and run separately in Linux evidence CI.'},
                 'limits': ['Actual Windows construction and Unicode byte restoration only; no model/native package execution.',
                            'Receiver macOS rehashed SDK/package/archive and re-inspected PE tables; no Windows execution or ACL reconstruction on receiver.',
                            'No offline runtime, desktop machine, chat/Bark/Laya coexistence, held-out accuracy or performance qualification.',
                            'No Microsoft redistribution terms/entitlement or complete static-dependency source qualification.',
                            'Opaque DLL/archive payload stays in the named CI artifact and local receiver cache, not this Git evidence directory.']}
(output / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
files = {path.relative_to(output).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)}
         for path in output.rglob('*') if path.is_file()}
(output / 'collection.json').write_text(json.dumps({'files': files, 'scope': 'All retained evidence bytes; opaque runtime/archive reverified separately and not committed.'}, indent=2) + '\n')
print(json.dumps({'evidence_files': len(files), 'archive_entries': len(archive['files']), 'archive_sha256': archive['archive_sha256']}))
