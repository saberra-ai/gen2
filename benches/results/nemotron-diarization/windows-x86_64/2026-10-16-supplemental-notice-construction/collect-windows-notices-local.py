"""Reverify macOS construction of Windows bytes; no Windows/native execution."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from package_windows import CRT, NAMES, verify
from windows_archive import byte_inventory, digest, restore_archive
from windows_notices import verify_supplement

cache = Path('/Users/victor/.cache/gen2-nemotron3-n0/platform-ci')
base = cache / 'windows-supplement-construction-local-1'
sdk = cache / 'windows-38882923286/nemo-sdk'
output = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-16-supplemental-notice-construction'
read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
receipt = read(base / 'receipt.json')
archive = read(base / 'development-windows-runtime-files.json')
assert receipt['package_construction_passed'] is True and receipt['runtime_qualified'] is False
manifest = verify(base / 'runtime')
assert digest(base / 'runtime/package.json') == receipt['package_manifest_sha256']
for name, expected in manifest['tools'].items():
    original = subprocess.check_output(['git', 'show', source + ':tools/nemotron-diarization/' + name], cwd=repo)
    assert hashlib.sha256(original).hexdigest() == expected
assert byte_inventory(base / 'runtime') == byte_inventory(base / 'relocated runtime ü') == archive['files']
with tempfile.TemporaryDirectory() as temp:
    restored = restore_archive(base / archive['archive'], archive, Path(temp) / 'receiver relocated runtime ü')
    assert verify(restored) == manifest and byte_inventory(restored) == archive['files']
verify_supplement(base / 'runtime', manifest['source_notice_supplement'])
for name in set(NAMES.values()) | CRT:
    assert (base / 'runtime/lib' / name).read_bytes() == (sdk / 'bin' / name).read_bytes()
notices = sdk / 'share/licenses/nemo-speech'
upstream = [path for path in notices.rglob('*') if path.is_file()]
assert len(upstream) == 8
for path in upstream:
    assert path.read_bytes() == (base / 'runtime/notices/nemo-speech' / path.relative_to(notices)).read_bytes()
output.mkdir(parents=True, exist_ok=False)
for name in ['receipt.json', 'development-windows-runtime-files.json']:
    shutil.copy2(base / name, output / name)
shutil.copy2(base / 'runtime/package.json', output / 'package.json')
shutil.copytree(base / 'runtime/notices', output / 'notices')
shutil.copy2(__file__, output / 'collect-windows-notices-local.py')
with (output / 'tests.log').open('w') as stream:
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tools/nemotron-diarization'], cwd=repo, stdout=stream, stderr=subprocess.STDOUT)
assert result.returncode == 0
qualification = {'schema_version': 1, 'source_commit': source,
    'local_byte_construction_and_restore_passed': True, 'runtime_qualified': False,
    'release_qualified': False, 'redistribution_qualified': False,
    'packaging_host': receipt['packaging_host'], 'native_source_commit': manifest['native_source_commit'],
    'native_receipt_sha256': manifest['native_receipt_sha256'], 'archive': receipt['archive'],
    'archive_entries': len(archive['files']), 'unchanged_qualified_dlls': 13, 'unchanged_installed_upstream_notices': 8,
    'supplemental_notice_files': manifest['source_notice_supplement']['files'],
    'model_free_tool_tests': 88,
    'limits': ['Actual macOS construction and three byte-equivalent runtime trees including a fresh Unicode ZIP restoration.',
        'All 13 qualified Windows DLLs and eight original installed upstream notices are unchanged; three supplemental texts and source-selection provenance added.',
        'No Windows, model, installer or native execution in this receipt; no complete attribution, Microsoft distribution rights, quality or release qualification.']}
(output / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
files = {path.relative_to(output).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)} for path in output.rglob('*') if path.is_file()}
(output / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(json.dumps({'source_commit': source, 'evidence_files': len(files), 'runtime_archive_entries': len(archive['files'])}))
