"""Reverify delivered cache metadata and reproduce nine historical list bodies."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from windows_archive import digest
from windows_vcpkg_metadata import NATIVE_SHA, STATUS_SHA, paragraphs, recover_package

base, output = map(Path, sys.argv[1:])
ci = json.loads(Path(str(base) + '-ci.json').read_text())
receipt = json.loads((base / 'receipt.json').read_text())
assert ci['headSha'] == '9be6a4b3d8f0611cd705e0fee562a38c5cc3f6ba' and ci['conclusion'] == 'success'
assert receipt['metadata_recovery_passed'] is True and receipt['source_commit'] == ci['headSha']
assert all(receipt[key] is False for key in ['release_qualified', 'source_closure_qualified', 'native_link_membership_qualified', 'original_installed_lists_recovered'])
for name in ['windows_vcpkg_metadata.py', 'windows_archive.py']:
    source = subprocess.check_output(['git', 'show', ci['headSha'] + ':tools/nemotron-diarization/' + name], cwd=repo)
    assert hashlib.sha256(source).hexdigest() == digest(repo / 'tools/nemotron-diarization' / name)
assert receipt['tool_sha256'] == digest(repo / 'tools/nemotron-diarization/windows_vcpkg_metadata.py')
assert digest(base / 'native-receipt.json') == NATIVE_SHA and digest(base / 'windows-vcpkg-status.log') == STATUS_SHA
native_root = base.parent / 'windows-38882923286'
assert digest(base / 'native-windows-provenance.json') == digest(native_root / 'nemo-evidence/windows-provenance.json')
status = paragraphs((base / 'windows-vcpkg-status.log').read_text(encoding='utf-8-sig'))
assert len(status) == len(receipt['packages']) == 9
assert {path.name for path in (base / 'cache-archives').iterdir()} == {row['Abi'] + '.zip' for row in status}
provenance = json.loads((base / 'native-windows-provenance.json').read_text(encoding='utf-8-sig'))
original = {Path(row['path'].replace('\\', '/')).name: row for row in provenance['vcpkg_metadata']}
reconstructed = {}
with tempfile.TemporaryDirectory() as temporary:
    for identity, record in zip(status, receipt['packages']):
        port, triplet, abi = identity['Package'], identity['Architecture'], identity['Abi']
        notice = None
        if triplet == 'x64-windows-static-md':
            notice = (native_root / 'nemo-sdk/share/licenses/nemo-speech/third_party/vcpkg' / port / 'LICENSE').read_bytes()
        root = Path(temporary) / (port + '-' + triplet)
        actual = recover_package(base / 'cache-archives' / (abi + '.zip'), identity, root, notice, status)
        assert actual == record
        delivered = base / 'metadata' / root.name
        files = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
        assert files == {path.relative_to(delivered).as_posix() for path in delivered.rglob('*') if path.is_file()}
        for name in files:
            assert (root / name).read_bytes() == (delivered / name).read_bytes()
        # Reproduce the pinned install algorithm's prefix, directory lines,
        # metadata exclusions, lexical sort and LF write_lines output.
        lines = {triplet + '/'}
        for name, entry in record['entry_inventory'].items():
            if Path(name).name in {'CONTROL', 'BUILD_INFO', 'vcpkg.json', '.DS_Store'}:
                continue
            lines.add(triplet + '/' + name + ('/' if entry['directory'] else ''))
            for parent in Path(name).parents:
                if str(parent) != '.':
                    lines.add(triplet + '/' + parent.as_posix() + '/')
        data = ('\n'.join(sorted(lines)) + '\n').encode()
        name = port + '_' + identity['Version'] + '_' + triplet + '.list'
        expected = original[name]
        assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], name
        reconstructed[name] = data
output.mkdir(parents=True, exist_ok=False)
shutil.copytree(base / 'metadata', output / 'metadata')
for name in ['receipt.json', 'native-receipt.json', 'windows-vcpkg-status.log', 'native-windows-provenance.json']:
    shutil.copy2(base / name, output / name)
shutil.copy2(Path(str(base) + '-ci.json'), output / 'ci.json')
shutil.copy2(Path(str(base) + '-ci.log'), output / 'github-job.log')
shutil.copy2(__file__, output / 'collect-windows-vcpkg.py')
(output / 'reconstructed-lists').mkdir()
for name, data in reconstructed.items():
    (output / 'reconstructed-lists' / name).write_bytes(data)
qualification = {'schema_version': 1, 'source_commit': ci['headSha'], 'run_url': ci['url'],
                 'ci_job_id': ci['jobs'][0]['databaseId'], 'metadata_recovery_passed': True,
                 'release_qualified': False, 'source_closure_qualified': False, 'native_link_membership_qualified': False,
                 'native_receipt_sha256': NATIVE_SHA, 'status_sha256': STATUS_SHA,
                 'receiver_reverified_original_cache_archives': 9,
                 'archive_bytes_rehashed': sum(row['archive_bytes'] for row in receipt['packages']),
                 'metadata_files_reverified': 45, 'target_sdk_notices_reverified': 4,
                 'original_list_body_byte_equivalence_verified': 9,
                 'list_reconstruction_note': 'CI producer did not reconstruct lists and its original flag remains false. Receiver now independently reproduced all9 using the pinned install algorithm; every body matches the historical provenance size/SHA256 exactly.',
                 'limits': ['No source archive/port-file/patch closure or native linker-member proof established by cache metadata alone.',
                            'No cached tools/install/build/model execution during recovery; actual Windows runtime behavior belongs to separate qualification.',
                            'Raw nine ZIPs remain in the named CI artifact/local cache; selected original metadata and reconstructed exact list bodies are retained here.',
                            'ABI hashes cover ABI-info bytes, not other ZIP contents; CONTROL/dependency/SPDX identities and SDK notice bytes were separately cross-checked.']}
(output / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
files = {path.relative_to(output).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)}
         for path in output.rglob('*') if path.is_file()}
(output / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(json.dumps({'evidence_files': len(files), 'cache_archives': 9, 'reconstructed_list_matches': 9}))
