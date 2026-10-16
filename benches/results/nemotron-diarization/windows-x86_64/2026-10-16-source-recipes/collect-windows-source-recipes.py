"""Recheck pinned recipe/source/notice bytes without building or executing them."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
base, output = map(Path, sys.argv[1:])
sha = lambda data: hashlib.sha256(data).hexdigest()
manifest = json.loads((base / 'manifest.json').read_text())
assert sha((base / 'manifest.json').read_bytes()) == '7f96b57e68aa2aed798e483bb92a3ab541f7d6a332299b655dd7347af27b5b1d'
pin = '9e593bb18ea69cc5095e012465dcd675a822ed0d'
assert manifest['vcpkg_baseline'] == pin and len(manifest['ports']) == 6
ports = {}
for port in manifest['ports']:
    records = port['files']
    for row in records:
        path = base / row['path']
        data = path.read_bytes()
        assert len(data) == row['bytes'] and sha(data) == row['sha256']
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == row['git_blob_sha1']
        assert row['url'].startswith('https://raw.githubusercontent.com/microsoft/vcpkg/' + pin + '/ports/' + port['port'] + '/')
        ports[row['path']] = row
assert len(ports) == 34
recovery = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-16-cache-metadata-recovery'
comparisons = 0
for package in manifest['package_port_verification']:
    name = package['package']
    port = name.split('-x64-windows')[0]
    reference = recovery / 'metadata' / name
    local = reference
    abi_path = local / 'share' / port / 'vcpkg_abi_info.txt'
    assert sha(abi_path.read_bytes()) == package['abi_info_sha256']
    abi_fields = dict(line.split(' ', 1) for line in abi_path.read_text().splitlines())
    spdx = json.loads((local / 'share' / port / 'vcpkg.spdx.json').read_text())
    entries = [row for row in spdx['files'] if row['SPDXID'].startswith('SPDXRef-port-file-')]
    expected = {row['path'] for row in ports.values() if row['path'].startswith('ports/' + port + '/')}
    assert {'ports/' + port + '/' + row['fileName'].removeprefix('./') for row in entries} == expected
    for row in entries:
        name = row['fileName'].removeprefix('./')
        digest = ports['ports/' + port + '/' + name]['sha256']
        assert row['checksums'] == [{'algorithm': 'SHA256', 'checksumValue': digest}]
        assert abi_fields[name] == digest
        comparisons += 1
assert comparisons == 53
for row in manifest['upstream_sources']:
    data = (base / row['path']).read_bytes()
    assert len(data) == row['bytes'] and sha(data) == row['sha256']
    assert hashlib.sha512(data).hexdigest() == row['sha512']
    for port in row['ports']:
        assert row['sha512'] in (base / 'ports' / port / 'portfile.cmake').read_text()
for row in manifest['retained_source_notices']:
    data = (base / row['retained_path']).read_bytes()
    assert len(data) == row['bytes'] and sha(data) == row['sha256']
    with tarfile.open(base / row['source_archive'], 'r:gz') as source:
        member = source.getmember(row['member'])
        assert member.isfile() and member.size == len(data)
        assert source.extractfile(member).read(len(data) + 1) == data
for row in manifest['research_scripts']:
    data = (base / row['path']).read_bytes()
    assert len(data) == row['bytes'] and sha(data) == row['sha256']
output.mkdir(parents=True, exist_ok=False)
shutil.copytree(base / 'ports', output / 'ports')
shutil.copytree(base / 'notices', output / 'notices')
shutil.copytree(base / 'metadata', output / 'github-api-directory-responses')
for name in ['manifest.json', 'vcpkg-LICENSE.txt', 'research_acquire.py', 'research_verify.py', 'research_notices.py']:
    shutil.copy2(base / name, output / name)
shutil.copy2(__file__, output / 'collect-windows-source-recipes.py')
qualification = {'schema_version': 1, 'vcpkg_baseline': pin, 'recipe_source_bytes_verified': True,
                 'release_qualified': False, 'complete_source_notice_closure': False, 'native_link_membership_qualified': False,
                 'raw_port_files_verified': 34, 'package_spdx_and_abi_port_fields_verified': 53,
                 'source_archives_verified': 3, 'source_archive_bytes_verified': sum(row['bytes'] for row in manifest['upstream_sources']),
                 'source_notices_retained': len(manifest['retained_source_notices']),
                 'receiver_harness_note': 'Initial receiver incorrectly assumed acquisition copied package metadata; it only retained GitHub directory responses. Corrected to read the already-verified committed recovery metadata directly, then reran byte checks before collection.',
                 'acquisition_manifest_sha256': sha((base / 'manifest.json').read_bytes()),
                 'limits': ['Recipe identity and source-byte checks only; no source/cached tool execution, installs, native builds or model runs.',
                            'Not every toolchain/helper ABI field or final native linker member is established.',
                            'Recipe-selected copyright equality does not establish full downstream attribution. Darts-clone distinct BSD and esaxx scope remain explicit packaging checks.',
                            'Raw source archives remain in the external acquisition cache; pinned recipes, manifest and selected source notices are retained here.',
                            'Existing Windows runtime and metadata qualification receipts remain unchanged.']}
(output / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
files = {path.relative_to(output).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes())}
         for path in output.rglob('*') if path.is_file()}
(output / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(json.dumps({'evidence_files': len(files), 'port_files': 34, 'spdx_abi_comparisons': comparisons, 'source_archives': 3}))
