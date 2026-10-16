"""Independently re-extract pinned Microsoft bytes as data; never executes payloads."""
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from package_windows import CRT, summary
from windows_archive import digest
from windows_pe import inspect_pe

base = Path('/Users/victor/.cache/gen2-nemotron3-n0/research/windows-central-prerequisite')
output = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-16-central-prerequisite-static'
read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
manifest = read(base / 'manifest.json')
assert digest(base / 'manifest.json') == 'b3e3270200adbf1ddc077ae8ec5fc67a0f1ef1c74623038e3c2c27675784c383'
for row in manifest['artifacts']:
    assert (base / row['path']).stat().st_size == row['bytes'] and digest(base / row['path']) == row['sha256']
lock = read(repo / 'tools/nemotron-diarization/windows-vc-runtime.lock.json')
installer = base / 'VC_redist.x64.exe'
raw = installer.read_bytes()
assert len(raw) == lock['installer']['size'] and hashlib.sha256(raw).hexdigest() == lock['installer']['sha256'] == manifest['installer_sha256']
static, minimum = read(base / 'static-inspection.json'), read(base / 'minimum-x64-cab-inspection.json')
assert static['installer_executed'] is False and static['installer_authenticode_verified'] is False
assert digest(repo / 'tools/nemotron-diarization/windows_pe.py') == minimum['source_helpers']['windows_pe.py_sha256']
assert sorted(CRT) == minimum['source_helpers']['crt_names']
assert minimum['dlls_executed'] is False and minimum['authenticode_verified'] is False and minimum['actual_install_paths_verified'] is False
for row in static['tool_files']:
    assert Path(row['path']).stat().st_size == row['bytes'] and digest(Path(row['path'])) == row['sha256']
sevenzip = '/opt/homebrew/bin/7z'
commands = []
with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    def extract(source, target):
        argv = [sevenzip, 'x', '-y', '-o' + str(target), str(source)]
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        commands.append({'argv': argv, 'returncode': result.returncode,
                         'stdout_sha256': hashlib.sha256(result.stdout).hexdigest(), 'stderr_sha256': hashlib.sha256(result.stderr).hexdigest()})
    extract(installer, root / 'ux')
    assert (root / 'ux/0').read_bytes() == (base / 'ux/0').read_bytes()
    burn = ET.fromstring((root / 'ux/0').read_bytes())
    ns = {'b': 'http://schemas.microsoft.com/wix/2008/Burn'}
    container = burn.find('b:Container', ns).attrib
    extent = static['attached_container']
    cab = raw[extent['offset']:extent['offset'] + extent['bytes']]
    assert len(cab) == int(container['FileSize']) == extent['bytes']
    assert hashlib.sha1(cab).hexdigest() == container['Hash'].lower() == extent['sha1']
    assert hashlib.sha256(cab).hexdigest() == extent['sha256']
    (root / 'attached.cab').write_bytes(cab)
    extract(root / 'attached.cab', root / 'attached')
    expected = {row['archive_source']: row for row in static['payloads']}
    seen = set()
    for parent, folder in [(burn.find('b:UX', ns), 'ux'), (burn, 'attached')]:
        for payload in parent.findall('b:Payload', ns):
            fields = payload.attrib
            name = folder + '/' + fields['SourcePath']
            data = (root / name).read_bytes()
            row = expected[name]
            assert len(data) == int(fields['FileSize']) == row['bytes']
            assert hashlib.sha1(data).hexdigest() == fields['Hash'].lower()
            assert hashlib.sha256(data).hexdigest() == row['sha256']
            assert row['logical_path'] == fields['FilePath']
            assert data == (base / row['layout_path']).read_bytes()
            seen.add(name)
    assert len(seen) == 24 and seen == set(expected)
    assert digest(root / 'attached/a4') == minimum['cab_sha256']
    extract(root / 'attached/a4', root / 'minimum')
    actual = {path.name: {'bytes': path.stat().st_size, 'sha256': digest(path)} for path in (root / 'minimum').iterdir()}
    assert actual == {row['cab_logical_name']: {key: row[key] for key in ['bytes', 'sha256']} for row in minimum['cab_inventory']}
    assert len(actual) == 12 and {row['runtime_name'] for row in minimum['selected_dlls']} == CRT | {'vcomp140.dll'}
    for row in minimum['selected_dlls']:
        info = inspect_pe(root / 'minimum' / row['cab_logical_name'])
        assert info['machine'] == 'AMD64' and info['is_dll'] and info['format'] == 'PE32+'
        assert summary(info) == {key: row[key] for key in summary(info)}
        argv = [sevenzip, 'l', '-slt', '-tPE', str(root / 'minimum' / row['cab_logical_name'])]
        listing = subprocess.check_output(argv, text=True)
        commands.append({'argv': argv, 'returncode': 0, 'stdout_sha256': hashlib.sha256(listing.encode()).hexdigest()})
        file_version = re.search(r'^Comment = FileVersion: (.+)$', listing, re.M).group(1)
        product_version = re.search(r'^ProductVersion: (.+)$', listing, re.M).group(1)
        assert file_version == row['file_version_static_string'] == product_version == row['product_version_static_string'] == lock['version']
    app_data = ET.fromstring((root / 'ux/u17').read_bytes())
    license_urls = [element.attrib['LicenseUrl'] for element in app_data.iter() if 'LicenseUrl' in element.attrib]
    assert license_urls == [lock['terms']['bundle_selected_url']]
    assert lock['terms']['bundle_selected_url'].encode() in (root / 'ux/u2').read_bytes()
certificate = static['static_certificate_region']
assert raw[certificate['offset']:certificate['offset'] + certificate['bytes']] == (base / certificate['path']).read_bytes()
assert digest(base / certificate['path']) == certificate['sha256']
terms = base / 'Visual-C-V14-License-Redistributable_and_Runtime_ENU.docx'
assert terms.stat().st_size == lock['terms']['observed_docx_bytes'] and digest(terms) == lock['terms']['observed_docx_sha256']
with zipfile.ZipFile(terms) as bundle:
    document = ET.fromstring(bundle.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paragraphs = [''.join(text.text or '' for text in paragraph.findall('.//w:t', ns)) for paragraph in document.findall('.//w:p', ns)]
    assert ('\n'.join(paragraphs) + '\n').encode() == (base / 'runtime-terms-derived.txt').read_bytes()
output.mkdir(parents=True, exist_ok=False)
# Metadata/code/logs only: proprietary installer, runtime/payload binaries,
# original DOCX/full derived terms and complete landing HTML remain external.
excluded = {'Visual-C-V14-License-Redistributable_and_Runtime_ENU.docx', 'runtime-terms-derived.txt', 'terms-landing.html'}
for row in manifest['artifacts']:
    if row['path'] not in excluded:
        target = output / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / row['path'], target)
shutil.copy2(base / 'manifest.json', output / 'acquisition-manifest.json')
shutil.copy2(__file__, output / 'collect-windows-prerequisite-static.py')
qualification = {'schema_version': 1, 'research_metadata_reverified': True,
    'installer_bytes_and_sha256_reverified': lock['installer'],
    'fresh_static_payloads_size_sha1_sha256_reverified': 24, 'fresh_minimum_cab_members_reverified': 12,
    'selected_amd64_dll_pe_reports_recomputed': 9, 'actual_bundle_selected_terms_url': license_urls[0],
    'terms_docx_bytes_and_sha256_reverified': lock['terms'], 'derived_docx_text_recomputed': True,
    'opaque_binaries_and_full_terms_excluded_from_git': True,
    'static_extractor_commands': commands,
    'installer_authenticode_verified': False, 'installer_executed': False, 'actual_install_paths_verified': False,
    'central_runtime_qualified': False, 'release_qualified': False, 'redistribution_qualified': False,
    'limits': ['macOS static bytes/extraction only; no installer, DLL, MSI, native model or Windows code execution.',
        'Static PE certificate presence and resource version strings are not Windows Authenticode/version API/loader results.',
        'CAB logical names are not verified MSI installed destinations; exact central runtime signature/setup/restart/loaded-module results remain pending.',
        'Current selected terms association and acquisition are not assent, historical/future applicability, a distribution grant or complete legal qualification.']}
(output / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
files = {path.relative_to(output).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)} for path in output.rglob('*') if path.is_file()}
(output / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(json.dumps({'evidence_files': len(files), 'fresh_static_payloads': len(seen), 'fresh_inner_files': len(actual)}))
