"""Recover original ABI-bound cache metadata without installing or executing it.

This does not reconstruct missing installed lists or establish native linker
membership, complete source closure, redistribution rights or runtime behavior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import zipfile

from windows_archive import digest, safe_relative

NATIVE_SHA = '18fcacac2b79a31a9ff1791ae62c09defb43a70bbf2c9cf173bdbc3991af26d3'
STATUS_SHA = 'aa85bcc183d2163072bdee4a245ce2c400f4a3d7aa68fac1ec59efb56d4f2061'
CACHE_KEY = 'nemo-diar-vcpkg-9e593bb18ea69cc5095e012465dcd675a822ed0d-x64-static-md'
TOOL_CREATOR = 'Tool: vcpkg-2026-07-27-98d7cb0cf1f4686a3e43aa5672b6230c1d56bce8'


def paragraphs(text):
    records, current, key = [], {}, None
    for line in [*text.splitlines(), '']:
        if not line:
            if current:
                records.append(current)
            current, key = {}, None
        elif line.startswith((' ', '\t')) and key:
            current[key] += '\n' + line.lstrip()
        else:
            key, separator, value = line.partition(':')
            if not separator or not key or key in current:
                raise ValueError('invalid or duplicate vcpkg control field')
            current[key] = value.strip()
    return records


def recover_package(archive, identity, output, notice=None, identities=None):
    # Bound central-directory allocation before constructing ZipInfo objects.
    if archive.stat().st_size > 512 * 1024 * 1024:
        raise ValueError('vcpkg cache ZIP exceeds byte bound')
    with archive.open('rb') as raw:
        raw.seek(0, 2)
        length = raw.tell()
        raw.seek(max(0, length - 65557))
        tail = raw.read(65557)
        offset = tail.rfind(b'PK\x05\x06')
        if offset < 0 or len(tail) - offset < 22:
            raise ValueError('vcpkg ZIP end record absent')
        signature, disk, central_disk, count, total, size, start, comment = struct.unpack('<4s4H2IH', tail[offset:offset + 22])
        if (disk or central_disk or count != total or not 1 <= total <= 20000
                or size > 4 * 1024 * 1024 or start + size + 22 + comment != length
                or len(tail) - offset != 22 + comment):
            raise ValueError('unsupported or unbounded vcpkg central directory')
        eocd = length - 22 - comment
        if eocd >= 20:
            raw.seek(eocd - 20)
            if raw.read(4) == b'PK\x06\x07':
                raise ValueError('ZIP64 cannot override bounded classic vcpkg metadata')
        raw.seek(0)
        with zipfile.ZipFile(raw) as bundle:
            entries, folded = {}, set()
            if len(bundle.infolist()) != total:
                raise ValueError('vcpkg central entry count differs')
            for info in bundle.infolist():
                name = info.filename.removesuffix('/')
                safe_relative(name)
                if name.casefold() in folded:
                    raise ValueError('duplicate vcpkg ZIP path')
                folded.add(name.casefold())
                entries[name] = {'bytes': info.file_size, 'compressed_bytes': info.compress_size,
                                 'crc32': f'{info.CRC:08x}', 'directory': info.is_dir()}
            port = identity['Package']
            wanted = ['CONTROL', 'BUILD_INFO', *['share/' + port + '/' + name for name in
                      ['vcpkg_abi_info.txt', 'vcpkg.spdx.json', 'copyright']]]
            payloads = {}
            for name in wanted:
                info = bundle.getinfo(name)
                mode = info.external_attr >> 16
                if (info.is_dir() or info.flag_bits & 1 or stat.S_IFMT(mode) not in {0, stat.S_IFREG}
                        or info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                        or not 0 < info.file_size <= 4 * 1024 * 1024):
                    raise ValueError('unsupported vcpkg metadata member')
                with bundle.open(info) as stream:
                    payload = stream.read(4 * 1024 * 1024 + 1)
                if len(payload) != info.file_size:
                    raise ValueError('vcpkg metadata byte count differs')
                payloads[name] = payload
    abi = hashlib.sha256(payloads[wanted[2]]).hexdigest()
    control = paragraphs(payloads['CONTROL'].decode('utf-8-sig'))
    # These nine historical packages have only core paragraphs. Status adds
    # its installation field to the original serialized CONTROL paragraph.
    if control != [{key: value for key, value in identity.items() if key != 'Status'}] or abi != identity['Abi']:
        raise ValueError('vcpkg actual ABI/control identity differs from qualified status')
    abi_fields = {}
    for line in payloads[wanted[2]].decode('utf-8').splitlines():
        key, separator, value = line.partition(' ')
        if not separator or not value or key in abi_fields:
            raise ValueError('ambiguous vcpkg ABI-info fields')
        abi_fields[key] = value
    if abi_fields.get('triplet') != identity['Architecture'] or abi_fields.get('features') != 'core':
        raise ValueError('vcpkg ABI triplet/features differ from qualified core package')
    identities = identities or [identity]
    known = {(row['Package'], row['Architecture']): row['Abi'] for row in identities}
    dependencies = {}
    for spec in filter(None, identity.get('Depends', '').split(', ')):
        if not re.fullmatch(r'[a-z0-9-]+(?::x64-windows(?:-static-md)?)?', spec):
            raise ValueError('unsupported qualified vcpkg dependency form')
        name, _, architecture = spec.partition(':')
        expected = known.get((name, architecture or identity['Architecture']))
        if not expected or name in dependencies:
            raise ValueError('missing or ambiguous qualified vcpkg dependency ABI')
        dependencies[name] = expected
    observed_dependencies = {name: abi_fields[name] for name in {row['Package'] for row in identities} if name in abi_fields}
    if observed_dependencies != dependencies:
        raise ValueError('ABI-info dependencies differ from qualified installed graph')
    build, = paragraphs(payloads['BUILD_INFO'].decode('utf-8-sig'))
    if identity['Architecture'] == 'x64-windows-static-md' and (
            build.get('CRTLinkage') != 'dynamic' or build.get('LibraryLinkage') != 'static'):
        raise ValueError('target package is not static with dynamic CRT')
    spdx = json.loads(payloads[wanted[3]])
    version = identity['Version'] + ('#' + identity['Port-Version'] if identity.get('Port-Version', '0') != '0' else '')
    spec = identity['Package'] + ':' + identity['Architecture']
    if (not isinstance(spdx, dict) or spdx.get('SPDXID') != 'SPDXRef-DOCUMENT'
            or spdx.get('spdxVersion') != 'SPDX-2.3' or spdx.get('name') != spec + '@' + version + ' ' + abi
            or spdx.get('creationInfo', {}).get('creators') != [TOOL_CREATOR]):
        raise ValueError('vcpkg SPDX document/tool identity differs')
    packages = spdx.get('packages', [])
    for identifier, name, revision in [('SPDXRef-port', identity['Package'], version), ('SPDXRef-binary', spec, abi)]:
        rows = [row for row in packages if row.get('SPDXID') == identifier]
        if len(rows) != 1 or rows[0].get('name') != name or rows[0].get('versionInfo') != revision:
            raise ValueError('vcpkg SPDX port/binary identity differs')
    if notice is not None and payloads[wanted[4]] != notice:
        raise ValueError('cached target copyright differs from qualified SDK notice')
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return {'identity': identity, 'archive_sha256': digest(archive, 512 * 1024 * 1024),
            'archive_bytes': archive.stat().st_size, 'abi_info_sha256': abi,
            'target_notice_matches_sdk': notice is not None,
            'control_fields_match_status': True, 'abi_dependency_graph_verified': True, 'spdx_identity_verified': True,
            'entry_inventory': entries, 'metadata': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                                                     for name, data in payloads.items()},
            'scope': 'Recovered cache metadata with CONTROL/status, raw ABI-info/dependency, SPDX identity and optional notice cross-checks. ABI hash covers ABI-info alone; ZIP entry sizes/CRCs are declarations, not rehashed library bytes.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['archives', 'native-evidence', 'sdk-prefix', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {'schema_version': 1, 'metadata_recovery_passed': False, 'release_qualified': False,
               'source_closure_qualified': False, 'native_link_membership_qualified': False,
               'original_installed_lists_recovered': False, 'packages': []}
    try:
        evidence = args.native_evidence.resolve(strict=True)
        if digest(evidence / 'receipt.json') != NATIVE_SHA or digest(evidence / 'windows-vcpkg-status.log') != STATUS_SHA:
            raise ValueError('exact qualified historical Windows receipts required')
        native = json.loads((evidence / 'receipt.json').read_text(encoding='utf-8-sig'))
        if native['development_native_gates_passed'] is not True:
            raise ValueError('passing native prerequisite required')
        status = paragraphs((evidence / 'windows-vcpkg-status.log').read_text(encoding='utf-8-sig'))
        if len(status) != 9 or any(row['Status'] != 'install ok installed' for row in status):
            raise ValueError('exact nine qualified vcpkg package identities required')
        installed = {name.replace('\\', '/'): row for name, row in native['installed_files'].items()}
        raw = args.output / 'cache-archives'
        raw.mkdir()
        for row in status:
            abi, port, triplet = row['Abi'], row['Package'], row['Architecture']
            if (not re.fullmatch('[0-9a-f]{64}', abi) or not re.fullmatch('[a-z0-9-]+', port)
                    or triplet not in {'x64-windows', 'x64-windows-static-md'}):
                raise ValueError('unsafe qualified vcpkg selector')
            source = args.archives / abi[:2] / (abi + '.zip')
            observed = source.lstat()
            if (not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1
                    or getattr(observed, 'st_file_attributes', 0) & 0x400
                    or observed.st_size > 512 * 1024 * 1024):
                raise ValueError('exact cached vcpkg ABI ZIP absent')
            archive = raw / (abi + '.zip')
            shutil.copy2(source, archive)
            if digest(source, 512 * 1024 * 1024) != digest(archive, 512 * 1024 * 1024):
                raise ValueError('cache archive changed during copy')
            notice = None
            if triplet == 'x64-windows-static-md':
                relative = 'share/licenses/nemo-speech/third_party/vcpkg/' + port + '/LICENSE'
                path = args.sdk_prefix / relative
                if {'bytes': path.stat().st_size, 'sha256': digest(path)} != installed[relative]:
                    raise ValueError('SDK target notice differs from native receipt')
                notice = path.read_bytes()
            record = recover_package(archive, row, args.output / 'metadata' / (port + '-' + triplet), notice, status)
            receipt['packages'].append(record)
        shutil.copy2(evidence / 'receipt.json', args.output / 'native-receipt.json')
        shutil.copy2(evidence / 'windows-vcpkg-status.log', args.output / 'windows-vcpkg-status.log')
        shutil.copy2(evidence / 'windows-provenance.json', args.output / 'native-windows-provenance.json')
        receipt.update(metadata_recovery_passed=True, native_source_commit=native['source_commit'],
                       native_receipt_sha256=NATIVE_SHA, status_sha256=STATUS_SHA,
                       recovery_cache_key=CACHE_KEY, recovery_cache_branch='refs/heads/codex/nemotron3-platforms',
                       source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True).strip(),
                       tool_sha256=digest(Path(__file__)),
                       scope='Additive original cache/ABI/control/notice recovery only; no installs, tool execution, builds or model runs.')
    except Exception as error:
        receipt['error'] = str(error)
        raise
    finally:
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
