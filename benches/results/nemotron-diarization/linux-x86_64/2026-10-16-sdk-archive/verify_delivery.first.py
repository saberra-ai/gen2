"""Recheck exact delivered Linux CI SDK bytes and cross-host filesystem metadata.

Usage: python verify_delivery.py /path/to/downloaded/CI/artifact
No native code or model execution. Requires Python >=3.12.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile

EXPECTED_SHA256 = '716a14fc161bc9edc10ffd2240ddff83ef98a51bd287fa2b967b44ee75dfaba3'
root = Path(sys.argv[1])
evidence = root / 'nemo-evidence'
manifest = json.loads((evidence / 'development-sdk-files.json').read_text())
receipt = json.loads((evidence / 'receipt.json').read_text())
archive = evidence / 'development-sdk.tar.gz'
assert receipt['development_native_gates_passed']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED_SHA256 == manifest['archive_sha256'] == receipt['development_sdk_archive']['archive_sha256']
assert archive.stat().st_size == manifest['archive_bytes'] == receipt['development_sdk_archive']['archive_bytes']
with tempfile.TemporaryDirectory(prefix='relocated Linux SDK ü ') as temporary:
    relocated = Path(temporary)
    with tarfile.open(archive) as tar:
        expected_names = set()
        for relative, record in manifest['files'].items():
            assert relative == '.' or (not Path(relative).is_absolute() and '..' not in Path(relative).parts)
            name = 'sdk' if relative == '.' else 'sdk/' + relative
            expected_names.add(name)
            member = tar.getmember(name)
            assert member.mode == record['mode'] and 0 <= member.mode <= 0o777
            assert {'file': member.isfile(), 'directory': member.isdir(), 'symlink': member.issym()}[record['type']]
            if record['type'] == 'symlink':
                assert member.linkname == record['target']
        assert len(tar.getmembers()) == len(expected_names) and {m.name for m in tar.getmembers()} == expected_names
        tar.extractall(relocated, filter='data')
    sdk = relocated / 'sdk'
    for relative, record in manifest['files'].items():
        path = sdk / relative
        if stat.S_IMODE(path.lstat().st_mode) != record['mode']:
            os.chmod(path, record['mode'], follow_symlinks=record['type'] != 'symlink')
    actual = {'.', *(p.relative_to(sdk).as_posix() for p in sdk.rglob('*'))}
    assert actual == set(manifest['files'])
    for relative, record in manifest['files'].items():
        path = sdk / relative
        assert stat.S_IMODE(path.lstat().st_mode) == record['mode']
        if record['type'] == 'symlink':
            assert path.is_symlink() and os.readlink(path) == record['target'] and path.resolve(strict=True).is_relative_to(sdk)
        elif record['type'] == 'file':
            assert not path.is_symlink() and path.is_file() and path.stat().st_size == record['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        else:
            assert not path.is_symlink() and path.is_dir()
print(json.dumps({'verified': True, 'archive_sha256': EXPECTED_SHA256, 'inventory_entries': len(manifest['files']), 'release_qualified': False}))
