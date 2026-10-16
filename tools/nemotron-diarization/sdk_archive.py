"""Preserve a development SDK's exact filesystem metadata in a CI artifact.

This is an SDK snapshot, not a closed runtime or portable release package.
Extractors that filter permissions must restore the recorded safe mode bits
before comparing the extracted inventory, including directory modes.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile


def inventory(root):
    root = root.resolve(strict=True)
    records = {}
    for path in [root, *sorted(root.rglob('*'))]:
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
            raise ValueError(f'unsupported special permissions: {relative}')
        record = {'mode': mode}
        if stat.S_ISLNK(metadata.st_mode):
            target = path.readlink()
            if target.is_absolute() or not path.resolve(strict=True).is_relative_to(root):
                raise ValueError(f'escaping SDK alias: {relative}')
            record.update(type='symlink', target=target.as_posix())
        elif stat.S_ISDIR(metadata.st_mode):
            record['type'] = 'directory'
        elif stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise ValueError(f'unsupported hardlink: {relative}')
            data = path.read_bytes()
            record.update(type='file', bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        else:
            raise ValueError(f'unsupported SDK file type: {relative}')
        records[relative] = record
    return records


def write_archive(sdk, output, stem='development-sdk'):
    """Write a deterministic tar plus its complete lstat inventory; no extraction."""
    sdk, output = Path(sdk), Path(output)
    if Path(stem).name != stem or not stem:
        raise ValueError('archive stem must be a simple filename')
    records = inventory(sdk)
    archive = output / (stem + '.tar.gz')
    manifest = output / (stem + '-files.json')
    # Exclusive creation prevents replacing evidence from an earlier run.
    with archive.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for relative, record in records.items():
                    name = 'sdk' if relative == '.' else 'sdk/' + relative
                    info = tarfile.TarInfo(name)
                    info.mode = record['mode']
                    if record['type'] == 'directory':
                        info.type = tarfile.DIRTYPE
                        tar.addfile(info)
                    elif record['type'] == 'symlink':
                        info.type = tarfile.SYMTYPE
                        info.linkname = record['target']
                        tar.addfile(info)
                    else:
                        data = (sdk / relative).read_bytes()
                        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                            raise ValueError(f'SDK changed during archive creation: {relative}')
                        info.size = len(data)
                        tar.addfile(info, io.BytesIO(data))
    # Detect concurrent changes to aliases, permissions or the file set too.
    if inventory(sdk) != records:
        raise ValueError('SDK filesystem changed during archive creation')
    result = {'schema_version': 1, 'release_qualified': False,
              'scope': 'Development filesystem snapshot; preserves aliases/modes, does not by itself close dependencies or establish relocation.',
              'archive': archive.name, 'archive_bytes': archive.stat().st_size,
              'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': records}
    with manifest.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    return {key: value for key, value in result.items() if key != 'files'}


def restore_archive(archive, manifest, destination):
    """Verify delivered bytes, extract safely, restore modes and publish atomically."""
    archive, destination = Path(archive), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('refusing to replace an extracted snapshot')
    if archive.stat().st_size != manifest['archive_bytes'] or hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise ValueError('delivered archive checksum differs from snapshot manifest')
    records = manifest['files']
    if records.get('.', {}).get('type') != 'directory':
        raise ValueError('snapshot has no root directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.restore-sdk-') as temporary:
        extracted = Path(temporary)
        with tarfile.open(archive) as tar:
            expected = set()
            for relative, record in records.items():
                path = Path(relative)
                if not relative or path.is_absolute() or path.drive or '..' in path.parts or path.as_posix() != relative:
                    raise ValueError('unsafe snapshot file path')
                name = 'sdk' if relative == '.' else 'sdk/' + relative
                expected.add(name)
                member = tar.getmember(name)
                if member.mode != record['mode'] or not 0 <= member.mode <= 0o777:
                    raise ValueError('archive mode differs from validated snapshot')
                valid_type = {'file': member.isfile(), 'directory': member.isdir(), 'symlink': member.issym()}
                if not valid_type.get(record['type'], False):
                    raise ValueError('archive file type differs from validated snapshot')
                if record['type'] == 'symlink' and member.linkname != record['target']:
                    raise ValueError('archive alias differs from validated snapshot')
            members = tar.getmembers()
            if len(members) != len(expected) or {member.name for member in members} != expected:
                raise ValueError('archive file set differs from validated snapshot')
            tar.extractall(extracted, filter='data')
        root = (extracted / 'sdk').resolve(strict=True)
        # The data filter removes some directory/file permission bits; macOS
        # also creates symlinks with a different default mode than Linux.
        for relative, record in records.items():
            path = root / relative
            if stat.S_IMODE(path.lstat().st_mode) != record['mode']:
                path.chmod(record['mode'], follow_symlinks=record['type'] != 'symlink')
        if inventory(root) != records:
            raise ValueError('extracted filesystem differs from validated snapshot')
        root.rename(destination)
    return destination
