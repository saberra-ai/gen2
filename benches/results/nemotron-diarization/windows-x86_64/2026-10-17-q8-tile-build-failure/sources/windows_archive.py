"""Portable Windows runtime bytes; ACLs are supplied by the destination host.

Archives carry exact file/directory identities, never aliases or executable
permission claims. Windows filename safety is checked even on a POSIX receiver.
"""
import hashlib
from contextlib import contextmanager
import itertools
import json
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tempfile
import zipfile

MAX_ENTRIES = 2048
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or len(name.encode('utf-16-le')) > 65532 or path.is_absolute() or path.as_posix() != name or '..' in path.parts
            or name == '.' or any(re.search(r'[<>:"|?*\\\x00-\x1f]', part) or part.endswith((' ', '.'))
                                  or len(part.encode('utf-16-le')) > 510
                                  or re.fullmatch(r'(?:CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³])', part.partition('.')[0].rstrip(' '), re.IGNORECASE)
                                  for part in path.parts)):
        raise ValueError('unsafe Windows runtime path: ' + repr(name))
    return path


def digest(path, limit=MAX_FILE_BYTES):
    result = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            count += len(block)
            if count > limit:
                raise ValueError('file grew beyond bounded hashing limit')
            result.update(block)
    return result.hexdigest()


def byte_inventory(root):
    root = Path(root)
    records, folded, total = {}, set(), 0
    for path in itertools.chain([root], root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if relative != '.':
            safe_relative(relative)
        key = relative.casefold()
        if key in folded or len(records) >= MAX_ENTRIES:
            raise ValueError('colliding or excessive Windows runtime entries')
        folded.add(key)
        metadata = path.lstat()
        if getattr(metadata, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Windows reparse points are not runtime payload')
        if stat.S_ISDIR(metadata.st_mode):
            records[relative] = {'type': 'directory'}
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            total += metadata.st_size
            if metadata.st_size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError('Windows runtime byte limit exceeded')
            records[relative] = {'type': 'file', 'bytes': metadata.st_size, 'sha256': digest(path)}
        else:
            raise ValueError('Windows payload requires regular files/directories without hardlinks')
    return dict(sorted(records.items()))


def validate_records(records):
    if not isinstance(records, dict) or records.get('.') != {'type': 'directory'} or len(records) > MAX_ENTRIES:
        raise ValueError('invalid Windows archive root/entry count')
    folded, total = set(), 0
    for name, record in records.items():
        if name != '.':
            path = safe_relative(name)
            for parent in path.parents:
                if records.get(parent.as_posix()) != {'type': 'directory'}:
                    raise ValueError('Windows archive parent directory missing')
        if name.casefold() in folded:
            raise ValueError('case-colliding Windows archive entries')
        folded.add(name.casefold())
        if record == {'type': 'directory'}:
            continue
        if (not isinstance(record, dict) or set(record) != {'type', 'bytes', 'sha256'} or record['type'] != 'file'
                or type(record['bytes']) is not int or not 0 <= record['bytes'] <= MAX_FILE_BYTES
                or not re.fullmatch('[0-9a-f]{64}', record['sha256'])):
            raise ValueError('invalid Windows archive file record')
        total += record['bytes']
    if total > MAX_TOTAL_BYTES:
        raise ValueError('Windows archive total byte limit exceeded')


def member_name(name, record):
    name = 'runtime' if name == '.' else 'runtime/' + name
    return name + '/' if record['type'] == 'directory' else name


def write_archive(root, output, stem='development-windows-runtime'):
    root, output = Path(root), Path(output)
    safe_relative(stem)
    if '/' in stem:
        raise ValueError('archive stem must be a filename')
    records = byte_inventory(root)
    validate_records(records)
    archive = output / (stem + '.zip')
    with archive.open('xb') as raw:
        with zipfile.ZipFile(raw, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=False) as bundle:
            for name, record in records.items():
                info = zipfile.ZipInfo(member_name(name, record), date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                directory = record['type'] == 'directory'
                info.external_attr = ((stat.S_IFDIR | 0o755) if directory else (stat.S_IFREG | 0o644)) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                data = b''
                if not directory:
                    with (root / name).open('rb') as stream:
                        data = stream.read(record['bytes'] + 1)
                    if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                        raise ValueError('runtime changed while writing archive')
                bundle.writestr(info, data)
    if byte_inventory(root) != records:
        raise ValueError('runtime filesystem changed while writing archive')
    manifest = {'schema_version': 1, 'release_qualified': False, 'archive': archive.name,
                'scope': 'Exact portable file bytes/directory names; no Windows ACL or runtime qualification.',
                'archive_bytes': archive.stat().st_size, 'archive_sha256': digest(archive, MAX_TOTAL_BYTES + 64 * 1024 * 1024), 'files': records}
    with (output / (stem + '-files.json')).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')
    return manifest


@contextmanager
def bounded_zip(archive, expected):
    """Validate the small central directory before ZipFile allocates ZipInfo.

    Our writer needs no ZIP64, split archives, extras or comments. Do not trust
    the EOCD entry count alone: stream through the exact expected directory.
    """
    with archive.open('rb') as raw:
        raw.seek(0, 2)
        length = raw.tell()
        if length < 22:
            raise ValueError('truncated Windows ZIP end record')
        raw.seek(length - 22)
        signature, disk, central_disk, count, total, size, offset, comment = struct.unpack('<4s4H2IH', raw.read(22))
        expected_size = sum(46 + len(name.encode('utf-8')) for name in expected)
        if (signature != b'PK\x05\x06' or disk or central_disk or comment or count != total
                or total != len(expected) or total > MAX_ENTRIES or size != expected_size
                or offset + size + 22 != length):
            raise ValueError('unsupported or unbounded Windows ZIP central directory')
        raw.seek(offset)
        seen = set()
        for _ in range(total):
            header = raw.read(46)
            if len(header) != 46:
                raise ValueError('truncated Windows ZIP central entry')
            fields = struct.unpack('<4s6H3I5H2I', header)
            name_length, extra_length, comment_length, entry_disk = fields[10:14]
            if (fields[0] != b'PK\x01\x02' or fields[3] not in {0, 0x800} or extra_length
                    or comment_length or entry_disk or raw.tell() + name_length > offset + size):
                raise ValueError('unsupported Windows ZIP central entry')
            name = raw.read(name_length).decode('utf-8')
            if name not in expected or name in seen or fields[9] != expected[name][1].get('bytes', 0):
                raise ValueError('Windows ZIP central entry differs from bounded manifest')
            seen.add(name)
        if raw.tell() != offset + size or seen != set(expected):
            raise ValueError('Windows ZIP central extent differs')
        raw.seek(0)
        with zipfile.ZipFile(raw) as bundle:
            yield bundle


def restore_archive(archive, manifest, destination):
    archive, destination = Path(archive), Path(destination)
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError('refusing to replace existing restored runtime')
    if (archive.stat().st_size != manifest['archive_bytes']
            or digest(archive, MAX_TOTAL_BYTES + 64 * 1024 * 1024) != manifest['archive_sha256']):
        raise ValueError('delivered Windows archive bytes differ')
    records = manifest['files']
    validate_records(records)
    expected = {member_name(name, record): (name, record) for name, record in records.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.windows-runtime-') as temporary:
        root = Path(temporary) / 'runtime'
        with bounded_zip(archive, expected) as bundle:
            members = bundle.infolist()
            if len(members) != len(expected) or {member.filename for member in members} != set(expected):
                raise ValueError('Windows archive member set differs')
            for member in members:
                _, record = expected[member.filename]
                kind = stat.S_IFDIR if record['type'] == 'directory' else stat.S_IFREG
                if (member.flag_bits & 1 or member.compress_type not in {zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED}
                        or member.create_system != 3 or stat.S_IFMT(member.external_attr >> 16) != kind
                        or member.file_size != record.get('bytes', 0)):
                    raise ValueError('unsupported Windows archive type/size/encoding')
            for name, record in sorted(records.items(), key=lambda item: (len(PurePosixPath(item[0]).parts), item[0])):
                path = root if name == '.' else root / name
                if record['type'] == 'directory':
                    path.mkdir()
                else:
                    with bundle.open(member_name(name, record)) as stream:
                        data = stream.read(record['bytes'] + 1)
                    if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                        raise ValueError('restored Windows file bytes differ')
                    with path.open('xb') as target:
                        target.write(data)
        if byte_inventory(root) != records:
            raise ValueError('restored Windows inventory differs')
        root.rename(destination)
    return destination
