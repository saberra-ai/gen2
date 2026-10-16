"""Bind an opaque GitHub artifact and every extracted member before publication."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

from build_valid_mel_candidate import descriptor


def admit_delivery(archive, metadata, expected, destination):
    archive, metadata, destination = Path(archive), Path(metadata), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('fresh artifact destination required')
    if archive.is_symlink() or metadata.is_symlink():
        raise ValueError('artifact inputs must be regular files')
    api = json.loads(metadata.read_text())
    if (api['id'] != expected['id'] or api['name'] != expected['name']
            or api['workflow_run']['id'] != expected['run_id']
            or api['workflow_run']['head_sha'] != expected['source_commit']
            or api['size_in_bytes'] != expected['zip']['bytes']
            or api['digest'] != 'sha256:' + expected['zip']['sha256']
            or descriptor(archive) != expected['zip']):
        raise ValueError('artifact run/source/bytes differ from frozen delivery')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        members = bundle.infolist()
        if len(members) != len(expected['files']) or {m.filename for m in members} != set(expected['files']):
            raise ValueError('artifact member set differs from frozen delivery')
        stage = Path(temporary) / 'delivery'
        stage.mkdir()
        for member in members:
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (path.is_absolute() or Path(member.filename).drive or '..' in path.parts or path.as_posix() != member.filename
                    or not path.parts or member.is_dir() or stat.S_IFMT(mode) not in {0, stat.S_IFREG}):
                raise ValueError('unsafe artifact member')
            frozen = expected['files'][member.filename]
            if member.file_size != frozen['bytes']:
                raise ValueError('artifact member size differs')
            raw = bundle.read(member)
            if len(raw) != frozen['bytes'] or hashlib.sha256(raw).hexdigest() != frozen['sha256']:
                raise ValueError('artifact member checksum differs')
            output = stage / member.filename
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(raw)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('artifact destination appeared during admission')
        stage.rename(destination)
    return api
