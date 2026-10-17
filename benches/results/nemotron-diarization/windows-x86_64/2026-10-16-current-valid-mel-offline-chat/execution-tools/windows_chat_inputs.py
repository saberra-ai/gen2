"""Qualification-only model mount, separate from bounded runtime ZIP inputs."""
from pathlib import Path
import stat

from windows_archive import digest, safe_relative


def model_inventory(root, lock):
    root = Path(root)
    entries = lock['artifacts']
    if not 1 <= len(entries) <= 8:
        raise ValueError('bounded chat fixture list required')
    expected = {entry['name']: entry for entry in entries}
    if len(expected) != len(entries) or set(path.name for path in root.iterdir()) != set(expected):
        raise ValueError('chat mount must contain exactly the locked fixture files')
    if root.is_symlink() or getattr(root.lstat(), 'st_file_attributes', 0) & 0x400:
        raise ValueError('chat mount cannot be an alias')
    records = {}
    for name, entry in expected.items():
        safe_relative(name)
        if '/' in name or type(entry['size']) is not int or not 0 < entry['size'] <= 512 * 1024 * 1024:
            raise ValueError('unsupported chat fixture name/size')
        path = root / name
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or getattr(info, 'st_file_attributes', 0) & 0x400
                or info.st_size != entry['size'] or digest(path, entry['size']) != entry['sha256']):
            raise ValueError('chat fixture bytes/type differ from immutable lock')
        records[name] = {'bytes': info.st_size, 'sha256': entry['sha256']}
    return records
