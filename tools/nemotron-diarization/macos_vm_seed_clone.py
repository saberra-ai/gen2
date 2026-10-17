"""Enforced same-volume clone of one verified regular file; never copy fallback."""
import ctypes
import os
from pathlib import Path
import stat


def file_identity(info):
    return {'device': info.st_dev, 'inode': info.st_ino, 'bytes': info.st_size,
        'allocated_bytes': info.st_blocks * 512, 'mode': stat.S_IMODE(info.st_mode),
        'uid': info.st_uid, 'links': info.st_nlink, 'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


def _fclone(source_fd, directory_fd, name):
    library = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
    clone = library.fclonefileat
    clone.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    clone.restype = ctypes.c_int
    ctypes.set_errno(0)
    # CLONE_NOFOLLOW, no best-effort copy API or fallback.
    if clone(source_fd, directory_fd, os.fsencode(name), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def clone_regular(source, directory, name):
    source, directory = Path(source), Path(directory)
    if Path(name).name != name or name in {'', '.', '..'}:
        raise ValueError('fresh destination basename required')
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            before = os.fstat(source_fd)
            target_directory = os.fstat(directory_fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_nlink != 1
                    or stat.S_IMODE(before.st_mode) != 0o600
                    or target_directory.st_uid != os.getuid() or stat.S_IMODE(target_directory.st_mode) != 0o700
                    or before.st_dev != target_directory.st_dev):
                raise ValueError('owned private same-volume source/destination required')
            try:
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError('clone destination exists')
            _fclone(source_fd, directory_fd, name)
            target = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            after = os.fstat(source_fd)
            if file_identity(before) != file_identity(after) or file_identity(source.lstat()) != file_identity(after):
                raise ValueError('clone source changed')
            if (not stat.S_ISREG(target.st_mode) or target.st_ino == before.st_ino
                    or target.st_dev != before.st_dev or target.st_size != before.st_size
                    or target.st_uid != before.st_uid or stat.S_IMODE(target.st_mode) != 0o600 or target.st_nlink != 1):
                raise ValueError('clone destination identity differs')
            os.fsync(directory_fd)
            return {'operation': 'fclonefileat', 'returncode': 0, 'fallback_copy': False,
                'source_before': file_identity(before), 'source_after': file_identity(after),
                'destination': file_identity(target), 'content_evidence': 'OS enforced copy-on-write clone; no whole-file SHA256 claim.'}
        finally:
            os.close(directory_fd)
    finally:
        os.close(source_fd)
