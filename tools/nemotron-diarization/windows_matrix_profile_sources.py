"""One exact matrix observer patch and import-only adapters of the prior producer."""
import hashlib
from pathlib import Path
import subprocess


PATCH_TOOL = 'patches/matrix-path-profile.patch'
PROFILE_PATCH = 'gen2-matrix-path-profile.patch'
PATCH_SHA256 = '8e541d97b33801970ab17be9933373045caea71bf5ecfd752e5fee9cc88666cc'
ORIGINAL_SERIES_SHA256 = '35e50f5c92e37c10f6e6dded26f51dd290196104990224918b455c1d828b76b8'
PROFILE_SERIES_SHA256 = '34f75ee463bc0a86b27a7b50c3c98f8ddfcf4265faa7bc0f0177b83afca19186'
CPU_SOURCE = {'bytes': 142477, 'sha256': '93551fb37e0ec9ced9c4f3f0adfe87c4d04a873aec8fb23338417296bbb3f8f2'}


def once(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError('one exact profile derivation site required')
    return raw.replace(old, new)


def derive_worker(raw):
    for old, new in [
        (b'from windows_cpu_profile_metrics import ', b'from windows_matrix_profile_metrics import '),
        (b'from windows_cpu_profile_verify import ', b'from windows_matrix_profile_verify import '),
        (b'from windows_cpu_profile_sources import ', b'from windows_matrix_profile_sources import ')]:
        raw = once(raw, old, new)
    return raw


def derive_verifier(raw):
    return once(raw, b'from windows_cpu_profile_sources import ', b'from windows_matrix_profile_sources import ')


def derive_compiler(raw):
    for old, new in [
        (b'from windows_cpu_profile_metrics import (', b'from windows_matrix_profile_metrics import ('),
        (b'from windows_cpu_profile_worker import ', b'from windows_matrix_profile_worker import '),
        (b'from windows_cpu_profile_metrics import PCM_SHA256', b'from windows_matrix_profile_metrics import PCM_SHA256')]:
        raw = once(raw, old, new)
    return raw


def upstream_series(source):
    raw = subprocess.check_output(['git', 'show', 'HEAD:patches/series'], cwd=source)
    names = [line.split('#', 1)[0].strip() for line in raw.decode().splitlines()]
    names = [name for name in names if name]
    if len(names) != 19 or len(set(names)) != 19 or not raw.endswith(b'\n'):
        raise ValueError('exact nineteen-patch upstream series required')
    for name in names:
        path = Path(source) / 'patches' / name
        if (path.is_symlink() or path.read_bytes() != subprocess.check_output(
                ['git', 'show', 'HEAD:patches/' + name], cwd=source)):
            raise ValueError('original GGML patch payload changed')
    return raw


def profile_patch():
    raw = (Path(__file__).resolve().parent / PATCH_TOOL).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PATCH_SHA256:
        raise ValueError('diagnostic CPU profile patch changed')
    return raw


def profile_source_hashes(original):
    if original.get('patches/series') != ORIGINAL_SERIES_SHA256:
        raise ValueError('original pinned patch-series identity required')
    return {**original, 'patches/series': PROFILE_SERIES_SHA256}


def install_profile(source, output):
    source, output = Path(source), Path(output)
    series = source / 'patches/series'
    original = upstream_series(source)
    target = source / 'patches' / PROFILE_PATCH
    if series.is_symlink() or series.read_bytes() != original or target.exists() or target.is_symlink():
        raise ValueError('fresh original series required before timing patch')
    raw = profile_patch()
    with target.open('xb') as stream:
        stream.write(raw)
    series.write_bytes(original + PROFILE_PATCH.encode() + b'\n')
    (output / PROFILE_PATCH).write_bytes(raw)
    verify_profile_series(source)


def verify_profile_series(source):
    source = Path(source)
    original = upstream_series(source)
    series, patch = source / 'patches/series', source / 'patches' / PROFILE_PATCH
    if (series.is_symlink() or patch.is_symlink() or series.read_bytes() != original + PROFILE_PATCH.encode() + b'\n'
            or patch.read_bytes() != profile_patch()):
        raise ValueError('exact appended diagnostic profile series required')
