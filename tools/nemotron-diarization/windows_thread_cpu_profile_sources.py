"""One exact matrix observer patch and import-only adapters of the prior producer."""
import hashlib
from pathlib import Path
import subprocess


PATCH_TOOL = 'patches/thread-cpu-profile.patch'
PROFILE_PATCH = 'gen2-thread-cpu-profile.patch'
PATCH_SHA256 = '2ea1112aac6c45436057556c603c7cd443141e05ee67d8f81a7c6b6b2a22d5dc'
ORIGINAL_SERIES_SHA256 = '35e50f5c92e37c10f6e6dded26f51dd290196104990224918b455c1d828b76b8'
PROFILE_SERIES_SHA256 = 'fd870951eb08525003777aacb630299a3f5738dcb498b6914fe550e04cde3dbd'
CPU_SOURCE = {'bytes': 148721, 'sha256': '82b72d5f985ec94e4c47807de274f6a441bcb306e3161090e36785ffa80f5ae9'}


def once(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError('one exact profile derivation site required')
    return raw.replace(old, new)


def derive_worker(raw):
    for old, new in [
        (b'from windows_cpu_profile_metrics import ', b'from windows_thread_cpu_profile_metrics import '),
        (b'from windows_cpu_profile_verify import ', b'from windows_thread_cpu_profile_verify import '),
        (b'from windows_cpu_profile_sources import ', b'from windows_thread_cpu_profile_sources import ')]:
        raw = once(raw, old, new)
    return raw


def derive_verifier(raw):
    return once(raw, b'from windows_cpu_profile_sources import ', b'from windows_thread_cpu_profile_sources import ')


def derive_compiler(raw):
    for old, new in [
        (b'from windows_cpu_profile_metrics import (', b'from windows_thread_cpu_profile_metrics import ('),
        (b'from windows_cpu_profile_worker import ', b'from windows_thread_cpu_profile_worker import '),
        (b'from windows_cpu_profile_metrics import PCM_SHA256', b'from windows_thread_cpu_profile_metrics import PCM_SHA256')]:
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
