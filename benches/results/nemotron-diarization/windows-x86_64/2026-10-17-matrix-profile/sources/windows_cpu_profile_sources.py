"""Exact diagnostic patch-series extension and mechanically derived build code."""
import hashlib
from pathlib import Path
import subprocess


PATCH_TOOL = 'patches/cpu-operation-profile.patch'
PROFILE_PATCH = 'gen2-cpu-operations-profile.patch'
PATCH_SHA256 = '6f37c2c4ad387d50957fa632e61e7b6dce8d7111ea9bac7a831198c16e1a5f52'
ORIGINAL_SERIES_SHA256 = '35e50f5c92e37c10f6e6dded26f51dd290196104990224918b455c1d828b76b8'
PROFILE_SERIES_SHA256 = 'f7b62fb539ae3a8dfc32dca518281d810c8c71eab7a99ebdf78e483ca0d7383e'
CPU_SOURCE = {'bytes': 136705, 'sha256': 'dbd58cbf43f80728ea4bdc75337aee4adb8e9651c0327af7316d70d3e949049a'}


def once(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError('one exact profile derivation site required')
    return raw.replace(old, new)


def derive_worker(raw):
    raw = raw[:raw.index(b'\n\ndef installed_files(root):')]
    raw = once(raw, b'from windows_worker_metrics import ', b'from windows_cpu_profile_metrics import ')
    raw = once(raw, b'from windows_worker_sources_v2 import checkout_environment, verify_completed_source',
        b'from windows_cpu_profile_verify import checkout_environment, verify_completed_source\n'
        b'from windows_cpu_profile_sources import install_profile')
    site = b"        self.verify_sources()\n        (self.output / 'diagnostic-inference.diff')"
    return once(raw, site, b'        install_profile(self.source, self.output)\n' + site)


def derive_verifier(raw):
    site = b'from windows_compat import ORIGINAL_SHA256, rewrite_materializer\n'
    raw = once(raw, site, site + b'from windows_cpu_profile_sources import PROFILE_PATCH, verify_profile_series\n')
    raw = once(raw, b'    expected = INFERENCE_FILES | ', b"    expected = INFERENCE_FILES | {'patches/series'} | ")
    site = b"    runtime = lock['runtime']\n"
    raw = once(raw, site, site + b'    verify_profile_series(source)\n')
    site = b'            if any(not (build is not None and tree == source and (source / name).resolve().is_relative_to(build))'
    return once(raw, site,
        b"            if any(not ((tree == source and name == 'patches/' + PROFILE_PATCH)\n"
        b'                        or (build is not None and tree == source and (source / name).resolve().is_relative_to(build)))')


def derive_compiler(raw):
    raw = raw[:raw.index(b'\n\ndef main():')]
    raw = once(raw, b'from windows_compiler_fp_metrics import (', b'from windows_cpu_profile_metrics import (')
    raw = once(raw, b'from windows_worker_diagnostic_v2 import Diagnostic as WorkerDiagnostic',
               b'from windows_cpu_profile_worker import Diagnostic as WorkerDiagnostic')
    return once(raw, b'from windows_worker_metrics import PCM_SHA256, compare_native, telemetry',
                b'from windows_cpu_profile_metrics import PCM_SHA256, compare_native, telemetry')


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
