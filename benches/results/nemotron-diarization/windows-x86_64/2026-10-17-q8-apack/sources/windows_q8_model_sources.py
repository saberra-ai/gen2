"""Exact SDK composition for the separately versioned full-model Q8 trial."""
import hashlib
from pathlib import Path

from q8_mx4_source import ORIGINAL, PATH, transform
from windows_cpu_profile_sources import once, upstream_series


PATCH_TOOL = 'patches/q8-model-mx4.patch'
Q8_PATCH = 'gen2-q8-model-mx4.patch'
ORIGINAL_SERIES_SHA256 = '35e50f5c92e37c10f6e6dded26f51dd290196104990224918b455c1d828b76b8'
GUARD = b'#define GEN2_Q8_MX4_EXPECTED 1\n'


def validate_variant(variant):
    if type(variant) is not str or variant not in ('control', 'mx4'):
        raise ValueError('explicit control or mx4 SDK variant required')
    return variant


def candidate_source():
    raw = (Path(__file__).resolve().parent / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
    if {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} != ORIGINAL:
        raise ValueError('canonical SGEMM fixture changed')
    return GUARD + transform(raw)


def q8_patch():
    import difflib
    tools = Path(__file__).resolve().parent
    original = (tools / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
    expected = ''.join(difflib.unified_diff(original.decode().splitlines(True),
        candidate_source().decode().splitlines(True), fromfile='a/' + PATH, tofile='b/' + PATH)).encode()
    raw = (tools / PATCH_TOOL).read_bytes()
    if raw != expected:
        raise ValueError('exact integrated Q8 patch required')
    return raw


def series_bytes(original, variant):
    validate_variant(variant)
    if hashlib.sha256(original).hexdigest() != ORIGINAL_SERIES_SHA256:
        raise ValueError('original pinned nineteen-patch series required')
    return original if variant == 'control' else original + Q8_PATCH.encode() + b'\n'


def source_hashes(original_hashes, original_series, variant):
    validate_variant(variant)
    if original_hashes.get('patches/series') != ORIGINAL_SERIES_SHA256:
        raise ValueError('original native source hashes required')
    return {**original_hashes, 'patches/series': hashlib.sha256(series_bytes(original_series, variant)).hexdigest()}


def install_q8_patch(source, output, variant):
    source, output = Path(source), Path(output)
    original = upstream_series(source)
    expected = series_bytes(original, variant)
    series, target = source / 'patches/series', source / 'patches' / Q8_PATCH
    if series.is_symlink() or series.read_bytes() != original or target.exists() or target.is_symlink():
        raise ValueError('fresh original series required before Q8 composition')
    if variant == 'mx4':
        raw = q8_patch()
        with target.open('xb') as stream:
            stream.write(raw)
        series.write_bytes(expected)
        with (output / Q8_PATCH).open('xb') as stream:
            stream.write(raw)
    verify_q8_series(source, variant)


def verify_q8_series(source, variant):
    source = Path(source)
    expected = series_bytes(upstream_series(source), variant)
    series, patch = source / 'patches/series', source / 'patches' / Q8_PATCH
    if series.is_symlink() or series.read_bytes() != expected:
        raise ValueError('exact variant patch series required')
    if variant == 'control':
        if patch.exists() or patch.is_symlink():
            raise ValueError('control must not contain candidate patch')
    elif patch.is_symlink() or patch.read_bytes() != q8_patch():
        raise ValueError('exact candidate patch bytes required')


def derive_worker(raw):
    raw = raw[:raw.index(b'\n\ndef installed_files(root):')]
    raw = once(raw, b'from windows_worker_metrics import ', b'from windows_q8_model_metrics import ')
    raw = once(raw, b'from windows_worker_sources_v2 import checkout_environment, verify_completed_source',
        b'from windows_q8_model_verify import checkout_environment, verify_completed_source\n'
        b'from windows_q8_model_sources import install_q8_patch, validate_variant')
    raw = once(raw, b'def __init__(self, source, prefix, output, jobs, cap, protocol, frozen):',
        b'def __init__(self, source, prefix, output, jobs, cap, protocol, frozen, variant):')
    raw = once(raw, b'        super().__init__(source, prefix, output, jobs, True)',
        b'        self.variant = validate_variant(variant)\n        super().__init__(source, prefix, output, jobs, True)')
    raw = once(raw, b'metadata(cap)', b'metadata(cap, self.variant)')
    raw = once(raw, b"self.protocol['variants'][str(self.cap)]['source_sha256'])",
        b"self.protocol['variants'][self.variant]['source_sha256'], self.variant)")
    raw = once(raw, b"(self.tools / 'native_stream_probe_valid_mel.cpp').read_bytes(), self.cap):",
        b"(self.tools / 'native_stream_probe_valid_mel.cpp').read_bytes(), self.cap, self.variant):")
    site = b"        self.verify_sources()\n        (self.output / 'diagnostic-inference.diff')"
    raw = once(raw, site, b'        install_q8_patch(self.source, self.output, self.variant)\n' + site)
    raw = once(raw, b'derive_probe(original.read_bytes(), self.cap)', b'derive_probe(original.read_bytes(), self.cap, self.variant)')
    raw = once(raw, b'version(self.cap)', b'version(self.cap, self.variant)')
    return once(raw, b'One exact historical quoted version literal replaced with diagnostic-only cap identity.',
        b'One exact historical quoted version literal replaced with diagnostic-only variant identity.')


def derive_verifier(raw):
    raw = once(raw, b'def verify_completed_source(source, build, lock, patch, windows, source_hashes):',
        b'def verify_completed_source(source, build, lock, patch, windows, source_hashes, variant):')
    site = b'from windows_compat import ORIGINAL_SHA256, rewrite_materializer\n'
    raw = once(raw, site, site + b'from windows_q8_model_sources import Q8_PATCH, verify_q8_series\n')
    raw = once(raw, b'    expected = INFERENCE_FILES | ',
        b"    expected = INFERENCE_FILES | ({'patches/series'} if variant == 'mx4' else set()) | ")
    site = b"    runtime = lock['runtime']\n"
    raw = once(raw, site, site + b'    verify_q8_series(source, variant)\n')
    site = b'            if any(not (build is not None and tree == source and (source / name).resolve().is_relative_to(build))'
    return once(raw, site,
        b"            if any(not ((tree == source and variant == 'mx4' and name == 'patches/' + Q8_PATCH)\n"
        b'                        or (build is not None and tree == source and (source / name).resolve().is_relative_to(build)))')


def derive_compiler(raw):
    raw = raw[:raw.index(b'\n\ndef main():')]
    raw = once(raw, b'from windows_compiler_fp_metrics import (', b'from windows_q8_model_metrics import (')
    raw = once(raw, b'from windows_worker_diagnostic_v2 import Diagnostic as WorkerDiagnostic',
        b'from windows_q8_model_worker import Diagnostic as WorkerDiagnostic')
    raw = once(raw, b'from windows_worker_metrics import PCM_SHA256, compare_native, telemetry',
        b'from windows_q8_model_metrics import PCM_SHA256, compare_native, telemetry')
    raw = once(raw, b'def __init__(self, source, prefix, output, jobs, kind, protocol, frozen):',
        b'def __init__(self, source, prefix, output, jobs, kind, protocol, frozen, variant):')
    raw = once(raw, b'super().__init__(source, prefix, output, jobs, 4, protocol, frozen)',
        b'super().__init__(source, prefix, output, jobs, 4, protocol, frozen, variant)')
    return once(raw, b'database, compiler_files, self.compiler, self.kind, self.compiler_aliases)',
        b'database, compiler_files, self.compiler, self.kind, self.compiler_aliases, self.variant)')
