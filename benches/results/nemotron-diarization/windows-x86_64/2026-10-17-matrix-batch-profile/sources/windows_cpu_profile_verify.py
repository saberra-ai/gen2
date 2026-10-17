"""Exact diagnostic source admission; never admits a production SDK."""
import hashlib
import os
from pathlib import Path
import subprocess

from build_valid_mel_candidate import descriptor
from pooled_runtime import BOUNDED_PROFILE, compatibility_patch_bytes
from windows_compat import ORIGINAL_SHA256, rewrite_materializer
from windows_cpu_profile_sources import PROFILE_PATCH, verify_profile_series


INFERENCE_FILES = {
    'src/asr/diar/diar_pipeline.cpp', 'src/asr/diar/diarizer.cpp',
    'src/runtime/ggml/backend.cpp', 'src/runtime/ggml/runtime.h',
}


def checkout_environment(environment):
    # Diarization builds consume Git-tracked source/pointers, never unrelated
    # TTS LFS payloads. Keep this policy in child mappings, not Git globals.
    return {**environment, 'GIT_LFS_SKIP_SMUDGE': '1'}


def verify_completed_source(source, build, lock, patch, windows, source_hashes):
    source = Path(source).resolve()
    if lock.get('profile') != BOUNDED_PROFILE or not windows:
        raise ValueError('bounded Windows source profile required')
    if build is not None:
        build = Path(build).resolve()
        if not build.is_relative_to(source / 'build'):
            raise ValueError('compiler tree must be inside isolated source build directory')
    environment = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}

    def capture(argv, root=source):
        return subprocess.check_output(['git', '-c', 'core.autocrlf=false', '-c', 'core.eol=lf', *argv],
                                       cwd=root, env=environment)

    runtime = lock['runtime']
    verify_profile_series(source)
    if capture(['rev-parse', 'HEAD']).decode().strip() != runtime['revision']:
        raise ValueError('bounded Windows native base differs')
    if capture(['diff', '--cached', '--binary', '--no-ext-diff', '--ignore-submodules=all']):
        raise ValueError('staged bounded Windows source changes')
    changed = set(capture(['diff', '--name-only', '--ignore-submodules=all']).decode().splitlines())
    expected = INFERENCE_FILES | {'patches/series'} | ({'cmake/llama_cpp.cmake'} if build is not None else set())
    if changed != expected:
        raise ValueError('bounded Windows source changes exceed declared composition')
    series = compatibility_patch_bytes(lock)
    if Path(patch).read_bytes() != series:
        raise ValueError('bounded Windows inference patch series differs')
    for item in runtime['compatibility_patches']:
        for name, expected_hash in item['source_sha256_before'].items():
            if hashlib.sha256(capture(['show', runtime['revision'] + ':' + name])).hexdigest() != expected_hash:
                raise ValueError('bounded Windows pre-patch source differs: ' + name)
    if set(source_hashes) != set(runtime['source_sha256']):
        raise ValueError('diagnostic source coverage differs')
    for name, expected_hash in source_hashes.items():
        if descriptor(source / name)['sha256'] != expected_hash:
            raise ValueError('bounded Windows source bytes differ: ' + name)
    if build is not None:
        original = capture(['show', 'HEAD:cmake/llama_cpp.cmake'])
        if (hashlib.sha256(original).hexdigest() != ORIGINAL_SHA256
                or (source / 'cmake/llama_cpp.cmake').read_bytes() != rewrite_materializer(original)):
            raise ValueError('bounded Windows source differs from exact LF overlay')
    llama = source / 'llama.cpp'
    if (capture(['rev-parse', 'HEAD'], llama).decode().strip() != runtime['llama_revision']
            or capture(['status', '--porcelain', '--untracked-files=no'], llama)):
        raise ValueError('bounded Windows pristine llama differs')
    for tree in [source, llama]:
        for flags in [[], ['--ignored']]:
            extras = capture(['ls-files', '--others', *flags, '--exclude-standard'], tree).decode().splitlines()
            if any(not ((tree == source and name == 'patches/' + PROFILE_PATCH)
                        or (build is not None and tree == source and (source / name).resolve().is_relative_to(build)))
                   for name in extras):
                raise ValueError('unrecorded bounded Windows compiler inputs')
