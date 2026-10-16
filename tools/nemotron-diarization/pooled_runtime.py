"""Explicit pooled candidate; preserve the frozen original profile consumers."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from build_valid_mel_candidate import descriptor
from platform_valid_mel_sources import verify_completed_source as verify_original_completed_source
import runtime_profiles as original
from runtime_profiles import PROFILE, profile_provenance


POOLED_PROFILE = 'gen2-valid-mel-pool-v1'
POOL_FILES = {'src/runtime/ggml/backend.cpp', 'src/runtime/ggml/runtime.h'}


def load_profile(profile=PROFILE):
    if profile == PROFILE:
        return original.load_profile()
    if profile != POOLED_PROFILE:
        raise ValueError('unsupported native runtime profile')
    tools = Path(__file__).resolve().parent
    path = tools / 'runtime-valid-mel-pool.lock.json'
    lock = json.loads(path.read_text())
    parent_path, parent = original.load_profile()
    if lock.get('parent_runtime_profile') != {'path': parent_path.name, **descriptor(parent_path)}:
        raise ValueError('pool profile parent differs')
    patches = lock['runtime']['compatibility_patches']
    if (len(patches) != 2 or patches[:1] != parent['runtime']['compatibility_patches']
            or patches[1].get('path') != 'patches/persistent-cpu-pool.patch'):
        raise ValueError('unexpected pooled patch series')
    patch = patches[1]
    if descriptor(tools / patch['path']) != {key: patch[key] for key in ['bytes', 'sha256']}:
        raise ValueError('pool patch bytes differ')
    if any(set(patch.get(key, {})) != POOL_FILES for key in ['source_sha256_before', 'source_sha256_after']):
        raise ValueError('pool source provenance is incomplete')
    expected = copy.deepcopy(parent)
    expected.update(profile=POOLED_PROFILE, profile_status=lock['profile_status'],
                    parent_runtime_profile=lock['parent_runtime_profile'])
    runtime = expected['runtime']
    runtime['cmake_definitions']['NEMO_SPEECH_VERSION_METADATA'] = POOLED_PROFILE
    runtime['reported_version'] = 'nemo-speech-asr 0.2.0+' + POOLED_PROFILE
    runtime['compatibility_patches'] = patches
    runtime['source_sha256'].update(patch['source_sha256_after'])
    if lock != expected:
        raise ValueError('pool profile changes unrelated runtime settings')
    return path, lock


def compatibility_patch_bytes(lock):
    tools = Path(__file__).resolve().parent
    return b''.join((tools / item['path']).read_bytes() for item in lock['runtime']['compatibility_patches'])


def validate_provenance(record):
    profile = record.get('runtime_profile')
    if profile == PROFILE:
        return original.validate_provenance(record)
    path, lock = load_profile(profile)
    for key, expected in profile_provenance(path, lock).items():
        if record.get(key) != expected:
            raise ValueError('pooled artifact provenance differs: ' + key)
    return lock


def _verify_pooled_source(source, lock, build=None):
    source = Path(source).resolve()
    runtime = lock['runtime']
    capture = lambda args, root=source: subprocess.check_output(args, cwd=root)
    if capture(['git', 'rev-parse', 'HEAD']).decode().strip() != runtime['revision']:
        raise ValueError('pooled native base differs')
    if capture(['git', 'diff', '--binary', '--no-ext-diff', '--ignore-submodules=all']) != compatibility_patch_bytes(lock):
        raise ValueError('pooled source differs from declared patches')
    if capture(['git', 'diff', '--cached', '--binary', '--no-ext-diff', '--ignore-submodules=all']):
        raise ValueError('staged pooled source changes')
    for item in runtime['compatibility_patches']:
        for name, expected in item['source_sha256_before'].items():
            if hashlib.sha256(capture(['git', 'show', runtime['revision'] + ':' + name])).hexdigest() != expected:
                raise ValueError('pre-patch source differs from pinned base: ' + name)
    for name, expected in runtime['source_sha256'].items():
        if descriptor(source / name)['sha256'] != expected:
            raise ValueError('pooled source bytes differ: ' + name)
    llama = source / 'llama.cpp'
    if (capture(['git', 'rev-parse', 'HEAD'], llama).decode().strip() != runtime['llama_revision']
            or capture(['git', 'status', '--porcelain', '--untracked-files=no'], llama)):
        raise ValueError('pooled pristine llama differs')
    if build is not None:
        build = Path(build).resolve()
        if not build.is_relative_to(source / 'build'):
            raise ValueError('compiler tree must be inside the isolated source build directory')
    for tree in [source, llama]:
        for flags in [[], ['--ignored']]:
            names = capture(['git', 'ls-files', '--others', *flags, '--exclude-standard'], tree).decode().splitlines()
            if any(not (build is not None and tree == source and (source / name).resolve().is_relative_to(build)) for name in names):
                raise ValueError('extra pooled compiler inputs')


def verify_source(source, lock):
    if lock['profile'] == PROFILE:
        return original.verify_source(source, lock)
    _verify_pooled_source(source, lock)


def verify_completed_source(source, build, lock, patch, windows):
    if lock['profile'] == PROFILE:
        return verify_original_completed_source(source, build, lock, patch, windows)
    if windows or Path(patch).read_bytes() != compatibility_patch_bytes(lock):
        raise ValueError('unsupported pooled source composition')
    _verify_pooled_source(source, lock, build)
