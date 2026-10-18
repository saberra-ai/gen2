"""Materialize the frozen graph observer on Linux; never configure a compiler or load a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

NATIVE_SOURCE = 'ce65591dffd22e387416f0950a88f607b1761d44'
NATIVE_PROTOCOL = {'bytes': 674377, 'sha256': 'e033090bdb03a947326c8decd633e1d92c816170694ff1369c623c265ed54d52'}
PROCESS_HELPER = {'bytes': 22591, 'sha256': '6b7f4c6d3a9a672ad0a89c790fa0d51608adad4c3667395c2da1329c68e48b08'}
START_FLOOR = 8 * 1024**3


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def descriptor(path):
    digest, count = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            count += len(block)
    return {'bytes': count, 'sha256': digest.hexdigest()}


def admit_native(raw):
    if identity(raw) != NATIVE_PROTOCOL:
        raise ValueError('exact frozen graph-only native protocol required')
    value = json.loads(raw)
    if (value['order'] != [False, True, True, False] or value['expected_graphs'] != 84
            or value['expected_matrices'] != 129 or value['matrix_emission_bytes_per_row'] != 2048
            or value['variants']['4']['reported_version'] != 'nemo-speech-asr 0.2.0+gen2-diagnostic-graph-cpu-v1'):
        raise ValueError('original graph-only diagnostic identity and coverage required')
    return value


def admit_output(repo, output, free_bytes, platform_name):
    if platform_name != 'linux' or free_bytes < START_FLOOR:
        raise ValueError('Linux source-only receiver and original 8 GiB floor required')
    if (output.exists() or output.is_symlink() or not output.parent.is_dir()
            or output.is_relative_to(repo) or repo.is_relative_to(output)):
        raise ValueError('fresh disjoint source-only output required')


def merge_inputs(fixed, additions):
    for path, expected in additions.items():
        if path in fixed and fixed[path] != expected:
            raise ValueError('source-only protocol cannot replace a native binding')
        fixed[path] = expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    if args.output.is_symlink():
        parser.error('fresh regular source-only output required')
    repo, output = tools.parents[1], args.output.resolve()
    admit_output(repo, output, shutil.disk_usage(output.parent).free, sys.platform)
    native_path = tools / 'windows-graph-cpu-profile.protocol.json'
    native_raw = native_path.read_bytes()
    native = admit_native(native_raw)
    protocol_path = tools / 'graph-cpu-source-admission.protocol.json'
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    if (protocol['schema_version'] != 1 or protocol['native_source'] != NATIVE_SOURCE
            or protocol['native_protocol'] != NATIVE_PROTOCOL
            or protocol['minimum_free_bytes'] != START_FLOOR
            or protocol['expected_acquisition_commands'] != 8 or protocol['expected_composition_commands'] != 11
            or protocol['expected_materialized_files'] != 3425 or protocol['compiler_sdk_model_execution'] is not False):
        raise ValueError('exact source-only protocol and original floor required')
    fixed = {tools / name: value for name, value in native['tools'].items()}
    fixed.update({repo / name: value for name, value in native['repository_files'].items()})
    merge_inputs(fixed, {repo / name: value for name, value in protocol['files'].items()})
    fixed[native_path] = NATIVE_PROTOCOL
    fixed[protocol_path] = identity(protocol_raw)
    helper = tools / 'remote_q8_apack_replay.py'
    if fixed.get(helper) != PROCESS_HELPER or Path(__file__).resolve() not in fixed:
        raise ValueError('bound process helper and source-only controller required')

    def frozen():
        for path, expected in fixed.items():
            if descriptor(path) != expected:
                raise ValueError('source-only bound input changed: ' + str(path))

    frozen()
    # Only after checking the complete closure may project code execute.
    from frozen_git_sources import verify_git_sources
    from platform_native import capture
    commit = capture(['git', 'rev-parse', 'HEAD'], cwd=repo)
    verify_git_sources(repo, commit, {path.relative_to(repo).as_posix(): value for path, value in fixed.items()})
    from remote_q8_apack_replay import bounded_process
    from remote_thread_cpu_replay import inherited_process
    if not args.worker:
        output.mkdir()
        outer = {'completed': False, 'compiled_sdk_or_model': False, 'source_commit': commit,
                 'minimum_free_bytes': START_FLOOR, 'scope': __doc__}
        try:
            row = bounded_process([sys.executable, '-B', str(Path(__file__).resolve()),
                '--output', str(output / 'stage'), '--worker'], output / 'stage.stdout.log',
                output / 'stage.stderr.log', cwd=repo,
                env={**os.environ, 'GEN2_REMOTE_OWNED_STAGE': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
                timeout=900, limit=8 * 1024**2)
            raw = (output / 'stage/qualification.json').read_bytes()
            result = json.loads(raw)
            if (not row['completed'] or not row['leader_reaped'] or not row['group_gone']
                    or result['completed'] is not True or result['compiled_sdk_or_model'] is not False
                    or len(result['materialized_files']) != 3425 or len(result['commands']) != 19
                    or any(command['completed'] is not True or command['leader_reaped'] is not True
                        or command['returncode'] != 0 or command['stage_group'] != row['pid']
                        or command['group_owner'] != 'outer bounded stage supervisor'
                        for command in result['commands'])):
                raise ValueError('complete source-only owned-stage proof required')
            frozen()
            outer.update(completed=True, stage=row, stage_qualification=identity(raw))
        except BaseException as error:
            outer['error'] = repr(error)
            raise
        finally:
            (output / 'qualification.json').write_text(json.dumps(outer, indent=2) + '\n')
        print(json.dumps(outer))
        return
    if os.environ.get('GEN2_REMOTE_OWNED_STAGE') != '1' or os.getpgrp() != os.getpid():
        raise ValueError('source worker must inherit its registered outer process group')
    from pooled_runtime import BOUNDED_PROFILE, load_profile, compatibility_patch_bytes
    from build_valid_mel_candidate import source_inventory
    from windows_compat import apply_windows_lf_compat
    from windows_worker_metrics import DIARIZER, BACKEND, transform
    from windows_graph_cpu_profile_sources import install_profile, CPU_SOURCE
    from windows_graph_cpu_profile_verify import verify_completed_source
    from windows_graph_cpu_profile_metrics import canonical_audit
    from ggml_graph_cpu_profile_source import PATH
    _, lock = load_profile(BOUNDED_PROFILE)
    output.mkdir()
    evidence = output / 'evidence'
    evidence.mkdir()
    (output / 'protocol.json').write_bytes(protocol_raw)
    q = {'completed': False, 'source_commit': commit, 'native_protocol': NATIVE_PROTOCOL,
         'scope': __doc__, 'compiled_sdk_or_model': False, 'commands': [],
         'minimum_free_bytes': START_FLOOR, 'initial_free_bytes': shutil.disk_usage(output).free,
         'inputs': {path.relative_to(repo).as_posix(): value for path, value in fixed.items()}}

    def persist():
        (output / 'qualification.json').write_text(json.dumps(q, indent=2) + '\n')

    environment = {**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                   'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1',
                   'GIT_CONFIG_COUNT': '2', 'GIT_CONFIG_KEY_0': 'core.autocrlf',
                   'GIT_CONFIG_VALUE_0': 'false', 'GIT_CONFIG_KEY_1': 'core.eol', 'GIT_CONFIG_VALUE_1': 'lf'}

    def run(name, command, cwd=repo, phase='composition'):
        frozen()
        free = shutil.disk_usage(output).free
        if free < START_FLOOR:
            raise ValueError('original 8 GiB source-only floor no longer met')
        row = {'name': name, 'phase': phase, 'free_bytes_before': free, 'completed': False}
        q['commands'].append(row)
        persist()
        try:
            row.update(inherited_process(list(map(str, command)), evidence / (name + '.stdout.log'),
                evidence / (name + '.stderr.log'), cwd=cwd, env=environment, timeout=180, limit=8 * 1024**2))
        finally:
            persist()
        if not row['completed'] or not row['leader_reaped'] or row['returncode'] != 0:
            raise ValueError('source-only command ownership/cleanup failed')
        frozen()
        print('PASS ' + name, flush=True)

    persist()
    try:
        pristine = output / 'pristine'
        pristine.mkdir()
        for name, path, url, revision in [
            ('native', pristine / 'native', 'https://github.com/NVIDIA/NeMo-Speech.cpp.git', lock['runtime']['revision']),
            ('llama', pristine / 'native/llama.cpp', 'https://github.com/ggml-org/llama.cpp.git', lock['runtime']['llama_revision'])]:
            for suffix, command in [
                ('init', ['git', 'init', path]), ('origin', ['git', '-C', path, 'remote', 'add', 'origin', url]),
                ('fetch', ['git', '-C', path, 'fetch', '--depth=1', 'origin', revision]),
                ('checkout', ['git', '-C', path, 'checkout', '--detach', revision])]:
                run(name + '-' + suffix, command, phase='acquisition')
        source = output / 'source'
        for name, origin, target, revision in [
            ('native', pristine / 'native', source, lock['runtime']['revision']),
            ('llama', pristine / 'native/llama.cpp', source / 'llama.cpp', lock['runtime']['llama_revision'])]:
            run('clone-' + name, ['git', 'clone', '--no-hardlinks', '--no-checkout', origin, target])
            run('checkout-' + name, ['git', '-C', target, 'checkout', '--detach', revision])
        for index, item in enumerate(lock['runtime']['compatibility_patches']):
            patch = tools / item['path']
            run('check-patch-' + str(index), ['git', 'apply', '--check', patch], source)
            run('apply-patch-' + str(index), ['git', 'apply', patch], source)
            shutil.copyfile(patch, evidence / patch.name)
        patch = evidence / 'bounded-inference.patch'
        patch.write_bytes(compatibility_patch_bytes(lock))
        for name in (DIARIZER, BACKEND):
            path = source / name
            path.write_bytes(transform(name, path.read_bytes(), 4))
        install_profile(source, evidence)
        verify_completed_source(source, None, lock, patch, True, native['variants']['4']['source_sha256'])
        apply_windows_lf_compat(source, evidence)
        build = source / 'build/gen2-cpu-diar'
        verify_completed_source(source, build, lock, patch, True, native['variants']['4']['source_sha256'])
        materialized = build / '_deps/llama.cpp'
        run('materialize-twenty-patches', ['cmake', '-DSOURCE_DIR=' + str(source / 'llama.cpp'),
            '-DPATCH_DIR=' + str(source / 'patches'), '-DDEST_DIR=' + str(materialized),
            '-P', source / 'cmake/llama_cpp.cmake'])
        expected = canonical_audit()['files'].copy()
        expected[PATH] = CPU_SOURCE
        actual = source_inventory(materialized)
        if actual != expected or len(actual) != 3425:
            raise ValueError('complete 3425-file composition differs')
        if ([row['phase'] for row in q['commands']] != ['acquisition'] * 8 + ['composition'] * 11):
            raise ValueError('exact eight acquisitions and eleven compositions required')
        verify_completed_source(source, build, lock, patch, True, native['variants']['4']['source_sha256'])
        frozen()
        q.update(completed=True, materialized_files=actual, changed_ggml_files=[PATH],
                 profiled_cpu_source=descriptor(materialized / PATH), source_hashes=native['variants']['4']['source_sha256'])
    except BaseException as error:
        q['error'] = repr(error)
        raise
    finally:
        persist()
    print(json.dumps({'completed': True, 'commands': len(q['commands']), 'materialized_files': len(actual),
                      'compiled_sdk_or_model': False}))


if __name__ == '__main__':
    main()
