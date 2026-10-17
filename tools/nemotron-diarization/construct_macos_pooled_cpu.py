"""Build an isolated pooled CPU candidate and exercise public failed-load cleanup.

No valid weights, frontend or inference are admitted. The old profile remains
the default; this candidate requires separate neural and release qualification.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from build_valid_mel_candidate import descriptor
from compare_macos_cpu_pools import model_free_policy
from macos_cpu_architecture import PORTABLE_ARM64
from macos_loaded_libraries import audit_loaded_libraries
from platform_macos_portable import SANDBOX_CONTROL, construction_policy, construction_roots
from qualification_storage import StorageGuard, atomic_json
from qualify_macos_portable_runtime import policy
from qualify_valid_mel_pnola import execute_guarded
from pooled_runtime import POOLED_PROFILE, load_profile, profile_provenance
from sdk_archive import inventory, write_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['protocol', 'upstream', 'source', 'sdk', 'package', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('actual Apple Silicon Mac required')
    upstream, source, sdk, package, output = construction_roots(
        [getattr(args, name) for name in ['upstream', 'source', 'sdk', 'package', 'output']])
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    if any(p.is_relative_to(repo) or repo.is_relative_to(p) for p in [source, sdk, package, output]):
        parser.error('candidate outputs must be outside the checkout')
    raw = args.protocol.read_bytes()
    p = json.loads(raw)
    lock_path, lock = load_profile(POOLED_PROFILE)

    def frozen():
        if args.protocol.read_bytes() != raw or p['runtime_provenance'] != profile_provenance(lock_path, lock):
            raise ValueError('pooled construction protocol/profile differs')
        for key, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[key].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen pooled construction input differs: ' + name)

    frozen()
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(raw)
    q = {'schema_version': 1, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'protocol': descriptor(args.protocol),
        'construction_and_negative_c_api_controls_passed': False, 'commands': [],
        'valid_models_loaded': 0, 'frontend_or_neural_calls': 0,
        'release_qualified': False, 'limits': p['limits']}
    persist = lambda: atomic_json(output / 'qualification.json', q)
    guard = None
    env = {'PATH': '/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(Path.home()),
           'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1', 'GIT_LFS_SKIP_SMUDGE': '1',
           'HOMEBREW_NO_AUTO_UPDATE': '1', 'HOMEBREW_NO_INSTALL_CLEANUP': '1'}
    sandbox = output / 'construction.sb'
    sandbox.write_text(construction_policy(upstream, repo))

    def run(name, argv, *, profile=sandbox, timeout=1200):
        frozen()
        guard.before_model()
        actual = (['/usr/bin/sandbox-exec', '-f', str(profile)] if profile else []) + list(map(str, argv))
        row = {'name': name, 'argv': actual, 'cwd': str(repo)}
        q['commands'].append(row)
        persist()
        with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
            execute_guarded(actual, cwd=repo, environment=env, stdout=stdout, stderr=stderr,
                guard=guard, record=row, persist=persist, timeout=timeout)
        persist()
        if row['returncode'] != 0:
            raise ValueError('pooled construction command failed: ' + name)

    try:
        guard = StorageGuard(output, **p['storage'])
        probes = [output / ('denied.' + suffix)
                  for suffix in ['nemo', 'gguf', 'safetensors', 'onnx', 'npy', 'f32']]
        for probe in probes:
            probe.write_bytes(b'not a model')
            if probe.read_bytes() != b'not a model':
                raise ValueError('positive format control failed')
        run('sandbox-controls', [sys.executable, '-c', SANDBOX_CONTROL, *probes], timeout=30)
        for probe in probes:
            probe.unlink()
        atomic_json(output / 'host.json', {
            'cpu': subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
            'platform': platform.platform(), 'python': sys.version,
            'os': subprocess.check_output(['/usr/bin/sw_vers'], text=True),
            'xcode': subprocess.check_output(['/usr/bin/xcodebuild', '-version'], text=True)})
        run('prepare', [sys.executable, tools / 'prepare_valid_mel_source.py',
            '--runtime-profile', POOLED_PROFILE, '--upstream', upstream,
            '--destination', source, '--output', output / 'preparation'])
        run('build', [sys.executable, tools / 'build_valid_mel_sdk.py', '--runtime-profile', POOLED_PROFILE,
            '--source', source, '--prefix', sdk, '--output', output / 'build', '--jobs', '2',
            '--cpu-arch', PORTABLE_ARM64, '--sandbox-profile', sandbox], profile=None)
        run('package', [sys.executable, tools / 'package_macos.py', '--sdk-prefix', sdk,
            '--build-receipt', output / 'build/build.json', '--output', package])
        manifest = json.loads((package / 'package.json').read_text())
        originals = output / 'source-libraries'
        originals.mkdir()
        for name, item in manifest['libraries'].items():
            original = Path(item['source'])
            if descriptor(original)['sha256'] != item['source_sha256']:
                raise ValueError('package source library changed')
            shutil.copy2(original, originals / name)
        work = output / 'work'
        binary = work / 'bin/failed-load-probe'
        binary.parent.mkdir(parents=True)
        (work / 'nemo').symlink_to(package, target_is_directory=True)
        run('failed-load-probe-build', ['/usr/bin/clang++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
            '-arch', 'arm64', '-march=' + PORTABLE_ARM64, '-mmacosx-version-min=26.0',
            tools / 'native_failed_load_probe.cpp', '-I' + str(sdk / 'include'), '-L' + str(package / 'lib'),
            '-lnemo_speech_asr_c', '-Wl,-rpath,@loader_path/../nemo/lib', '-o', binary])
        protected = {'sdk': inventory(sdk), 'package': inventory(package), 'binary': descriptor(binary)}
        failed_policy = output / 'failed-load.sb'
        failed_policy.write_text(model_free_policy(policy(
            denied_reads=[repo, upstream, source, sdk], readonly=[work, repo, upstream, source, sdk, package])))
        missing = work / 'missing-model'
        if missing.exists() or missing.is_symlink():
            raise ValueError('negative model path must be absent')
        run('failed-loads', ['/usr/bin/env', 'DYLD_PRINT_LIBRARIES=1', binary,
            lock['runtime']['reported_version'], missing], profile=failed_policy, timeout=60)
        result = json.loads((output / 'failed-loads.stdout.log').read_text())
        counts = result['threads_after_each_failure']
        if (result['failed_model_construction_attempts'] != 101 or result['valid_models_loaded'] != 0
                or result['frontend_or_neural_calls'] != 0 or result['thread_growth_allowance'] != 2
                or result['maximum_requested_settling_wait_milliseconds'] != 100
                or len(result['immediate_thread_counts']) != 101
                or len(result['requested_settling_wait_milliseconds']) != 101
                or any(w < 10 or w > 100 or w % 10 for w in result['requested_settling_wait_milliseconds'])
                or len(counts) != 101 or min(counts) < 1 or max(counts) > counts[0] + 2):
            raise ValueError('failed-load cleanup control differs')
        if (inventory(sdk) != protected['sdk'] or inventory(package) != protected['package']
                or descriptor(binary) != protected['binary'] or missing.exists() or missing.is_symlink()):
            raise ValueError('negative control changed protected inputs')
        q['failed_loads'] = result
        q['loaded_images'] = audit_loaded_libraries((output / 'failed-loads.stderr.log').read_text(),
                                                   package, manifest, binary)
        q['binary'] = protected['binary']
        q['sdk_archive'] = write_archive(sdk, output, 'pooled-cpu-sdk')
        q['package_archive'] = write_archive(package, output, 'pooled-cpu-runtime')
        q['controls_archive'] = write_archive(binary.parent, output, 'construction-controls')
        frozen()
        q['construction_and_negative_c_api_controls_passed'] = True
    except BaseException as error:
        q['error'] = repr(error)
        raise
    finally:
        if guard:
            guard.release()
            q['storage'] = guard.record
        persist()
    print(json.dumps({'constructed': True, 'profile': POOLED_PROFILE, 'valid_models_loaded': 0,
                      'failed_construction_attempts': 101, 'frontend_or_neural_calls': 0}))


if __name__ == '__main__':
    main()
