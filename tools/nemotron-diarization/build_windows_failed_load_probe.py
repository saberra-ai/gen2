"""Compile bounded failure/observer helpers; run only the model-free Windows observer."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from build_valid_mel_candidate import descriptor
from package_bounded_windows import qualified_sdk
from bounded_platform_sdk_receipt import load_profile
from platform_native import msvc_environment, msvc_probe_command
from windows_naming import NAMES
from windows_pe import inspect_pe


def validate_observer(value):
    if (value.get('known_worker_observed') is not True or value.get('known_worker_retired') is not True
            or any(type(value.get(name)) is not int or value[name] != 0
                   for name in ['valid_models_loaded', 'frontend_or_neural_calls'])
            or any(type(value.get(name)) is not int or value[name] < 1
                   for name in ['baseline_threads', 'active_threads', 'settled_threads'])
            or type(value.get('requested_retirement_wait_ms')) is not int
            or value['requested_retirement_wait_ms'] not in range(0, 101, 10)):
        raise ValueError('actual known-worker Windows observer control required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['sdk-prefix', 'native-evidence', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Windows' or platform.machine().lower() not in {'amd64', 'x86_64'}:
        parser.error('requires actual Windows AMD64/MSVC')
    if args.output.exists() or args.output.is_symlink():
        parser.error('fresh helper output required')
    tools = Path(__file__).resolve().parent
    sdk = args.sdk_prefix.resolve(strict=True)
    native = args.native_evidence.resolve(strict=True)
    output = args.output.resolve()
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in [sdk, native]):
        parser.error('helper output must be disjoint from immutable SDK/evidence')
    output.mkdir(parents=True)
    q = {'schema_version': 1, 'observer_control_passed': False, 'failed_loads_executed': False,
         'valid_models_loaded': 0, 'frontend_or_neural_calls': 0, 'release_qualified': False, 'commands': []}
    save = lambda: (output / 'qualification.json').write_text(json.dumps(q, indent=2) + '\n')

    def run(name, argv, env):
        row = {'name': name, 'argv': list(map(str, argv)), 'cwd': str(output)}
        q['commands'].append(row)
        save()
        start = time.monotonic()
        try:
            with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
                result = subprocess.run(row['argv'], cwd=output, env=env, stdout=stdout, stderr=stderr, timeout=120)
            row['returncode'] = result.returncode
            result.check_returncode()
        except (OSError, subprocess.TimeoutExpired) as error:
            row['execution_error'] = repr(error)
            raise
        finally:
            row['elapsed_seconds'] = time.monotonic() - start
            save()

    try:
        build, _, _ = qualified_sdk(sdk, native, load_profile()[1])
        q['native_source_commit'] = build['source_commit']
        q['native_receipt'] = descriptor(native / 'receipt.json')
        q['sources'] = {name: descriptor(tools / name) for name in ['windows_own_threads.h',
            'windows_thread_observer_control.cpp', 'native_failed_load_probe_windows.cpp', 'build_windows_failed_load_probe.py']}
        environment = msvc_environment(dict(os.environ))
        observer = output / 'thread-observer-control.exe'
        failure = output / 'failed-load-probe.exe'
        run('observer-build', msvc_probe_command(environment, [tools / 'windows_thread_observer_control.cpp'],
            [tools], 'kernel32.lib', observer), environment)
        info = inspect_pe(observer)
        if (info['machine'] != 'AMD64' or info['is_dll'] or info['bound_imports_present']
                or any(name in set(NAMES) | set(NAMES.values()) for name in
                       info['imports'] + info['delay_imports'] + info['forwarder_dependencies'])):
            raise ValueError('observer must be a separate AMD64 executable without NeMo imports')
        run('observer-control', [observer], environment)
        value = json.loads((output / 'observer-control.stdout.log').read_text(encoding='utf-8-sig'))
        validate_observer(value)
        q.update(observer_control_passed=True, observer=value, observer_binary=descriptor(observer), observer_pe=info)
        run('failed-load-build', msvc_probe_command(environment, [tools / 'native_failed_load_probe_windows.cpp'],
            [tools, sdk / 'include'], sdk / 'lib/gen2_nemo_windows_nemo_speech_asr_c.lib', failure), environment)
        q.update(failed_load_binary=descriptor(failure), failed_load_pe=inspect_pe(failure))
        # Re-admit the input after compilation; no SDK/evidence mutation is allowed.
        qualified_sdk(sdk, native, load_profile()[1])
    except BaseException as error:
        q['error'] = repr(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
