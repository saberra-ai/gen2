"""Native diagnostic using the existing session timers and exact retained SDK."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from artifacts import acquire, matches
from build_valid_mel_candidate import descriptor
from frozen_git_sources import verify_git_sources
from pooled_runtime import BOUNDED_PROFILE, load_profile
from qualification_storage import atomic_json
from windows_naming import verify_private_outputs
from windows_worker_metrics import compare_native, telemetry, version
from windows_session_timing import ORDER, extract_delivery, file_inventory, profiling_environment, summarize


RECEIPT = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix'
HOST_SCRIPT = r'''$ErrorActionPreference='Stop'
$os=Get-CimInstance Win32_OperatingSystem
$cpu=@(Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,Architecture,NumberOfCores,NumberOfLogicalProcessors,ProcessorId)
@{os=@{caption=$os.Caption;version=$os.Version;build=$os.BuildNumber};cpu=$cpu;
avx2=[System.Runtime.Intrinsics.X86.Avx2]::IsSupported;fma=[System.Runtime.Intrinsics.X86.Fma]::IsSupported} | ConvertTo-Json -Depth 8
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64', 'x86_64'):
        parser.error('actual Windows x86_64 diagnostic receiver required')
    tools = Path(__file__).resolve().parent
    repo, output = tools.parents[1], args.output.resolve()
    if output.exists() or output.is_relative_to(repo) or output.is_symlink():
        parser.error('new output outside repository required')
    protocol_path = tools / 'windows-session-profile.protocol.json'
    protocol_bytes = protocol_path.read_bytes()
    p = json.loads(protocol_bytes)
    git_env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, env=git_env, text=True).strip()

    def frozen():
        if protocol_path.read_bytes() != protocol_bytes or p['order'] != ORDER:
            raise ValueError('frozen session profile protocol/order changed')
        for group, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen profiling input changed: ' + name)

    frozen()
    expected = {**{'tools/nemotron-diarization/' + name: value for name, value in p['tools'].items()}, **p['repository_files']}
    verify_git_sources(repo, commit, expected)
    if subprocess.check_output(['git', 'show', commit + ':tools/nemotron-diarization/' + protocol_path.name],
                               cwd=repo, env=git_env) != protocol_bytes:
        raise ValueError('committed profiling protocol required')
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(protocol_bytes)
    q = {'schema_version': 1, 'source_commit': commit, 'protocol': descriptor(protocol_path),
         'profile_completed': False, 'release_qualified': False, 'package_qualified': False,
         'public_or_lifecycle_qualified': False, 'order': ORDER, 'cases': [], 'commands': [],
         'source_delivery': p['source_delivery'], 'limits': p['limits']}
    persist = lambda: atomic_json(output / 'qualification.json', q)
    persist()

    def run(name, argv, *, environment=None, stdout=None):
        frozen()
        row = {'name': name, 'argv': list(map(str, argv)), 'cwd': str(output)}
        q['commands'].append(row)
        persist()
        print('START ' + name, flush=True)
        started = time.monotonic()
        try:
            with (stdout or output / (name + '.stdout.log')).open('xb') as out, (output / (name + '.stderr.log')).open('xb') as err:
                result = subprocess.run(row['argv'], cwd=output, env=environment or os.environ.copy(),
                    stdout=out, stderr=err, timeout=p['command_timeout_seconds'])
            row['returncode'] = result.returncode
            result.check_returncode()
        except BaseException as error:
            row['error'] = repr(error)
            raise
        finally:
            row['elapsed_seconds'] = time.monotonic() - started
            persist()
        print('PASS ' + name, flush=True)

    try:
        source = p['source_delivery']
        base = 'repos/saberra-ai/gen2/actions/'
        run('github-cli-version', ['gh', '--version'])
        run('source-run', ['gh', 'api', base + 'runs/' + str(source['run_id'])])
        run('source-artifact', ['gh', 'api', base + 'artifacts/' + str(source['artifact_id'])])
        source_run = json.loads((output / 'source-run.stdout.log').read_text(encoding='utf-8-sig'))
        artifact = json.loads((output / 'source-artifact.stdout.log').read_text(encoding='utf-8-sig'))
        if (source_run['id'] != source['run_id'] or source_run['head_sha'] != source['head_sha']
                or source_run['conclusion'] != 'success' or source_run['repository']['full_name'] != 'saberra-ai/gen2'
                or artifact['id'] != source['artifact_id'] or artifact['expired'] is not False
                or artifact['workflow_run']['id'] != source['run_id']
                or artifact['workflow_run']['head_sha'] != source['head_sha']
                or artifact['size_in_bytes'] != source['archive']['bytes']
                or artifact['digest'] != 'sha256:' + source['archive']['sha256']):
            raise ValueError('exact completed diagnostic artifact identity required')
        archive = output / 'source-artifact.zip'
        run('download-sdk-evidence', ['gh', 'api', base + 'artifacts/' + str(source['artifact_id']) + '/zip'], stdout=archive)
        members = json.loads((repo / RECEIPT / 'artifact-members.json').read_text())
        delivered = output / 'delivered'
        extract_delivery(archive, source['archive'], members, delivered)
        producer = json.loads((repo / RECEIPT / 'qualification.json').read_text())
        if (producer['source_commit'] != source['head_sha'] or producer['diagnostic_completed'] is not True
                or producer['release_qualified'] is not False or producer['package_qualified'] is not False
                or producer['builds']['4']['reported_version'] != version(4)):
            raise ValueError('reviewed completed four-worker diagnostic source required')
        sdk_expected = {name.replace('\\', '/'): value for name, value in producer['builds']['4']['installed_files'].items()}
        if file_inventory(delivered / 'sdk-4') != sdk_expected:
            raise ValueError('delivered diagnostic SDK differs')
        sdk = output / 'sdk'
        shutil.copytree(delivered / 'sdk-4', sdk)
        if file_inventory(sdk) != sdk_expected:
            raise ValueError('copied diagnostic SDK differs')
        q['sdk_files'] = sdk_expected
        q['private_pe'] = verify_private_outputs(sdk, output)
        native = sdk / 'bin/gen2_native_stream_probe_valid_mel.exe'
        q['native_probe'] = descriptor(native)
        run('receiver-host', ['pwsh', '-NoProfile', '-NonInteractive', '-Command', HOST_SCRIPT])
        host = json.loads((output / 'receiver-host.stdout.log').read_text(encoding='utf-8-sig'))
        q['host'] = host
        if host['avx2'] is not True or host['fma'] is not True:
            raise ValueError('receiver AVX2/FMA prerequisite unavailable')
        _, lock = load_profile(BOUNDED_PROFILE)
        assets = output / 'assets'
        assets.mkdir()
        for item in lock['artifacts']:
            if item['role'] != 'reference':
                frozen()
                acquire(item, assets)
                if not matches(assets / item['name'], item):
                    raise ValueError('pinned profiling asset differs: ' + item['name'])
        assets_expected = file_inventory(assets)
        q['assets'] = assets_expected
        reference_path = delivered / 'evidence-4/case-0-cap-4.json'
        reference, reference_raw = json.loads(reference_path.read_text()), Path(str(reference_path) + '.f32').read_bytes()
        for index, mode in enumerate(ORDER):
            frozen()
            if (file_inventory(sdk) != sdk_expected or file_inventory(delivered) != members
                    or file_inventory(assets) != assets_expected or descriptor(archive) != source['archive']):
                raise ValueError('profiling SDK/delivery/assets changed')
            phase = 'case-' + str(index) + '-' + mode
            target = output / (phase + '.json')
            case = {'index': index, 'mode': mode, 'command_started': True,
                    'timing_environment': '1' if mode == 'timing' else None}
            q['cases'].append(case)
            persist()
            run(phase, [native, assets / 'Nemotron-3-Diarization.q8_0.gguf', assets / 'ami_en2002d_2132.wav',
                        '1', '-1', target], environment=profiling_environment(os.environ, mode))
            result = json.loads(target.read_text())
            if type(result.get('push_count')) is not int or result['push_count'] != 58:
                raise ValueError('fixed58-push native input partition required')
            raw = Path(str(target) + '.f32').read_bytes()
            log = (output / (phase + '.stderr.log')).read_text()
            workers = telemetry(log, 4)
            if workers['effective_threads'] != 4:
                raise ValueError('fixed four-worker receiver required')
            case.update(result=result, workers=workers, result_file=descriptor(target),
                        probabilities=descriptor(Path(str(target) + '.f32')),
                        numerical_comparison=compare_native(reference, result, reference_raw, raw, 4, 4),
                        session_profile=summarize(log, mode, result['elapsed_micros']))
            persist()
        frozen()
        if (file_inventory(sdk) != sdk_expected or file_inventory(delivered) != members
                or file_inventory(assets) != assets_expected or descriptor(archive) != source['archive']):
            raise ValueError('profiling inputs changed after execution')
        q['profile_completed'] = True
    except BaseException as error:
        q['error'] = repr(error)
        raise
    finally:
        persist()


if __name__ == '__main__':
    main()
