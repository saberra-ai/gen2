"""Frozen Windows v2 construction, central VC setup and offline/chat qualification."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from build_valid_mel_candidate import descriptor
from package_windows import qualified_sdk
from qualification_storage import atomic_json
from runtime_profiles import load_profile, validate_provenance
from windows_archive import byte_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-input', type=Path, required=True)
    parser.add_argument('--producer-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Windows' or platform.machine().lower() not in {'amd64', 'x86_64'}:
        parser.error('qualification requires actual Windows AMD64')
    output = args.output.absolute(); native = args.native_input.resolve(strict=True)
    if output.exists() or output.is_symlink():
        parser.error('a fresh qualification output is required')
    if output.resolve().is_relative_to(native) or native.is_relative_to(output.resolve()):
        parser.error('input and output must be disjoint')
    tools = Path(__file__).resolve().parent; repo = tools.parents[1]
    protocol_path = tools / 'valid-mel-windows-runtime.protocol.json'
    protocol_bytes = protocol_path.read_bytes(); p = json.loads(protocol_bytes)
    evidence = native / 'nemo-valid-mel-evidence'; sdk = native / 'nemo-valid-mel-sdk'
    def frozen():
        if protocol_path.read_bytes() != protocol_bytes:
            raise ValueError('Windows runtime protocol changed')
        for key, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[key].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen Windows source differs: ' + name)
        for name, expected in p['gitlinks'].items():
            if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo / name, text=True).strip() != expected
                    or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo / name, text=True)):
                raise ValueError('chat native source differs')
        if descriptor(evidence / 'receipt.json') != p['native_receipt']:
            raise ValueError('Windows native receipt differs')
        if byte_inventory(sdk) != p['sdk_inventory']:
            raise ValueError('Windows native SDK inventory differs')
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(protocol_bytes)
    commands = []
    receipt = {'schema_version': 1, 'release_qualified': False, 'redistribution_qualified': False,
               'package_construction_passed': False, 'setup_and_fresh_image_state_passed': False,
               'offline_container_gates_passed': False, 'chat_diarization_gates_passed': False,
               'protocol': descriptor(protocol_path), 'commands': commands,
               'scope': 'Tagged five-private-DLL v2 package, disposable publisher-VC image and actual offline native/public/lifecycle/CPU-chat gates on Windows Server 2022 AMD64. No physical desktop, Bark/Laya, heldout quality, uncontended performance or release qualification.'}
    def run(name, script, arguments):
        frozen()
        record = {'name': name, 'argv': [sys.executable, str(tools / script), *map(str, arguments)], 'cwd': str(repo)}
        commands.append(record); atomic_json(output / 'commands.json', commands)
        started = time.monotonic()
        try:
            with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
                completed = subprocess.run(record['argv'], cwd=repo, stdout=stdout, stderr=stderr)
            record['returncode'] = completed.returncode
        except OSError as error:
            record['startup_error'] = str(error)
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            atomic_json(output / 'commands.json', commands)
        completed.check_returncode()
        frozen()
    try:
        frozen()
        producer = json.loads(args.producer_metadata.read_text(encoding='utf-8-sig'))
        expected = p['producer']
        if (producer['id'] != expected['run_id'] or producer['head_sha'] != expected['source_commit']
                or producer['conclusion'] != 'success' or producer['repository']['full_name'] != expected['repository']):
            raise ValueError('successful exact SDK producer required')
        build, _, _ = qualified_sdk(sdk, evidence, load_profile()[1])
        if build['source_commit'] != expected['source_commit']:
            raise ValueError('native SDK source differs from producer')
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        receipt['source_commit'] = source
        (output / 'producer-run.json').write_bytes(args.producer_metadata.read_bytes())
        (output / 'native-receipt.json').write_bytes((evidence / 'receipt.json').read_bytes())
        atomic_json(output / 'frozen-inputs.json', {'source_commit': source, 'native_receipt': p['native_receipt'],
                    'tools': p['tools'], 'repository_files': p['repository_files'], 'gitlinks': p['gitlinks'],
                    'sdk_inventory': p['sdk_inventory'], 'producer': expected})
        package = output / 'construction'; prerequisites = output / 'prerequisites'; setup = output / 'central-setup'; runtime = output / 'runtime'
        run('construction', 'package_windows.py', ['--central-runtime', '--sdk-prefix', sdk, '--native-evidence', evidence, '--output', package])
        child = json.loads((package / 'receipt.json').read_text(encoding='utf-8-sig')); validate_provenance(child)
        if not child['package_construction_passed'] or not child['central_runtime_variant']:
            raise ValueError('five-DLL v2 construction required')
        receipt.update(package_construction_passed=True, construction_receipt=descriptor(package / 'receipt.json'))
        run('preflight', 'windows_container_preflight.py', ['--output', prerequisites])
        run('central-setup', 'windows_central_setup.py', ['--prerequisites', prerequisites / 'preflight.json', '--output', setup])
        child = json.loads((setup / 'receipt.json').read_text(encoding='utf-8-sig'))
        if not child['setup_and_fresh_image_state_passed'] or child['source_commit'] != source:
            raise ValueError('same-source actual central runtime setup required')
        receipt.update(setup_and_fresh_image_state_passed=True, setup_receipt=descriptor(setup / 'receipt.json'))
        run('offline-chat', 'qualify_windows_chat.py', ['--sdk-prefix', sdk, '--native-evidence', evidence,
            '--package-evidence', package, '--central-setup', setup, '--prerequisites', prerequisites / 'preflight.json', '--output', runtime])
        child = json.loads((runtime / 'receipt.json').read_text(encoding='utf-8-sig')); validate_provenance(child)
        if not child['offline_container_gates_passed'] or not child['chat_diarization_gates_passed'] or child['source_commit'] != source:
            raise ValueError('same-source actual offline and CPU chat qualification required')
        receipt.update(offline_container_gates_passed=True, chat_diarization_gates_passed=True,
                       runtime_receipt=descriptor(runtime / 'receipt.json'))
        frozen()
    except BaseException as error:
        receipt['error'] = str(error)
        raise
    finally:
        atomic_json(output / 'qualification.json', receipt)


if __name__ == '__main__':
    main()
