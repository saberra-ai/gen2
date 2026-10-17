"""Qualify the current Linux consumer against the unchanged historical native package."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from build_valid_mel_candidate import descriptor
from platform_sdk_receipt import validate_native_receipt
from qualification_storage import atomic_json
from restore_valid_mel_linux_sdk import restore
from runtime_profiles import validate_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-evidence', type=Path, required=True)
    parser.add_argument('--package-evidence', type=Path, required=True,
                        help='Construction wrapper output, containing its construction/ subdirectory.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('qualification requires x86_64 Linux')
    if args.output.exists() or args.output.is_symlink():
        parser.error('a fresh output directory is required')
    tools = Path(__file__).resolve().parent; repo = tools.parents[1]
    output = args.output.absolute(); native = args.native_evidence.resolve(strict=True)
    package = args.package_evidence.resolve(strict=True)
    if any(output.resolve().is_relative_to(root) or root.is_relative_to(output.resolve()) for root in [native, package]):
        parser.error('qualification output and producer evidence must be disjoint')
    protocol_path = tools / 'current-valid-mel-linux-runtime.protocol.json'
    protocol_bytes = protocol_path.read_bytes(); p = json.loads(protocol_bytes)
    inputs = {'native_receipt': native / 'receipt.json',
              'construction_qualification': package / 'qualification.json',
              'construction_receipt': package / 'construction/receipt.json',
              'package_manifest': package / 'construction/package.json',
              'runtime_archive': package / 'construction/development-runtime.tar.gz'}
    def frozen():
        if protocol_path.read_bytes() != protocol_bytes:
            raise ValueError('Linux runtime protocol changed')
        for key, root in [('tools', tools), ('repository_files', repo),
                          ('vendor_sources', repo / 'vendor/llama-cpp-sys-2/llama.cpp')]:
            for name, expected in p[key].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen Linux runtime source differs: ' + name)
        for name, path in inputs.items():
            if descriptor(path) != p['inputs'][name]:
                raise ValueError('frozen Linux runtime producer input differs: ' + name)
        for name, expected in p['gitlinks'].items():
            tree = repo / name
            actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip()
            if (actual != expected or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=tree)
                    or subprocess.check_output(['git', 'ls-files', '--others', '--ignored', '--exclude-standard', '-z'], cwd=tree)):
                raise ValueError('chat source gitlink differs: ' + name)
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(protocol_bytes)
    result = {'schema_version': 1, 'release_qualified': False, 'offline_container_gates_passed': False,
              'chat_diarization_gates_passed': False, 'commands': [], 'protocol': descriptor(protocol_path)}
    try:
        frozen()
        build = json.loads(inputs['native_receipt'].read_text()); validate_native_receipt(build)
        if build['source_commit'] != p['historical_sdk_source_commit']:
            raise ValueError('fetched historical Linux SDK producer differs')
        construction = json.loads(inputs['construction_qualification'].read_text())
        child = json.loads(inputs['construction_receipt'].read_text()); validate_provenance(child)
        if (construction['package_construction_passed'] is not True
                or child['package_construction_passed'] is not True
                or construction['source_commit'] != p['construction_source_commit']
                or child['source_commit'] != p['construction_source_commit']
                or descriptor(inputs['construction_receipt']) != construction['construction_receipt']
                or descriptor(inputs['native_receipt']) != construction['native_receipt']):
            raise ValueError('completed bound Linux construction required')
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        result['source_commit'] = source
        atomic_json(output / 'frozen-inputs.json', {'source_commit': source, 'inputs': p['inputs'],
                    'tools': p['tools'], 'repository_files': p['repository_files'], 'gitlinks': p['gitlinks']})
        sdk = restore(native, output / 'restored-sdk')
        stages = [('offline', 'qualify_linux_package.py',
                   ['--sdk-prefix', sdk, '--native-evidence', native, '--package-evidence', package / 'construction']),
                  ('chat', 'qualify_linux_chat.py', ['--container-evidence', output / 'offline'])]
        for name, script, arguments in stages:
            frozen()
            command = list(map(str, [sys.executable, tools / script, *arguments, '--output', output / name]))
            record = {'name': name, 'argv': command, 'cwd': str(repo)}; result['commands'].append(record)
            atomic_json(output / 'commands.json', result['commands'])
            started = time.monotonic()
            try:
                with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
                    completed = subprocess.run(command, cwd=repo, stdout=stdout, stderr=stderr)
                record['returncode'] = completed.returncode
            except OSError as error:
                record['startup_error'] = str(error)
                raise
            finally:
                record['elapsed_seconds'] = time.monotonic() - started
                atomic_json(output / 'commands.json', result['commands'])
            completed.check_returncode()
            receipt_path = output / name / 'receipt.json'
            receipt = json.loads(receipt_path.read_text()); validate_provenance(receipt)
            field = 'offline_container_gates_passed' if name == 'offline' else 'chat_diarization_gates_passed'
            if receipt.get(field) is not True or receipt['source_commit'] != source:
                raise ValueError('actual Linux runtime stage did not complete: ' + name)
            result[field] = True
            result[name + '_receipt'] = descriptor(receipt_path)
        frozen()
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        atomic_json(output / 'qualification.json', result)


if __name__ == '__main__':
    main()
