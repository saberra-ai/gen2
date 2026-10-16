"""Construct a current Linux package from one frozen successful SDK artifact; no NN."""
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
    parser.add_argument('--producer-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('construction requires x86_64 Linux')
    if args.output.exists() or args.output.is_symlink():
        parser.error('a fresh output directory is required')
    tools = Path(__file__).resolve().parent; repo = tools.parents[1]
    output = args.output.absolute(); evidence = args.native_evidence.resolve(strict=True)
    if output.resolve().is_relative_to(evidence) or evidence.is_relative_to(output.resolve()):
        parser.error('source evidence and construction output must be disjoint')
    protocol_path = tools / 'valid-mel-linux-construction.protocol.json'
    protocol_bytes = protocol_path.read_bytes(); protocol = json.loads(protocol_bytes)
    def frozen():
        if protocol_path.read_bytes() != protocol_bytes:
            raise ValueError('Linux construction protocol changed')
        for key, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in protocol[key].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen Linux construction source differs: ' + name)
        if descriptor(evidence / 'receipt.json') != protocol['native_receipt']:
            raise ValueError('Linux SDK receipt differs from frozen producer')
    output.mkdir(parents=True)
    (output / 'protocol.json').write_bytes(protocol_bytes)
    receipt = {'schema_version': 1, 'package_construction_passed': False, 'release_qualified': False,
               'scope': 'Tagged Linux package construction and exact archive identity only; no model execution.',
               'new_native_model_or_neural_calls': 0, 'protocol': descriptor(protocol_path), 'command': None}
    try:
        frozen()
        build = json.loads((evidence / 'receipt.json').read_text()); validate_native_receipt(build)
        producer = json.loads(args.producer_metadata.read_text())
        expected = protocol['producer']
        if (producer['id'] != expected['run_id'] or producer['head_sha'] != expected['source_commit']
                or producer['repository']['full_name'] != expected['repository']
                or build['source_commit'] != expected['source_commit']
                or build['development_sdk_archive'] != protocol['sdk_archive']):
            raise ValueError('Linux producer identity differs')
        (output / 'producer-run.json').write_bytes(args.producer_metadata.read_bytes())
        (output / 'native-receipt.json').write_bytes((evidence / 'receipt.json').read_bytes())
        source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        atomic_json(output / 'frozen-inputs.json', {'source_commit': source,
                    'native_receipt': descriptor(evidence / 'receipt.json'), 'producer': expected,
                    'protocol': descriptor(protocol_path), 'tools': protocol['tools'],
                    'repository_files': protocol['repository_files']})
        sdk = restore(evidence, output / 'restored-sdk')
        command = [sys.executable, str(tools / 'platform_linux_package.py'), '--sdk-prefix', str(sdk),
                   '--native-receipt', str(evidence / 'receipt.json'), '--output', str(output / 'construction')]
        record = {'argv': command, 'cwd': str(repo)}; receipt['command'] = record
        atomic_json(output / 'command.json', record)
        started = time.monotonic()
        try:
            with (output / 'construct.stdout.log').open('xb') as stdout, (output / 'construct.stderr.log').open('xb') as stderr:
                completed = subprocess.run(command, cwd=repo, stdout=stdout, stderr=stderr)
            record['returncode'] = completed.returncode
        except OSError as error:
            record['startup_error'] = str(error)
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            atomic_json(output / 'command.json', record)
        completed.check_returncode()
        child = json.loads((output / 'construction/receipt.json').read_text()); validate_provenance(child)
        if child['package_construction_passed'] is not True or child['source_commit'] != source:
            raise ValueError('same-source package construction did not pass')
        frozen()
        receipt.update(package_construction_passed=True, source_commit=source,
                       construction_receipt=descriptor(output / 'construction/receipt.json'),
                       native_receipt=protocol['native_receipt'], archive=child['archive'])
    except Exception as error:
        receipt['error'] = str(error)
        raise
    finally:
        atomic_json(output / 'qualification.json', receipt)


if __name__ == '__main__':
    main()
