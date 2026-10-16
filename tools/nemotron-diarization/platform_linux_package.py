"""Construct and archive a Linux runtime package after native SDK qualification.

This stage does not yet establish clean-container execution or coexistence.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from package_linux import digest, verify
from sdk_archive import write_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-prefix', type=Path, required=True)
    parser.add_argument('--native-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('a new output directory is required')
    output = args.output.resolve()
    output.mkdir(parents=True)
    package = output / 'runtime'
    receipt = {'schema_version': 1, 'release_qualified': False,
               'scope': 'Linux development runtime package construction, source/notice provenance, ELF closure and archive identity only.',
               'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'native_receipt_sha256': digest(args.native_receipt),
               'packager_sha256': digest(Path(__file__).with_name('package_linux.py')),
               'archive_tool_sha256': digest(Path(__file__).with_name('sdk_archive.py')),
               'runner_sha256': digest(Path(__file__)), 'package_construction_passed': False,
               'limits': ['No fresh-container, host namespace isolation, chat/Bark/Laya coexistence or quality qualification.']}
    try:
        with (output / 'package-build.log').open('x') as log:
            # Keep APT and native-tool diagnostics even if staging is rolled back.
            command = [sys.executable, str(Path(__file__).with_name('package_linux.py')),
                       '--sdk-prefix', str(args.sdk_prefix.resolve()), '--build-receipt', str(args.native_receipt.resolve()),
                       '--output', str(package)]
            (output / 'commands.json').write_text(json.dumps([command], indent=2) + '\n')
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = verify(package)
        (output / 'package.json').write_text(json.dumps(manifest, indent=2) + '\n')
        receipt['archive'] = write_archive(package, output, 'development-runtime')
        receipt['package_manifest_sha256'] = digest(package / 'package.json')
        receipt['package_construction_passed'] = True
    except Exception as error:
        receipt['error'] = str(error)
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        raise
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
