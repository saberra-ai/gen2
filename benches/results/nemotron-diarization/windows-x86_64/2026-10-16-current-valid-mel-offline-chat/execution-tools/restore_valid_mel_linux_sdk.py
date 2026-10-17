"""Restore a qualified Linux SDK artifact's modes/aliases without executing it."""
import argparse
import json
from pathlib import Path

from platform_sdk_receipt import validate_native_receipt
from sdk_archive import restore_archive


def restore(evidence, destination):
    evidence, destination = Path(evidence), Path(destination)
    build = json.loads((evidence / 'receipt.json').read_text())
    validate_native_receipt(build)
    if not build['platform'].startswith('Linux-') or build['machine'].lower() != 'x86_64':
        raise ValueError('a qualified x86_64 Linux SDK snapshot is required')
    manifest = json.loads((evidence / 'development-sdk-files.json').read_text())
    summary = {key: value for key, value in manifest.items() if key != 'files'}
    if summary != build['development_sdk_archive'] or manifest['archive'] != 'development-sdk.tar.gz':
        raise ValueError('Linux SDK archive identity differs from qualified receipt')
    return restore_archive(evidence / manifest['archive'], manifest, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(restore(args.native_evidence.resolve(strict=True), args.output.absolute()))


if __name__ == '__main__':
    main()
