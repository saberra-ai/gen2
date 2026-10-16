"""Filesystem-only recheck; requires the opaque CI tar and repository tools."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--tools', type=Path, required=True)
parser.add_argument('--destination', type=Path, required=True)
a = parser.parse_args()
sys.path.insert(0, str(a.tools))
from sdk_archive import inventory, restore_archive
from package_linux import check_source_bundle

here = Path(__file__).resolve().parent
evidence = here / 'nemo-package-evidence'
manifest = json.loads((evidence / 'development-runtime-files.json').read_text())
assert manifest['archive_sha256'] == '05cf4bc046ea0de8472f2aca72533ecd95106e6bf093cd657ae94a9aab80864e'
assert manifest['archive_bytes'] == 110126341
root = restore_archive(a.archive, manifest, a.destination)
assert inventory(root) == manifest['files']
assert hashlib.sha256((root / 'package.json').read_bytes()).hexdigest() == '7bfdc02af4a3ecd38abbf94d3187d0a2cbcb1c79b6e0ef2fd943b645011eb699'
package = json.loads((root / 'package.json').read_text())
for name, record in package['files'].items():
    assert manifest['files'][name] == record
for name, component in package['components'].items():
    source = component['source_bundle']
    assert check_source_bundle(root / 'sources' / source['source_package'], source['source_package'], source['source_version']) == source
    assert hashlib.sha256((root / 'notices' / name / 'copyright').read_bytes()).hexdigest() == component['copyright_sha256']
print('PASS: delivered archive, 45 filesystem entries, package manifest, exact source bundles and copyrights; no Linux execution')
