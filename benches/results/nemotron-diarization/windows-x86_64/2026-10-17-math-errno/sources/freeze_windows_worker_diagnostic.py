"""Preregister worker experiment bytes from an already admitted bounded tree."""
import argparse
import hashlib
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from pooled_runtime import BOUNDED_PROFILE, load_profile
from windows_worker_metrics import ORDER, PCM_SHA256, transform, version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bounded-source', type=Path, required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    _, lock = load_profile(BOUNDED_PROFILE)
    parent = json.loads((tools / 'bounded-windows-native.protocol.json').read_text())
    inputs = {}
    for name, expected in lock['runtime']['source_sha256'].items():
        raw = (args.bounded_source / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('admitted bounded source differs: ' + name)
        inputs[name] = raw
    tool_names = set(parent['tools']) | {
        'bounded-windows-native.protocol.json', 'windows_worker_diagnostic.py', 'windows_worker_metrics.py',
        'windows_worker_sources.py', 'test_windows_worker_diagnostic.py', 'freeze_windows_worker_diagnostic.py'}
    repo_names = set(parent['repository_files']) | {
        '.github/workflows/diarization-windows-worker-diagnostic.yml',
        'docs/research/nemotron-windows-worker-diagnostic.md'}
    p = {'schema_version': 1, 'order': ORDER, 'actual_pcm_sha256': PCM_SHA256,
         'tools': {name: descriptor(tools / name) for name in sorted(tool_names)},
         'repository_files': {name: descriptor(repo / name) for name in sorted(repo_names)},
         'variants': {str(cap): {'reported_version': version(cap), 'source_sha256': {
             name: hashlib.sha256(transform(name, raw, cap)).hexdigest() for name, raw in inputs.items()}}
             for cap in (1, 2, 4)},
         'limits': ['Diagnostic-only native C ABI worker-budget comparison; no supported release or package.',
                    'No public API/lifecycle/failed-load/offline/coexistence/quality/signing admission.',
                    'Fixed4,2,1,4 fresh-process sequence; preserve every observation without timing promotion.',
                    'Telemetry is configured persistent-pool size, not independent OS thread enumeration.']}
    target = tools / 'windows-worker-diagnostic.protocol.json'
    target.write_text(json.dumps(p, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'tools': len(p['tools']),
                      'repository_files': len(p['repository_files'])}))


if __name__ == '__main__':
    main()
