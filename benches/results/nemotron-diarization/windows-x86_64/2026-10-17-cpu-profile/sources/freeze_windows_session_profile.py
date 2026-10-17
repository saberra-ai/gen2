"""Freeze source-only controls for profiling an already retained diagnostic SDK."""
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from profile_windows_sessions import RECEIPT
from windows_session_timing import ORDER


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent = json.loads((tools / 'windows-worker-diagnostic-v2.protocol.json').read_text())
    collection = json.loads((repo / RECEIPT / 'collection.json').read_text())
    for name, expected in collection['files'].items():
        if descriptor(repo / RECEIPT / name) != expected:
            raise ValueError('reviewed source collection changed: ' + name)
    tool_names = set(parent['tools']) | {'windows-worker-diagnostic-v2.protocol.json', 'windows_archive.py',
        'windows_session_timing.py', 'profile_windows_sessions.py', 'test_windows_session_profile.py',
        'freeze_windows_session_profile.py'}
    repo_names = set(parent['repository_files']) | {RECEIPT + '/' + name for name in collection['files']}
    repo_names |= {RECEIPT + '/collection.json', '.github/workflows/diarization-windows-session-profile.yml',
                   'docs/research/nemotron-windows-session-profile.md'}
    p = {'schema_version': 1, 'order': ORDER, 'command_timeout_seconds': 600,
         'source_delivery': {'run_id': 39025306014, 'artifact_id': 12098214420,
             'head_sha': '1745243b221368855381ea36879ccdc1bb731528',
             'archive': {'bytes': 45429953, 'sha256': '81433805f066deb99d449ceeab93a2ada0d1c50052b3d698eb117cc5f411c786'}},
         'tools': {name: descriptor(tools / name) for name in sorted(tool_names)},
         'repository_files': {name: descriptor(repo / name) for name in sorted(repo_names)},
         'limits': ['Native timing diagnostic only; no supported release, public API, lifecycle or package qualification.',
                    'Exact existing SDK; no source/compiler/geometry/model/worker-budget change.',
                    'Numerical reference is from the producer host; compare timing only within this new receiver sequence.',
                    'Upstream session timers exclude cache construction/locking; frontend is timed separately; residual is unattributed.',
                    'No timing averaging or relaxation of original below-one RTF and quality requirements.']}
    target = tools / 'windows-session-profile.protocol.json'
    target.write_text(json.dumps(p, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'tools': len(p['tools']),
                      'repository_files': len(p['repository_files'])}))


if __name__ == '__main__':
    main()
