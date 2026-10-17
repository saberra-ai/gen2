"""Replay three completed app CLI traces without model, app or native calls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from live_frontier_metrics import frontier_observations


def descriptor(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    raw = args.protocol.read_bytes()
    protocol = json.loads(raw)
    out = args.output.resolve()
    if out.exists() or args.output.is_symlink() or out.is_relative_to(repo):
        parser.error('fresh output outside the repository required')

    def frozen():
        if args.protocol.read_bytes() != raw:
            raise ValueError('frontier protocol changed')
        for group, root in [('tools', tools), ('inputs', repo)]:
            for name, expected in protocol[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frontier analysis input changed: ' + name)

    frozen()
    result = {'schema_version': 1, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'analysis_completed': False, 'release_qualified': False, 'new_model_or_native_calls': 0,
        'cases': {}, 'limits': protocol['limits']}
    if [case['name'] for case in protocol['cases']] != ['valid-mel-cpu', 'valid-mel-metal', 'pooled-cpu']:
        raise ValueError('all three fixed completed app cases required')
    for case in protocol['cases']:
        live = json.loads((repo / case['result']).read_text())
        receiver = json.loads((repo / case['receipt']).read_text())
        construction = json.loads((repo / case['construction']).read_text())
        if (receiver['release_qualified'] is not False
                or receiver[case['success_field']] is not True
                or construction['pnola_source_files']['src-tauri/src/live.rs'] != protocol['observer_source']
                or live['passed'] is not True or live['microphone_used'] is not False
                or live['speaker_qualification'] is not True or live['sections'] != 11 or live['queued'] != 0
                or live['duration_seconds'] != 80.04 or live['speakers']['status'] != 'complete'
                or live['speakers']['timeline']['runtime_version'] != case['runtime_version']
                or receiver['source_commit'] != case['source_commit']
                or live['speakers']['timeline']['backend'] != case['backend']):
            raise ValueError('qualified complete source case identity differs')
        result['cases'][case['name']] = frontier_observations(live['provisional_revisions'], 80.04)
    frozen()
    result['analysis_completed'] = True
    out.mkdir(parents=True)
    (out / 'protocol.json').write_bytes(raw)
    (out / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
