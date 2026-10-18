"""Freeze the graph-only transport recipe; actual identities are separately observed."""
import argparse
import json
from pathlib import Path

from graph_cpu_transport import NATIVE_PROTOCOL, ORIGINS, RECIPE, admit_identities, identity, strict_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-identity', type=Path)
    parser.add_argument('--return-identity', type=Path)
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    recipe_path = tools / 'graph-cpu-receiver-sources.protocol.json'
    recipe_raw = recipe_path.read_bytes()
    if identity(recipe_raw) != RECIPE:
        raise ValueError('exact reviewed receiver recipe required')
    recipe = strict_json(recipe_raw)
    native_raw = (tools / 'windows-graph-cpu-profile.protocol.json').read_bytes()
    if identity(native_raw) != NATIVE_PROTOCOL:
        raise ValueError('exact native graph-only protocol required')
    native = strict_json(native_raw)
    names = set(recipe['files']) | {'tools/nemotron-diarization/graph-cpu-receiver-sources.protocol.json'}
    names |= {'tools/nemotron-diarization/' + name for name in (*ORIGINS,
        'test_receive_thread_cpu_bundle.py', 'receive_q8_apack_bundle.py',
        'graph_cpu_transport.py', 'test_graph_cpu_transport.py', 'freeze_graph_cpu_transport.py')}
    names |= {'.github/workflows/diarization-graph-cpu-replay.yml',
        'docs/research/nemotron-graph-cpu-transport.md'}
    for group, root in [('tools', tools), ('repository_files', repo)]:
        for name, expected in native[group].items():
            if identity((root / name).read_bytes()) != expected:
                raise ValueError('native input changed: ' + name)
    for name, expected in recipe['files'].items():
        if identity((repo / name).read_bytes()) != expected:
            raise ValueError('receiver recipe dependency changed: ' + name)
    for name, expected in ORIGINS.items():
        if identity((tools / name).read_bytes()) != expected:
            raise ValueError('original transport source changed: ' + name)
    common = {'schema_version': 1, 'native_protocol': NATIVE_PROTOCOL, 'receiver_recipe': RECIPE,
        'native_source': None, 'native_run': None, 'native_job': None, 'native_feature_run': None,
        'artifact_id': None, 'archive': None, 'original_start_floor_bytes': 8 * 1024**3,
        'native_archive_cap_bytes': 256 * 1024**2, 'bundle_cap_bytes': 128 * 1024**2,
        'files': {name: identity((repo / name).read_bytes()) for name in sorted(names)},
        'limits': ['No native execution. Identities remain unset until actual terminal API/archive evidence exists.',
            'Full-native acquisition/replay requires original8GiB Linux floor and owned groups; actual complete source/compiler/PE/library/graph/model-output evidence must replay.',
            'Native-source feature run must independently pass all13 jobs; raw feature API responses return with the bundle and are checked again locally.',
            'Local bounded return requires original4GiB floor, exact ZIP/bundle/result identities and separate object-store quarantine; source refs/index stay unchanged.',
            'Local quarantine integrity is separate from independent local semantic replay and review. No production/main or release qualification.']}
    if args.native_identity:
        actual = strict_json(args.native_identity.read_bytes())
        keys = ('native_source', 'native_run', 'native_job', 'native_feature_run', 'artifact_id', 'archive')
        if set(actual) != set(keys):
            raise ValueError('exact native identity fields required')
        common.update(actual)
        admit_identities(common, False)
    elif args.return_identity:
        raise ValueError('native identity required before return identity')
    local = {**common, 'remote_source': None, 'remote_run_id': None, 'remote_job_id': None,
        'delivery': None, 'result_commit': None, 'bundle': None, 'collection': None,
        'zip_cap': 128 * 1024**2, 'extract_cap': 192 * 1024**2, 'bundle_cap': 128 * 1024**2,
        'start_floor': 4 * 1024**3}
    if args.return_identity:
        actual = strict_json(args.return_identity.read_bytes())
        keys = ('remote_source', 'remote_run_id', 'remote_job_id', 'delivery', 'result_commit', 'bundle', 'collection')
        if set(actual) != set(keys):
            raise ValueError('exact return identity fields required')
        local.update(actual)
        admit_identities(local, True)
    results = {}
    for filename, value in [('remote-graph-cpu-replay.protocol.json', common),
                            ('graph-cpu-bundle-receive.protocol.json', local)]:
        raw = (json.dumps(value, indent=2) + '\n').encode()
        (tools / filename).write_bytes(raw)
        results[filename] = identity(raw)
    print(json.dumps({'protocols': results, 'files': len(names), 'identities_set': bool(args.native_identity),
        'new_native_calls': 0}))


if __name__ == '__main__':
    main()
