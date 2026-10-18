"""Freeze the independent graph CPU local replay before native dispatch."""
import json
from pathlib import Path

from graph_cpu_local_replay import NATIVE_PROTOCOL, ORIGIN, ORIGIN_ID, capture, identity, read


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    capture(repo / ORIGIN, ORIGIN_ID)
    names = set()
    for filename in ('remote-graph-cpu-replay.protocol.json', 'graph-cpu-bundle-receive.protocol.json'):
        path = tools / filename
        raw = capture(path)
        protocol = read(raw)
        if protocol['native_protocol'] != NATIVE_PROTOCOL:
            raise ValueError('exact native protocol required')
        for name, expected in protocol['files'].items():
            capture(repo / name, expected)
        names |= set(protocol['files']) | {'tools/nemotron-diarization/' + filename}
    native = read(capture(tools / 'windows-graph-cpu-profile.protocol.json', NATIVE_PROTOCOL))
    for group, root in [('tools', tools), ('repository_files', repo)]:
        for name, expected in native[group].items():
            capture(root / name, expected)
    names |= {ORIGIN, 'docs/research/nemotron-graph-cpu-local-replay.md'}
    names |= {'tools/nemotron-diarization/' + name for name in (
        'graph_cpu_local_replay.py', 'test_graph_cpu_local_replay.py', 'freeze_graph_cpu_local_replay.py')}
    value = {'schema_version': 1, 'native_protocol': NATIVE_PROTOCOL,
        'files': {name: identity(capture(repo / name)) for name in sorted(names)},
        'local_floor_bytes': 4 * 1024**3, 'new_compiler_sdk_model_calls': 0,
        'limits': ['Pre-execution recipe only; actual native/result identities remain separately frozen in transport protocols.',
            'All59 native commands,97 compiler entries,12 graph processes/144 vectors and four complete model histories retain original gates.',
            'Every returned Python AST must match the exact locally derived reviewed graph-only sources with two recorded literal prefix substitutions; returned modules are never imported.',
            'Local metadata command count is21 plus admitted result-member count; source19commands retain original8GiB and outer-owned supervision. Every returned remote/local command sidecar is checked.',
            'Only a passed actual raw-data replay establishes a result. Full native ZIP/SDK/control EXE/pristine archive remain remote; local replay excludes exactly the unavailable EXE stat.',
            'Graph CPU100ns values and genuine zeros remain separate from wall time. Both original observer bounds govern interpretation. No production, release or main qualification.']}
    raw = (json.dumps(value, indent=2) + '\n').encode()
    (tools / 'graph-cpu-local-replay.protocol.json').write_bytes(raw)
    print(json.dumps({'protocol': identity(raw), 'files': len(names), 'new_native_calls': 0}))


if __name__ == '__main__':
    main()
