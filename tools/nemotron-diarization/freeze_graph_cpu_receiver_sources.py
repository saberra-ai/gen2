"""Freeze graph-only saved-data receiver derivation before any native diagnostic."""
import json
from pathlib import Path

from graph_cpu_receiver_sources import NATIVE_PROTOCOL, ORIGIN, ORIGIN_FILES, bound_origins, identity


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    bound_origins(repo)
    native = json.loads((tools / 'windows-graph-cpu-profile.protocol.json').read_bytes())
    for group, root in [('tools', tools), ('repository_files', repo)]:
        for name, expected in native[group].items():
            if identity((root / name).read_bytes()) != expected:
                raise ValueError('native input changed: ' + name)
    source_raw = (tools / 'graph-cpu-source-admission.protocol.json').read_bytes()
    source = json.loads(source_raw)
    if source['native_protocol'] != NATIVE_PROTOCOL or source['minimum_free_bytes'] != 8 * 1024**3:
        raise ValueError('original source composition protocol and floor required')
    names = {ORIGIN + '/' + name for name in ORIGIN_FILES}
    names |= set(source['files']) | {'tools/nemotron-diarization/graph-cpu-source-admission.protocol.json'}
    names |= {'tools/nemotron-diarization/' + name for name in (
        'graph_cpu_receiver_sources.py', 'test_graph_cpu_receiver_sources.py', 'freeze_graph_cpu_receiver_sources.py')}
    names.add('docs/research/nemotron-graph-cpu-receiver-preparation.md')
    for name, expected in source['files'].items():
        if identity((repo / name).read_bytes()) != expected:
            raise ValueError('source composition dependency changed: ' + name)
    protocol = {'schema_version': 1, 'native_protocol': NATIVE_PROTOCOL,
        'native_tools': 149, 'native_repository_references': 2889,
        'linux_full_native_start_floor_bytes': 8 * 1024**3,
        'local_bounded_result_start_floor_bytes': 4 * 1024**3,
        'native_zip_cap_bytes': 128 * 1024**2, 'native_expansion_cap_bytes': 1024**3,
        'result_bundle_cap_bytes': 128 * 1024**2,
        'files': {name: identity((repo / name).read_bytes()) for name in sorted(names)},
        'native_identity': None, 'artifact_consumed': False, 'receivers_invoked': False,
        'limits': [
            'Source preparation only. No native dispatch, download, SDK/model execution or saved-artifact replay.',
            'Actual full native source/run/job/API/archive and feature CI identities must be frozen before consumption.',
            'The original Windows command supervision, 3425 source files, 97 compiler commands, private PE/66 SDK files and all three retained import libraries remain mandatory.',
            'All twelve graph processes/144 vectors precede four complete 6000x8 histories, 58 pushes and exact turns. Replay raw graph-only CPU and wall traces, compact profiles and both original 10 percent observer bounds.',
            'Graph CPU values retain 100ns units and genuine zero values. No per-region CPU attribution, cycle conversion, exact descheduled-time or pure-arithmetic claim.',
            'Owned-process remote acquisition, original floors, actual API/archive binding, bounded result-only return, independent local replay and review remain separately required before native execution.',
            'All original quality, physical desktop, human-review, fresh-machine, platform and signing gates remain open. No main or release promotion.']}
    raw = (json.dumps(protocol, indent=2) + '\n').encode()
    (tools / 'graph-cpu-receiver-sources.protocol.json').write_bytes(raw)
    print(json.dumps({'protocol': identity(raw), 'files': len(names), 'new_native_calls': 0}))


if __name__ == '__main__':
    main()
