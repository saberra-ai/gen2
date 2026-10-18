"""Freeze the graph-only diagnostic without building native code or acquiring models."""
import ast
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from windows_graph_cpu_profile_sources import (derive_compiler, derive_verifier,
    derive_worker, derive_metrics, derive_diagnostic, PATCH_TOOL, profile_source_hashes)
from windows_graph_cpu_profile_metrics import version


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent_path = tools / 'windows-thread-cpu-profile.protocol.json'
    if descriptor(parent_path) != {'bytes': 670257, 'sha256': 'e9809ffe025bbe127c27daa5f8dfa07916cab7fe24a40d074651103b8c3cfc66'}:
        raise ValueError('frozen preceding thread/cycle diagnostic protocol required')
    parent = json.loads(parent_path.read_bytes())
    for group, root in [('tools', tools), ('repository_files', repo)]:
        for name, expected in parent[group].items():
            if descriptor(root / name) != expected:
                raise ValueError('historical source or receipt changed: ' + name)
    compiler_path = tools / 'windows-compiler-fp.protocol.json'
    compiler = json.loads(compiler_path.read_bytes())
    if descriptor(compiler_path) != parent['tools'][compiler_path.name]:
        raise ValueError('original contraction-off compiler protocol required')
    for target, origin, derive in [
            ('windows_graph_cpu_profile_worker.py', 'windows_cpu_profile_worker.py', derive_worker),
            ('windows_graph_cpu_profile_verify.py', 'windows_cpu_profile_verify.py', derive_verifier),
            ('windows_graph_cpu_profile_compiler.py', 'windows_cpu_profile_compiler.py', derive_compiler),
            ('windows_graph_cpu_profile_metrics.py', 'windows_thread_cpu_profile_metrics.py', derive_metrics),
            ('windows_graph_cpu_profile_diagnostic.py', 'windows_thread_cpu_profile_diagnostic.py', derive_diagnostic)]:
        if (tools / target).read_bytes() != derive((tools / origin).read_bytes()):
            raise ValueError('import-only graph diagnostic adapter differs: ' + target)
    references = set(parent['repository_files']) | {
        '.github/workflows/diarization-windows-graph-cpu-profile.yml',
        'docs/research/nemotron-windows-graph-cpu-profile.md'}
    receipt = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-graph-cpu-api-ci'
    manifest = json.loads((repo / receipt / 'manifest.json').read_bytes())
    for name, expected in manifest.items():
        if descriptor(repo / receipt / name) != expected:
            raise ValueError('actual Windows API receipt changed: ' + name)
    references |= {receipt + '/' + name for name in manifest} | {receipt + '/manifest.json'}
    names = set(parent['tools']) | {parent_path.name, PATCH_TOOL}
    pending = ['windows_graph_cpu_profile_diagnostic.py', 'test_windows_graph_cpu_profile.py',
               'test_ggml_graph_cpu_profile.py', 'test_ggml_graph_cpu_windows.py',
               'test_windows_job_v2.py', 'freeze_windows_graph_cpu_profile.py']
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        names.add(name)
        for node in ast.walk(ast.parse((tools / name).read_bytes())):
            imports = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            for module in imports:
                local = (module or '').split('.')[0] + '.py'
                if (tools / local).is_file() and local not in visited:
                    pending.append(local)
    variant = {**compiler['variants']['4'], 'reported_version': version(4),
               'source_sha256': profile_source_hashes(compiler['variants']['4']['source_sha256'])}
    protocol = {
        'schema_version': 1, 'order': [False, True, True, False],
        'expected_graphs': 84, 'expected_matrices': 129, 'cpu_time_units': '100ns',
        'cpu_observer': 'two GetThreadTimes reads per enabled worker graph; no cycle counters',
        'matrix_emission_bytes_per_row': 2048,
        'actual_pcm_sha256': parent['actual_pcm_sha256'], 'variants': {'4': variant},
        'tools': {name: descriptor(tools / name) for name in sorted(names)},
        'repository_files': {name: descriptor(repo / name) for name in sorted(references)},
        'limits': [
            'Diagnostic-only graph CPU observation, never a production or release runtime.',
            'Exact batched wall observer plus declared graph-only CPU reads; arithmetic, planning, fusion and barriers unchanged.',
            'Original compiler/linker binaries, contraction-off flags, private PE and complete source/compiler/import bindings remain mandatory.',
            'Twelve fresh Windows graph processes at one/two/four workers, off/on/on/off, and all144 finite nonzero vectors precede model acquisition.',
            'Four fresh sixty-second model processes retain off/on/on/off order,6000x8 complete histories,58pushes,27turns and byte-exact original Q8 parity.',
            'Both original integer10percent predicates must pass before interpretation; every real-time, quality, app and release gate remains unchanged.',
            'Kernel/user100ns values retain genuine zeros and reject API errors/regressions. No per-region CPU share, cycles, exact descheduled time, pure arithmetic or causal scheduling claim.'
        ]}
    target = tools / 'windows-graph-cpu-profile.protocol.json'
    target.write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'tools': len(names),
                      'repository_files': len(references), 'local_import_closure': len(visited)}))


if __name__ == '__main__':
    main()
