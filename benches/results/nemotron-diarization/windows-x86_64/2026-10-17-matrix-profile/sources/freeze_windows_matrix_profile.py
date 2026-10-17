"""Freeze exact source adapters, full local closure and real preceding receipts."""
import ast
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from windows_matrix_profile_sources import derive_compiler, derive_verifier, derive_worker, PATCH_TOOL, profile_source_hashes
from windows_matrix_profile_metrics import version


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent_path = tools / 'windows-compiler-fp.protocol.json'
    if descriptor(parent_path) != {'bytes': 179909, 'sha256': '59d8b60f961ef91fe391f92b1b56440e397d6cb806125d88993c9242c615b87d'}:
        raise ValueError('frozen preceding full compiler protocol changed')
    parent = json.loads(parent_path.read_text())
    cpu_parent_path = tools / 'windows-cpu-profile.protocol.json'
    if descriptor(cpu_parent_path) != {'bytes': 275374, 'sha256': '15ba8af0c151e38b608437da32e39f5e4b92b9ebb5db2c6ae2692c8f910b6dd5'}:
        raise ValueError('frozen preceding CPU operation protocol changed')
    cpu_parent = json.loads(cpu_parent_path.read_text())
    for previous in (parent, cpu_parent):
        for group, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in previous[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('historical input changed: ' + name)
    for target, origin, derive in [
            ('windows_matrix_profile_worker.py', 'windows_cpu_profile_worker.py', derive_worker),
            ('windows_matrix_profile_verify.py', 'windows_cpu_profile_verify.py', derive_verifier),
            ('windows_matrix_profile_compiler.py', 'windows_cpu_profile_compiler.py', derive_compiler)]:
        if (tools / target).read_bytes() != derive((tools / origin).read_bytes()):
            raise ValueError('CPU profile adapter derivation changed')
    references = set(parent['repository_files']) | set(cpu_parent['repository_files']) | {
        '.github/workflows/diarization-windows-matrix-profile.yml',
        'docs/research/nemotron-windows-matrix-profile.md'}
    for receipt in ['windows-x86_64/2026-10-17-compiler-fp','windows-x86_64/2026-10-17-cpu-profile',
                    'apple-m4-pro/2026-10-17-matrix-path-control-failure',
                    'apple-m4-pro/2026-10-17-matrix-path-sgemm-control']:
        base = 'benches/results/nemotron-diarization/' + receipt
        manifest = json.loads((repo / base / 'collection.json').read_text())['files']
        for name, expected in manifest.items():
            if descriptor(repo / base / name) != expected:
                raise ValueError('actual preceding receipt changed: ' + base + '/' + name)
        references |= {base + '/' + name for name in manifest} | {base + '/collection.json'}
    names = set(parent['tools']) | set(cpu_parent['tools']) | {parent_path.name,PATCH_TOOL,'ggml_matrix_profile_control.cpp','fixtures/matrix-paths-ggml-cpu.c',cpu_parent_path.name}
    pending = ['windows_matrix_profile_diagnostic.py','test_windows_matrix_profile.py',
               'test_ggml_matrix_profile.py','test_windows_matrix_profile_report.py','freeze_windows_matrix_profile.py']
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
    variant = {**parent['variants']['4'],'reported_version':version(4),'source_sha256': profile_source_hashes(parent['variants']['4']['source_sha256'])}
    p = {'schema_version': 1, 'order': [False, True, True, False], 'expected_graphs':84,'expected_matrices':129,
        'actual_pcm_sha256': parent['actual_pcm_sha256'], 'variants': {'4': variant},
        'tools': {name: descriptor(tools / name) for name in sorted(names)},
        'repository_files': {name: descriptor(repo / name) for name in sorted(references)},
        'limits': ['Diagnostic-only CPU graph profiling with unchanged original numerical/performance/release criteria.',
            'One exact extra GGML matrix/timing patch and distinct diagnostic identity; no arithmetic, fusion, task or barrier change.',
            'Exact original Clang/MSVC probe/linker binaries, contraction-off flag and otherwise matching compiler arguments.',
            'Complete actual Windows model-free controls precede model acquisition; full model off/on/on/off outputs must remain byte-exact.',
            'Expected 84 graphs,129 matrix roots per graph and all four workers are mandatory; preserve failures without changing coverage requirements.',
            'Worker wall times overlap and include internal waits, scheduling and observer perturbation; no pure arithmetic or release claim.']}
    target = tools / 'windows-matrix-profile.protocol.json'
    target.write_text(json.dumps(p, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'tools': len(names), 'repository_files': len(references),
                      'local_import_closure': len(visited)}))


if __name__ == '__main__':
    main()
