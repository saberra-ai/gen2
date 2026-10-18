"""Freeze the one-flag controller, complete local imports and actual prior evidence."""
import ast
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from windows_math_errno_sources import derive_runner


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent_path = tools / 'windows-compiler-fp.protocol.json'
    if descriptor(parent_path) != {'bytes': 179909, 'sha256': '59d8b60f961ef91fe391f92b1b56440e397d6cb806125d88993c9242c615b87d'}:
        raise ValueError('preceding full compiler protocol changed')
    parent = json.loads(parent_path.read_text())
    for group, root in [('tools', tools), ('repository_files', repo)]:
        for name, expected in parent[group].items():
            if descriptor(root / name) != expected:
                raise ValueError('historical input changed: ' + name)
    if (tools / 'windows_math_errno_diagnostic.py').read_bytes() != derive_runner((tools / 'windows_compiler_fp_diagnostic.py').read_bytes()):
        raise ValueError('math errno controller derivation changed')
    references = set(parent['repository_files']) | {
        '.github/workflows/diarization-windows-math-errno.yml', 'docs/research/nemotron-windows-math-errno.md'}
    for receipt in ['2026-10-17-compiler-fp', '2026-10-17-cpu-profile']:
        base = 'benches/results/nemotron-diarization/windows-x86_64/' + receipt
        manifest = json.loads((repo / base / 'collection.json').read_text())['files']
        for name, expected in manifest.items():
            if descriptor(repo / base / name) != expected:
                raise ValueError('actual preceding receipt changed: ' + base + '/' + name)
        references |= {base + '/' + name for name in manifest} | {base + '/collection.json'}
    names = set(parent['tools']) | {parent_path.name, 'fp_contract_probe.cpp'}
    pending = ['windows_math_errno_diagnostic.py', 'test_windows_math_errno.py', 'freeze_windows_math_errno.py']
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
    protocol = {'schema_version': 1, 'order': ['control', 'no-errno', 'no-errno', 'control'],
        'actual_pcm_sha256': parent['actual_pcm_sha256'], 'variants': {'4': parent['variants']['4']},
        'tools': {name: descriptor(tools / name) for name in sorted(names)},
        'repository_files': {name: descriptor(repo / name) for name in sorted(references)},
        'limits': ['Native-only diagnostic; original numerical/performance/quality/production/release requirements remain.',
            'Identical original nineteen-patch inference source, compiler/linker/MSVC probe binaries and four-worker policy.',
            'Both SDKs explicit contraction off; candidate differs only by one C/CXX math-errno flag.',
            'All 97 commands, complete source/SDK evidence and actual effective compiler math flags required.',
            'Four full 60-second/58-push histories must be byte-exact; no relaxed tolerance or automatic retry.',
            '5-percent speedup and 10-percent control drift predicates do not waive strict below-one native/public RTF.']}
    path = tools / 'windows-math-errno.protocol.json'
    path.write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(path), 'tools': len(names), 'repository_files': len(references), 'local_import_closure': len(visited)}))


if __name__ == '__main__':
    main()
