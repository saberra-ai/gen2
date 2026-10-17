"""Freeze the contraction-only full SDK experiment and its actual baselines."""
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from windows_compiler_fp_sources import derive_runner


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent = json.loads((tools / 'windows-compiler-diagnostic-v2.protocol.json').read_text())
    for group, root in [('tools', tools), ('repository_files', repo)]:
        for name, expected in parent[group].items():
            if descriptor(root / name) != expected:
                raise ValueError('original full compiler experiment changed: ' + name)
    if (tools / 'windows_compiler_fp_diagnostic.py').read_bytes() != derive_runner(
            (tools / 'windows_compiler_diagnostic_v2.py').read_bytes()):
        raise ValueError('controller exceeds exact contraction adapter derivation')
    base = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-parity-failure'
    scalar = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-fp-contract-probe'
    for receipt in (base, scalar):
        collection = json.loads((repo / receipt / 'collection.json').read_text())
        for name, expected in collection['files'].items():
            if descriptor(repo / receipt / name) != expected:
                raise ValueError('actual diagnostic receipt changed: ' + receipt + '/' + name)
    names = set(parent['tools']) | {'windows-compiler-diagnostic-v2.protocol.json',
        'windows_compiler_fp_metrics.py', 'windows_compiler_fp_sources.py',
        'windows_compiler_fp_diagnostic.py', 'test_windows_compiler_fp.py', 'freeze_windows_compiler_fp.py'}
    references = set(parent['repository_files']) | {
        base + '/collection.json', base + '/collection-qualification.json',
        scalar + '/collection.json', scalar + '/collection-qualification.json',
        '.github/workflows/diarization-windows-compiler-fp.yml',
        'docs/research/nemotron-windows-compiler-fp.md',
        'docs/research/nemotron-windows-compiler-results.md', 'docs/research/nemotron-windows-fp-contract-results.md'}
    references |= {base + '/evidence-' + kind + '/' + name for kind in ('msvc', 'clang-cl')
                   for name in ('CMakeCache.txt', 'compile_commands.json')}
    p = {'schema_version': 1, 'order': parent['order'], 'actual_pcm_sha256': parent['actual_pcm_sha256'],
        'variants': parent['variants'], 'tools': {name: descriptor(tools / name) for name in sorted(names)},
        'repository_files': {name: descriptor(repo / name) for name in sorted(references)},
        'limits': parent['limits'] + [
            'Only candidate change: initialize Clang C/CXX flags with /clang:-ffp-contract=off, retaining platform defaults.',
            'Exact scalar-tested compiler binaries and original linkers; all compiler commands otherwise match saved baselines.',
            'Normalize only selected compiler path alias and isolated source root; original numerical/timing/release gates unchanged.',
            'Scalar contraction evidence supplies a hypothesis, not model causality or a release qualification.']}
    target = tools / 'windows-compiler-fp.protocol.json'
    target.write_text(json.dumps(p, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'tools': len(p['tools']),
                      'repository_files': len(p['repository_files'])}))


if __name__ == '__main__':
    main()
