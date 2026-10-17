"""Freeze actual SDK inputs and compact, fully re-admitted diagnostic ancestry."""
import ast
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from windows_q8_model_sources import (candidate_source, derive_compiler, derive_verifier,
    derive_worker, source_hashes)
from windows_q8_model_metrics import ORDER, PCM_SHA256


FP_RESULT = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-fp'
FP_PAYLOADS = ['CMakeCache.txt', 'compile_commands.json', 'compiler-C.cmake', 'compiler-CXX.cmake',
               'compiler-admission.json', 'platform-llama-materialized-audit.json']


def consumed_fp_files(repo):
    root = Path(repo) / FP_RESULT
    if descriptor(root / 'collection.json') != {'bytes': 49610, 'sha256': 'c945267fbffb89dff2e36f1afc93bcb038f0ecac691e0f8af87c0c4a26c8486f'}:
        raise ValueError('exact actual compiler-FP collection required')
    manifest = json.loads((root / 'collection.json').read_text())['files']
    for name, expected in manifest.items():
        if descriptor(root / name) != expected:
            raise ValueError('actual compiler-FP payload changed: ' + name)
    names = ['collection.json', *['evidence-clang-cl/' + name for name in FP_PAYLOADS]]
    return {FP_RESULT + '/' + name for name in names}


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent_path = tools / 'windows-compiler-fp.protocol.json'
    if descriptor(parent_path) != {'bytes': 179909, 'sha256': '59d8b60f961ef91fe391f92b1b56440e397d6cb806125d88993c9242c615b87d'}:
        raise ValueError('exact full-SDK contraction protocol required')
    parent = json.loads(parent_path.read_text())
    for group, root in [('tools', tools), ('repository_files', repo)]:
        for name, expected in parent[group].items():
            if descriptor(root / name) != expected:
                raise ValueError('historical SDK input changed: ' + name)
    # Re-admit complete recent ancestry now. Retain its protocol and manifest
    # identities at runtime; old duplicate vectors are not inputs to this SDK.
    lineage_path = tools / 'windows-q8-worker-timing.protocol.json'
    if descriptor(lineage_path) != {'bytes': 785549, 'sha256': '0eb0d37780ead508c3622cc57c4b7f0ae8ba32da3ebe317e0d46417707a26d97'}:
        raise ValueError('exact preceding worker timing protocol required')
    lineage = json.loads(lineage_path.read_text())
    for group, root in [('tools', tools), ('repository_files', repo)]:
        for name, expected in lineage[group].items():
            if descriptor(root / name) != expected:
                raise ValueError('historical kernel input changed: ' + name)
    references = set(parent['repository_files']) | consumed_fp_files(repo) | {
        '.github/workflows/diarization-windows-q8-model.yml',
        'docs/research/nemotron-windows-q8-model.md',
        'docs/research/nemotron-windows-q8-mx4-results.md',
        'docs/research/nemotron-windows-q8-worker-timing-results.md'}
    for receipt, expected_manifest in [
        ('2026-10-17-q8-mx4', {'bytes': 90509, 'sha256': '04b1990d3e6b92fa88d9ef0a59546465514d484c49300be6dbe66310ddeeb9db'}),
        ('2026-10-17-q8-worker-timing', {'bytes': 209383, 'sha256': 'c9f46380584c36f34352a2dc8b2d0fa62eb1da2dd0c6a58dc8f9c43b2f506bfc'}),
        ('2026-10-17-matrix-batch-profile', {'bytes': 67052, 'sha256': 'dd44361053387a87afeff82537d93873f1753d634bb57917c2e8db8e127fa8ea'})]:
        base = 'benches/results/nemotron-diarization/windows-x86_64/' + receipt
        if descriptor(repo / base / 'collection.json') != expected_manifest:
            raise ValueError('exact preceding receipt required: ' + receipt)
        manifest = json.loads((repo / base / 'collection.json').read_text())['files']
        for name, expected in manifest.items():
            if descriptor(repo / base / name) != expected:
                raise ValueError('preceding actual evidence changed: ' + receipt + '/' + name)
        references |= {base + '/collection.json', base + '/collection-qualification.json'}
    for target, origin, derive in [
        ('windows_q8_model_worker.py', 'windows_worker_diagnostic_v2.py', derive_worker),
        ('windows_q8_model_verify.py', 'windows_worker_sources_v2.py', derive_verifier),
        ('windows_q8_model_compiler.py', 'windows_compiler_fp_diagnostic.py', derive_compiler)]:
        if (tools / target).read_bytes() != derive((tools / origin).read_bytes()):
            raise ValueError('exact historical construction derivation required: ' + target)
    names = set(parent['tools']) | {parent_path.name, lineage_path.name, 'windows-q8-mx4.protocol.json',
        'ggml_matrix_profile_control.cpp', 'patches/q8-model-mx4.patch', 'fixtures/q8-tile-sgemm.cpp',
        'fixtures/q8-model-original-series.txt'}
    pending = ['windows_q8_model_diagnostic.py', 'test_windows_q8_model.py', 'freeze_windows_q8_model.py']
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
    original_series = (tools / 'fixtures/q8-model-original-series.txt').read_bytes()
    protocol = {'schema_version': 1, 'order': ORDER, 'actual_pcm_sha256': PCM_SHA256,
        'parent_full_sdk_protocol': descriptor(parent_path), 'preceding_kernel_protocol': descriptor(lineage_path),
        'candidate_source_bytes': len(candidate_source()),
        'variants': {variant: {'source_sha256': source_hashes(parent['variants']['4']['source_sha256'], original_series, variant)}
                     for variant in ('control', 'mx4')},
        'tools': {name: descriptor(tools / name) for name in sorted(names)},
        'repository_files': {name: descriptor(repo / name) for name in sorted(references)},
        'local_import_closure': sorted(visited),
        'limits': ['Native-only diagnostic; original quality/performance/production/public/release requirements remain.',
            'Both SDKs retain original compiler/linker/97 commands, contraction off and four-worker policy.',
            'Control has 19 original patches; candidate adds only exact SGEMM dispatch and active source guard.',
            'Two SDKs, six actual graph processes and 72 complete vectors must be per-shape byte-exact before model acquisition.',
            'Four fresh control/mx4/mx4/control complete 60-second/58-push histories must be byte-exact; no relaxed tolerance, retry or averaging.',
            '5-percent model speedup and 10-percent control drift remain separate from both candidates strictly below-one RTF.',
            'Failed standalone/worker timing predictions remain failed; no automatic production/main promotion.']}
    target = tools / 'windows-q8-model.protocol.json'
    target.write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'tools': len(names), 'repository_files': len(references),
        'reference_bytes': sum(row['bytes'] for row in protocol['repository_files'].values()), 'local_import_closure': len(visited)}))


if __name__ == '__main__':
    main()
