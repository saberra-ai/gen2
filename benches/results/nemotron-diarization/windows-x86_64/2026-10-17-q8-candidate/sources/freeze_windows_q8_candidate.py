"""Freeze unchanged bounded public semantics and the exact candidate producer."""
import ast
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from pooled_runtime import BOUNDED_PROFILE, load_profile, profile_provenance
from windows_q8_candidate import ORDER, LIFECYCLE, PROTOCOL
from windows_q8_candidate_metrics import BOUNDED_RESULT, BOUNDED_MANIFEST, PCM_SHA256
from windows_q8_candidate_producer_sources import derive_build
from windows_q8_candidate_sources import load_recipe, PARENT_RESULT, RECIPE_ID, RECIPE
from freeze_windows_q8_model import consumed_fp_files


def inputs(tools):
    tools = Path(tools)
    repo = tools.parents[1]
    parent_path = tools / 'windows-compiler-fp.protocol.json'
    if descriptor(parent_path) != {'bytes': 179909, 'sha256': '59d8b60f961ef91fe391f92b1b56440e397d6cb806125d88993c9242c615b87d'}:
        raise ValueError('exact original contraction protocol required')
    parent = json.loads(parent_path.read_text())
    for group, root in [('tools', tools), ('repository_files', repo)]:
        for name, expected in parent[group].items():
            if descriptor(root / name) != expected:
                raise ValueError('original SDK input changed: ' + name)
    if (tools / 'windows_q8_candidate_build.py').read_bytes() != derive_build(
            (tools / 'platform_bounded_windows_native.py').read_bytes()):
        raise ValueError('exact bounded SDK build derivation required')
    recipe_path, recipe = load_recipe()
    if descriptor(repo / BOUNDED_RESULT / 'collection.json') != BOUNDED_MANIFEST:
        raise ValueError('exact original bounded source collection required')
    references = set(parent['repository_files']) | consumed_fp_files(repo) | {
        '.github/workflows/diarization-windows-q8-candidate.yml',
        'docs/research/nemotron-windows-q8-candidate-producer.md',
        'docs/research/nemotron-windows-q8-model-results.md'}
    for base, names in [(BOUNDED_RESULT, ['collection.json', 'platform-llama-materialized-audit.json']),
            (PARENT_RESULT, ['collection.json', 'collection-qualification.json'])]:
        manifest = json.loads((repo / base / 'collection.json').read_text())['files']
        for name in names:
            if name != 'collection.json' and descriptor(repo / base / name) != manifest[name]:
                raise ValueError('actual parent payload changed: ' + base + '/' + name)
            references.add(base + '/' + name)
    names = set(parent['tools']) | {parent_path.name, RECIPE, 'ggml_matrix_profile_control.cpp',
        'patches/q8-model-mx4.patch', 'fixtures/q8-tile-sgemm.cpp', 'fixtures/q8-model-original-series.txt'}
    pending = ['windows_q8_candidate.py', 'freeze_windows_q8_candidate.py',
               'test_windows_q8_candidate.py', 'test_windows_q8_candidate_producer.py', 'test_windows_job_v2.py']
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        names.add(name)
        for node in ast.walk(ast.parse((tools / name).read_bytes())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                local = (module or '').split('.')[0] + '.py'
                if (tools / local).is_file() and local not in visited:
                    pending.append(local)
    lock_path, lock = load_profile(BOUNDED_PROFILE)
    return {'schema_version': 1, 'order': ORDER, 'actual_pcm_sha256': PCM_SHA256, 'required_lifecycle_tests': LIFECYCLE,
        'runtime_provenance': profile_provenance(lock_path, lock), 'build_recipe_id': RECIPE_ID,
        'build_recipe': descriptor(recipe_path), 'source_sha256_after': recipe['source_sha256_after'],
        'tools': {name: descriptor(tools / name) for name in sorted(names)},
        'repository_files': {name: descriptor(repo / name) for name in sorted(references)},
        'local_import_closure': sorted(visited),
        'limits': ['Production-candidate same-host source SDK/public/lifecycle qualification; packaging, offline relocation, coexistence, desktop and release remain separate.',
            'Original bounded V3 semantics, three native patches and four-worker policy; no diagnostic worker source or telemetry.',
            'Only exact SGEMM dispatch changes GGML; original compilers/linker/97 commands and contraction-off admission required.',
            'Retain all 97 compiler objects and all three graph import-library payloads before graph compilation, with source/command associations.',
            'Thirty-six full graph vectors must be byte-exact across workers1/2/4 before model acquisition.',
            'Four fresh native/public/public/native 60-second histories and both complete pairs must match every frozen probability byte.',
            'All four original native/public RTFs must be strictly below one; no retry, averaging or changed quality threshold.',
            'Original seven lifecycle tests and owned Windows Job cleanup must pass; failed timing remains recorded independently.',
            'Earlier 5-percent diagnostic prediction and incomplete graph retention remain failed; no automatic main or release promotion.']}


def main():
    tools = Path(__file__).resolve().parent
    p = inputs(tools)
    path = tools / PROTOCOL
    path.write_text(json.dumps(p, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(path), 'tools': len(p['tools']),
        'repository_files': len(p['repository_files']), 'local_imports': len(p['local_import_closure'])}))


if __name__ == '__main__':
    main()
