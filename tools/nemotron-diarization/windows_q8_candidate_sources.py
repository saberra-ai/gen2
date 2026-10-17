"""Pinned production-candidate recipe; preserve bounded frame/ABI semantics."""
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from pooled_runtime import BOUNDED_PROFILE, load_profile
from windows_compiler_fp_metrics import FLAG, baseline
from windows_cpu_profile_sources import once
from windows_q8_model_sources import (PATCH_TOOL, Q8_PATCH, ORIGINAL, candidate_source,
    install_q8_patch, source_hashes, verify_q8_series)

RECIPE = 'windows-q8-candidate.recipe.json'
RECIPE_ID = 'gen2-windows-cpu-q8-mx4-clang-v1'
PARENT_RESULT = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-model'
PARENT_MANIFEST = {'bytes': 69353, 'sha256': 'ef01e3c6ba330b48fbfcbb36b2714862324fc2c6779c9033a95aaee8c3fc203e'}
PARENT_QUALIFICATION = {'bytes': 98935, 'sha256': 'e3659a1a0b2f45cc038ed48187afe74ef625626dc14976eccd2e02521c5738e8'}


def candidate_source_hashes(original):
    tools = Path(__file__).resolve().parent
    return source_hashes(original, (tools / 'fixtures/q8-model-original-series.txt').read_bytes(), 'mx4')


def expected_recipe():
    tools = Path(__file__).resolve().parent
    path, lock = load_profile(BOUNDED_PROFILE)
    clang, msvc = baseline('clang-cl')[2], baseline('msvc')[2]
    raw = candidate_source()
    import hashlib
    return {'schema_version': 1, 'recipe_id': RECIPE_ID,
        'status': 'production candidate; actual native/public/lifecycle/package qualification required',
        'runtime_profile': BOUNDED_PROFILE, 'runtime_reported_version': lock['runtime']['reported_version'],
        'parent_runtime_profile': {'path': path.name, **descriptor(path)},
        'native_revision': lock['runtime']['revision'], 'llama_revision': lock['runtime']['llama_revision'],
        'platform': {'os': 'Windows', 'architecture': 'x86_64', 'isa': ['AVX2', 'FMA', 'F16C']},
        'compiler': {'kind': 'clang-cl', 'contraction_flag': FLAG, 'binary': clang['compiler_binary'],
            'linker_binary': clang['linker_binary'], 'probe_compiler_binary': msvc['compiler_binary']},
        'ggml': {'original_patch_count': 19, 'candidate_patch_count': 20,
            'appended_patch_name': Q8_PATCH, 'patch': {'path': PATCH_TOOL, **descriptor(tools / PATCH_TOOL)},
            'original_sgemm': ORIGINAL,
            'candidate_sgemm': {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}},
        'source_sha256_after': candidate_source_hashes(lock['runtime']['source_sha256']),
        'parent_native_diagnostic': {'path': PARENT_RESULT, 'manifest': PARENT_MANIFEST,
            'qualification': PARENT_QUALIFICATION, 'source_commit': '919ddf1fcf20f7b16d2e5143cffda14f83fb48df'},
        'limits': ['Build recipe identity is separate from the unchanged bounded V3 semantic runtime tag.',
            'No diagnostic worker telemetry or forced worker override; retain the original bounded four-worker policy.',
            'The earlier 5-percent prediction remains failed; missing graph-library retention is repaired in this producer.',
            'A recipe is not native/public/lifecycle/package/offline/coexistence or release qualification.']}


def admit_recipe(recipe):
    if recipe != expected_recipe():
        raise ValueError('exact production-candidate recipe required')
    return recipe


def load_recipe():
    tools = Path(__file__).resolve().parent
    path = tools / RECIPE
    recipe = admit_recipe(json.loads(path.read_text()))
    root = tools.parents[1] / PARENT_RESULT
    if (descriptor(root / 'collection.json') != PARENT_MANIFEST
            or descriptor(root / 'collection-qualification.json') != PARENT_QUALIFICATION):
        raise ValueError('exact actual native diagnostic result required')
    manifest = json.loads((root / 'collection.json').read_text())['files']
    if manifest['collection-qualification.json'] != PARENT_QUALIFICATION:
        raise ValueError('native diagnostic manifest association differs')
    result = json.loads((root / 'collection-qualification.json').read_text())
    if (result['source_commit'] != recipe['parent_native_diagnostic']['source_commit']
            or result['full_sequence_bit_exact'] is not True
            or result['timing_comparison']['both_candidates_strictly_real_time'] is not True
            or result['timing_comparison']['full_model_speedup_prediction_supported'] is not False
            or result['all_preregistered_evidence_retained'] is not False):
        raise ValueError('original native outcome and limits must be preserved')
    return path, recipe


def derive_verifier(raw):
    site = b'from windows_compat import ORIGINAL_SHA256, rewrite_materializer\n'
    raw = once(raw, site, site + b'from windows_q8_candidate_sources import candidate_source_hashes, verify_q8_series, Q8_PATCH\n')
    site = b"    runtime = lock['runtime']\n"
    raw = once(raw, site, site + b"    verify_q8_series(source, 'mx4')\n")
    raw = once(raw, b'    expected = INFERENCE_FILES | ', b"    expected = INFERENCE_FILES | {'patches/series'} | ")
    raw = once(raw, b"    for name, expected_hash in runtime['source_sha256'].items():",
        b"    for name, expected_hash in candidate_source_hashes(runtime['source_sha256']).items():")
    site = b'            if any(not (build is not None and tree == source and (source / name).resolve().is_relative_to(build))'
    return once(raw, site,
        b"            if any(not ((tree == source and name == 'patches/' + Q8_PATCH)\n"
        b'                        or (build is not None and tree == source and (source / name).resolve().is_relative_to(build)))')
