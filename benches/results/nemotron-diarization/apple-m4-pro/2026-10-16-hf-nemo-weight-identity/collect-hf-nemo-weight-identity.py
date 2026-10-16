"""Collect static tensor identity evidence; no neural inference or conversion."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
source = cache / 'research/hf-nemo-weight-identity'
output = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-hf-nemo-weight-identity'
def record(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
def read(path):
    return json.loads(path.read_text())
manifest = read(source / 'manifest.json')
for row in manifest['files']:
    assert Path(row['name']).name == row['name']
    assert record(source / row['name']) == {k: row[k] for k in ['bytes', 'sha256']}
report = read(source / 'report.json')
assert report['audit_passed'] is True
assert [report[k] for k in ['source_count', 'qkv_source_count', 'qkv_output_count', 'excluded_source_count', 'mapped_target_count', 'exact_matched_count']] == [363, 31, 93, 8, 417, 417]
assert len(report['all_source_inventory']) == 363
assert report['source_dtype_counts'] == {'torch.bfloat16': 363}
assert len(report['mapped_tensor_comparisons']) == 417
assert len({x['name'] for x in report['mapped_tensor_comparisons']}) == 417
assert all(x['exact_bytes_match'] and x['dtype'] == 'F32' and x['actual_hf_sha256'] == x['expected_bf16_cast_sha256'] for x in report['mapped_tensor_comparisons'])
assert report['torch_load_weights_only'] is True and report['transformers_imported'] is False
assert record(repo / 'tools/nemotron-diarization/runtime.lock.json')['sha256'] == report['runtime_lock_sha256']
assert record(source / 'audit.py')['sha256'] == report['script_sha256']
assert record(source / 'pinned-hf-converter.py')['sha256'] == report['mapping']['sha256']
assert record(source / 'extracted-mapping.ast.txt')['sha256'] == report['mapping']['extracted_ast_sha256']
for item in ['checkpoint', 'hf']:
    data = report[item]
    assert record(Path(data['path'])) == {k: data[k] for k in ['bytes', 'sha256']}
addendum = read(source / 'native-unused-head-audit.json')
native = cache / 'NeMo-Speech.cpp'
for name, expected in addendum['sources'].items():
    assert record(native / name) == expected
    blob = subprocess.check_output(['git', 'show', addendum['native_revision'] + ':' + name], cwd=native)
    assert {'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()} == expected
accounting = addendum['construction_accounting']
assert record(Path(accounting['path'])) == {k: accounting[k] for k in ['bytes', 'sha256']}
by_source = {x['name']: x for x in report['all_source_inventory']}
assert len(addendum['stored_activity_head_rows']) == 4
for row in addendum['stored_activity_head_rows']:
    original = by_source[row['source']]
    assert row['source_sha256'] == original['sha256'] and row['source_shape'] == original['shape']
    assert row['exact_source_mapping_cast_match'] is True
    assert row['output_sha256'] == row['expected_cast_sha256']
assert addendum['activity_heads_unused_by_pinned_sortformer_graph'] is True
assert not output.exists()
output.mkdir()
for path in source.iterdir():
    assert path.is_file()
    shutil.copy2(path, output / path.name)
shutil.copy2(Path(__file__), output / Path(__file__).name)
qualification = {'schema_version': 1, 'scope': 'Static publisher checkpoint-parameter identity and source-backed native unused-head addendum', 'hf_tensors_exactly_matched': 417, 'source_bf16_tensors': 363, 'hf_excluded_source_tensors': 8, 'native_stored_activity_head_tensors_unused_by_pinned_graph': 4, 'neural_model_instantiation_or_inference': False, 'native_conversion_performed': False, 'native_artifact_reread_by_weight_audit': False, 'collector_rehashed_both_publisher_files': True, 'collector_verified_all_source_blobs_and_construction_metadata': True, 'frontend_or_arithmetic_or_cache_equivalence_claimed': False, 'original_gate_status_changed': False, 'release_qualified': False, 'limits': ['Exact mapping/cast byte identity closes parameter identity only; it does not prove matching frontend buffers, F16 convolution, runtime arithmetic, framing, cache state or output probabilities.', 'Four native activity-head tensors are stored but source-unused by the pinned graph. The historical construction accounting remains unchanged.', 'Full source tensor bytes and both publisher artifacts remain outside Git.']}
(output / 'qualification.json').write_text(json.dumps(qualification, indent=2) + '\n')
files = {p.relative_to(output).as_posix(): record(p) for p in output.rglob('*') if p.is_file()}
(output / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(json.dumps({'files': len(files), 'hf_exact': 417, 'native_stored_unused': 4, 'inference': False}))
