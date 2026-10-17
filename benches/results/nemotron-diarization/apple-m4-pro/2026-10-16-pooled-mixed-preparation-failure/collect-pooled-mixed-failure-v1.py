"""Collect the pre-compilation mixed failure; never execute helpers/models."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

repo = Path('/Users/victor/workspace/gen2-nemotron3-n0')
cache = Path('/Users/victor/.cache/gen2-nemotron3-n0')
sys.path.insert(0, str(repo / 'tools/nemotron-diarization'))
from build_valid_mel_candidate import descriptor
from frozen_git_sources import verify_git_sources
root = cache / 'pooled-mixed-local-v1'
q = json.loads((root / 'qualification.json').read_text())
p = json.loads((root / 'protocol.json').read_text())
source = 'cc1c644b02520156fd8b8e6a989294acf6b77fe6'
assert q['source_commit'] == source and not q['pooled_cpu_mixed_residency_passed']
assert q['neural_commands_started'] == [] and q['error'] == "KeyError('assets/bark/nemo128.f32')"
assert len(q['commands']) == 1 and q['commands'][0]['name'] == 'compile-controls' and q['commands'][0]['returncode'] == 0
assert descriptor(root / 'protocol.json') == q['protocol']
assert (root / 'protocol.json').read_bytes() == subprocess.check_output(['git', 'show',
    source + ':tools/nemotron-diarization/macos-pooled-mixed.protocol.json'], cwd=repo)
pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
pins.update(p['repository_files'])
pins.update({'vendor/llama-cpp-sys-2/llama.cpp/' + n: v for n, v in p['vendor_sources'].items()})
verify_git_sources(repo, source, pins)
assert 'assets/bark/nemo128.f32' not in p['repository_files']
assert q['storage']['released']
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-mixed-preparation-failure'
assert not out.exists()
out.mkdir()
for path in root.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb'}:
        shutil.copy2(path, out / path.name)
for name in ['pooled-mixed-local-v1.controller.log', 'pooled-mixed-after-long-v1.json',
             'pooled-mixed-after-long-v1.log', 'run-pooled-mixed-after-long-v1.py']:
    shutil.copy2(cache / name, out / name)
shutil.copy2(__file__, out / Path(__file__).name)
(out / 'collection-qualification.json').write_text(json.dumps({'schema_version': 1,
    'source_commit': source, 'mixed_qualification_passed': False, 'preparation_failure_preserved': True,
    'declared_source_pins_verified_at_original_commit': len(pins),
    'mandatory_source_pin_omitted': 'assets/bark/nemo128.f32',
    'actual_commands': [{'name': 'compile-controls', 'returncode': 0}],
    'rust_compiler_commands_started': 0, 'native_version_queries_started': 0, 'neural_commands_started': [],
    'release_qualified': False,
    'limits': ['This is an actual pre-compilation protocol defect. Original mandatory source declaration is missing; neither source completeness nor mixed runtime success is claimed.',
        'Only model/network/read-only compilation controls ran. No Rust compiler, native version query, mixed neural test or app execution occurred. Failure remains failed; all original outputs and source pins are retained.',
        'Any corrected attempt requires a separate source/protocol freeze and fresh output, with the same assets, tests and acceptance rules.']}, indent=2) + '\n')
files = {path.relative_to(out).as_posix(): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, 'members', len(files) + 1)
