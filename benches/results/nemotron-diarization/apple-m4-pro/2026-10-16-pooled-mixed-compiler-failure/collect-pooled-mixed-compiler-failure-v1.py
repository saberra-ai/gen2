"""Collect the mixed Rust-build tool-resolution failure; never execute helpers/models."""
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
root = cache / 'pooled-mixed-local-v2'
q = json.loads((root / 'qualification.json').read_text())
p = json.loads((root / 'protocol.json').read_text())
source = '36d39aa0ec98c143d517ede4fd242687f736bbc8'
assert q['source_commit'] == source and not q['pooled_cpu_mixed_residency_passed']
assert q['neural_commands_started'] == [] and q['error'] == "ValueError('mixed command failed: mixed-helper-build')"
assert [(x['name'],x['returncode']) for x in q['commands']] == [('compile-controls',0),('mel-build-input-readable',0),('rustc-version',0),('mixed-helper-build',101)]
assert 'is `cmake` not installed?' in (root/'mixed-helper-build.stderr.log').read_text()
assert descriptor(root / 'protocol.json') == q['protocol']
assert (root / 'protocol.json').read_bytes() == subprocess.check_output(['git', 'show',
    source + ':tools/nemotron-diarization/macos-pooled-mixed-v2.protocol.json'], cwd=repo)
pins = {'tools/nemotron-diarization/' + n: v for n, v in p['tools'].items()}
pins.update(p['repository_files'])
verify_git_sources(repo, source, pins)
verify_git_sources(repo / 'vendor/llama-cpp-sys-2/llama.cpp', p['vendor_revision'], p['vendor_sources'])
assert 'assets/bark/nemo128.f32' in p['repository_files']
assert q['storage']['released']
out = repo / 'benches/results/nemotron-diarization/apple-m4-pro/2026-10-16-pooled-mixed-compiler-failure'
assert not out.exists()
out.mkdir()
for path in root.iterdir():
    if path.is_file() and path.suffix in {'.json', '.log', '.sb'}:
        shutil.copy2(path, out / path.name)
shutil.copy2(cache / 'pooled-mixed-local-v2.controller.log', out / 'controller.log')
shutil.copy2(__file__, out / Path(__file__).name)
preflight=cache/'pooled-mixed-build-tools-preflight-v1'
for path in preflight.rglob('*'):
    if path.is_file() and path.suffix in {'.json','.log','.sb','.c','.cpp'} or path.name=='CMakeLists.txt':
        destination=out/'corrected-toolchain-preflight'/path.relative_to(preflight)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
(out / 'collection-qualification.json').write_text(json.dumps({'schema_version': 1,
    'source_commit': source, 'mixed_qualification_passed': False, 'compiler_failure_preserved': True,
    'declared_parent_source_pins_verified_at_original_commit': len(pins),
    'declared_vendor_source_pins_verified_at_vendor_commit': len(p['vendor_sources']),
    'vendor_revision': p['vendor_revision'],
    'source_admission': 'Parent and vendor sources verified in their respective Git repositories at the declared revisions.',
    'compiler_tool_resolution_failure': 'Restricted PATH omits Homebrew CMake; CMake process never started.',
    'actual_commands': [{'name': x['name'], 'returncode': x['returncode']} for x in q['commands']],
    'rust_helper_build_commands_started': 1, 'native_version_queries_started': 0, 'neural_commands_started': [],
    'release_qualified': False,
    'limits': ['Actual compiler failure before CMake started. Rust helper build exited101; no native version query, neural or app command ran.',
        'Original V2 protocol and outputs remain failed. Corrected tool discovery requires a new source/protocol freeze and fresh output with unchanged assets, tests and acceptance rules.']}, indent=2) + '\n')
files = {path.relative_to(out).as_posix(): descriptor(path) for path in out.rglob('*') if path.is_file()}
(out / 'collection.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(out, 'members', len(files) + 1)
