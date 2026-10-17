"""Freeze a distinct two-row consumption kernel and complete worker comparison."""
import ast
import hashlib
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from q8_pair_source import control_source, transform
from q8_pair_experiment_source import derive_benchmark, derive_cmake, derive_metrics, derive_probe
from windows_q8_pair_metrics import ORDER, KERNEL_ORDER


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent_path = tools / 'windows-q8-worker-timing.protocol.json'
    if descriptor(parent_path) != {'bytes': 785549, 'sha256': '0eb0d37780ead508c3622cc57c4b7f0ae8ba32da3ebe317e0d46417707a26d97'}:
        raise ValueError('exact original worker timing protocol required')
    parent = json.loads(parent_path.read_text())
    for group, base in (('tools', tools), ('repository_files', repo)):
        for name, expected in parent[group].items():
            if descriptor(base / name) != expected:
                raise ValueError('original frozen worker experiment input changed: ' + name)
    references = set(parent['repository_files']) | {
        '.github/workflows/diarization-windows-q8-pair.yml', 'docs/research/nemotron-windows-q8-pair.md',
        'docs/research/nemotron-windows-q8-candidate-results.md', 'docs/research/nemotron-windows-q8-worker-timing-results.md'}
    result_root = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-candidate'
    if descriptor(result_root / 'collection.json') != {
            'bytes': 49607, 'sha256': '788eca425cfd79cbfadb89844f63fb78d8a85c1a08feb91d7e0f65f382db6e0b'}:
        raise ValueError('exact failed production collection required')
    manifest = json.loads((result_root / 'collection.json').read_text())['files']
    for name, value in manifest.items():
        if descriptor(result_root / name) != value:
            raise ValueError('actual production evidence changed: ' + name)
    references |= {str((result_root / name).relative_to(repo)) for name in manifest} | {
        str((result_root / 'collection.json').relative_to(repo))}
    for target, original, derive in (
            ('windows_q8_pair_probe.py', 'windows_q8_worker_timing_probe.py', derive_probe),
            ('windows_q8_pair_metrics.py', 'windows_q8_worker_timing_metrics.py', derive_metrics),
            ('q8_pair_benchmark.cmake', 'q8_worker_timing_benchmark.cmake', derive_cmake),
            ('q8_pair_benchmark.cpp', 'q8_worker_timing_benchmark.cpp', derive_benchmark)):
        if (tools / target).read_bytes() != derive((tools / original).read_bytes()):
            raise ValueError('exact pair experiment derivation required: ' + target)
    names = set(parent['tools']) | {parent_path.name, 'q8_pair_benchmark.cpp', 'q8_pair_benchmark.cmake'}
    pending = ['windows_q8_pair_probe.py', 'test_windows_q8_pair.py', 'freeze_windows_q8_pair.py']
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name); names.add(name)
        for node in ast.walk(ast.parse((tools / name).read_bytes())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                local = (module or '').split('.')[0] + '.py'
                if (tools / local).is_file() and local not in visited:
                    pending.append(local)
    original = (tools / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
    protocol = {key: value for key, value in parent.items()
                if key not in ('tools', 'repository_files', 'limits', 'parent_protocol')}
    protocol.update(schema_version=5, parent_protocol=descriptor(parent_path), order=ORDER, kernel_order=KERNEL_ORDER,
        control_source=identity(control_source(original)), candidate_source=identity(transform(original)),
        tools={name: descriptor(tools / name) for name in sorted(names)},
        repository_files={name: descriptor(repo / name) for name in sorted(references)},
        limits=[
            'Separate model-free 2x4 two-row consumption candidate versus the exact current production Mx4 dispatch. Production SDK and prior failed predicates remain unchanged.',
            'Only the guarded Q8 full-tile helper changes; original tile ownership, tail dispatch, A lookup/B F16C conversions, scale operand order, updot, per-output increasing block order, madd and hsum remain.',
            'Hypothesis: consuming each B vector across both A rows shortens simultaneous raw B-vector lifetimes. Static/generated instructions do not establish a measured spill cause.',
            'Same original compiler/linker/AVX2/contraction flags,22 shapes,12 fresh1/2/4-worker off/on/on/off processes; each shape runs control/pair/pair/control, one warmup and eight timed calls.',
            'Require all1056 complete vectors byte-exact,12 complete half tables,36 support rejections and all95 observed/reaped commands. Retain every failed or successful observation.',
            'Original5percent speedup and10percent drift require all12 primary shapes in every four-worker process; original5percent observer and10percent drift bounds require all48 primary/position comparisons.',
            'If numerical or observer bounds fail, withhold corresponding interpretation. Generated-loop stack operand widths are static counts, never measured traffic or a proven bottleneck.',
            'A model-free pass cannot establish SDK/model/public/lifecycle/RTF/package/desktop/release qualification. Quality, deferred human-word review, fresh-machine app and signing gates remain.'])
    target = tools / 'windows-q8-pair.protocol.json'
    target.write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'control': protocol['control_source'],
        'candidate': protocol['candidate_source'], 'tools': len(names), 'repository_files': len(references),
        'local_import_closure': len(visited)}))


if __name__ == '__main__': main()
