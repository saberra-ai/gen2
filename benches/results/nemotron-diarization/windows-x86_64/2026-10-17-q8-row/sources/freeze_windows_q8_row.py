"""Freeze one fixed 1x4 dispatch against the exact current production 2x4."""
import ast
import hashlib
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from q8_row_source import control_source, transform
from q8_row_experiment_source import derive_benchmark, derive_cmake, derive_metrics, derive_probe
from windows_q8_row_metrics import ORDER, KERNEL_ORDER


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent_path = tools / 'windows-q8-pair.protocol.json'
    if descriptor(parent_path) != {'bytes': 862413,
            'sha256': '865d607acc7de3d9306a7874ca657ddfec1ab894c9f50558e642cff7cbcb9959'}:
        raise ValueError('exact original paired-kernel protocol required')
    parent = json.loads(parent_path.read_text())
    for group, base in (('tools', tools), ('repository_files', repo)):
        for name, expected in parent[group].items():
            if descriptor(base / name) != expected:
                raise ValueError('original frozen experiment input changed: ' + name)
    references = set(parent['repository_files']) | {
        '.github/workflows/diarization-windows-q8-row.yml',
        'docs/research/nemotron-windows-q8-row.md',
        'docs/research/nemotron-windows-q8-pair-results.md'}
    result = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-pair'
    for name, wanted in {
            'collection.json': {'bytes': 212709,
                'sha256': 'a4ed4a36afcfeac94f4ba05b68c1bf11584eb80fd21adc21a557e1b0f5827aed'},
            'collection-qualification.json': {'bytes': 1751455,
                'sha256': '0372d72d2b4686e0de564c43af52a6b59e67c3b0feb945e153e80c5b7fb5ba5f'}}.items():
        if descriptor(result / name) != wanted:
            raise ValueError('exact retained pair result identity required: ' + name)
        references.add(str((result / name).relative_to(repo)))
    inspection = result.with_name('2026-10-17-q8-pair-loop-inspection')
    if descriptor(inspection / 'collection.json') != {'bytes': 1006,
            'sha256': 'd84d011bd9f52566fd6ecbac3688a306e7ae196b7ae94890f7c4dce042e3b05b'}:
        raise ValueError('exact static inspection collection required')
    for name, wanted in json.loads((inspection / 'collection.json').read_text())['files'].items():
        if descriptor(inspection / name) != wanted:
            raise ValueError('static inspection evidence changed: ' + name)
        references.add(str((inspection / name).relative_to(repo)))
    references.add(str((inspection / 'collection.json').relative_to(repo)))
    for target, original, derive in (
            ('windows_q8_row_probe.py', 'windows_q8_pair_probe.py', derive_probe),
            ('windows_q8_row_metrics.py', 'windows_q8_pair_metrics.py', derive_metrics),
            ('q8_row_benchmark.cmake', 'q8_pair_benchmark.cmake', derive_cmake),
            ('q8_row_benchmark.cpp', 'q8_pair_benchmark.cpp', derive_benchmark)):
        if (tools / target).read_bytes() != derive((tools / original).read_bytes()):
            raise ValueError('exact one-row experiment derivation required: ' + target)
    names = set(parent['tools']) | {parent_path.name, 'q8_row_benchmark.cpp', 'q8_row_benchmark.cmake'}
    pending = ['windows_q8_row_probe.py', 'test_windows_q8_row.py', 'freeze_windows_q8_row.py']
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
        if key not in ('tools', 'repository_files', 'limits', 'parent_protocol', 'control_source', 'candidate_source')}
    protocol.update(schema_version=6, parent_protocol=descriptor(parent_path), order=ORDER, kernel_order=KERNEL_ORDER,
        control_source=identity(control_source(original)), candidate_source=identity(transform(original)),
        tools={name: descriptor(tools / name) for name in sorted(names)},
        repository_files={name: descriptor(repo / name) for name in sorted(references)},
        limits=[
            'One fixed model-free1x4 dispatch comparison against the exact production2x4 Mx4 control. Production and every prior failed predicate remain unchanged.',
            'Only the full-region gemmMx4 template argument2-to-1 and compilation guard change; helper arithmetic, guarded types/ISA, even-row/four-column region and original tails remain exact.',
            'The candidate has four source accumulators rather than eight, but reloads B across rows and doubles full-region tile count; worker ownership changes with the unchanged duty formula. No spill or speed cause is established.',
            'Same original compiler/linker/AVX2/contraction flags,22 shapes,12 fresh1/2/4-worker off/on/on/off processes; control/row1/row1/control, one warmup and eight timed calls.',
            'Require all1056 complete vectors byte-exact,12 complete half tables,36 support rejections and all95 observed/reaped commands; retain every observation.',
            'Original5percent speedup and10percent drift require all12 primary shapes in each four-worker process; original5percent observer and10percent drift require all48 primary/position comparisons.',
            'Withhold corresponding interpretation if numerical or observer gates fail. Static instruction/stack counts are not measured traffic or a proven bottleneck.',
            'Prior pair result/manifest identities and complete static receipt are bound; this producer does not replay the prior pair vectors or replace their failed result.',
            'No SDK/model/public/lifecycle/RTF/package/desktop/release qualification follows a model-free pass. Original quality, deferred human review, fresh-machine and signing gates remain.'])
    target = tools / 'windows-q8-row.protocol.json'
    target.write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'control': protocol['control_source'],
        'candidate': protocol['candidate_source'], 'tools': len(names), 'repository_files': len(references),
        'local_import_closure': len(visited)}))


if __name__ == '__main__': main()
