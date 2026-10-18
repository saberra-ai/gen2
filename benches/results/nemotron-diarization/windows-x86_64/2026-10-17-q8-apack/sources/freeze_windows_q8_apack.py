"""Freeze bounded per-dispatch A packing against the exact production helper."""
import ast
import hashlib
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from q8_apack_source import control_source, transform
from q8_apack_experiment_source import derive_benchmark, derive_cmake, derive_metrics, derive_probe
from windows_q8_apack_metrics import ORDER, KERNEL_ORDER


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent_path = tools / 'windows-q8-mx2.protocol.json'
    if descriptor(parent_path) != {'bytes': 878017,
            'sha256': '5545fc259ba5ad4992517e5b6876cec3328e88ad9a32d839f75fe53a6a9b55cd'}:
        raise ValueError('exact packed-B-scale comparison protocol required')
    parent = json.loads(parent_path.read_bytes())
    for group, base in (('tools', tools), ('repository_files', repo)):
        for name, expected in parent[group].items():
            if descriptor(base / name) != expected:
                raise ValueError('previous frozen input changed: ' + name)
    references = set(parent['repository_files']) | {
        '.github/workflows/diarization-windows-q8-apack.yml',
        'docs/research/nemotron-windows-q8-apack.md',
        'docs/research/nemotron-windows-q8-mx2-results.md'}
    result = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-mx2'
    # These manifests and qualifications were closed and reviewed before this candidate.
    wanted = json.loads((tools / 'q8-apack-parent-identities.json').read_bytes())
    for root, names in [(result, ('collection.json', 'collection-qualification.json')),
            (result.with_name('2026-10-17-q8-mx2-loop-inspection'), ('collection.json',))]:
        for name in names:
            key = str((root / name).relative_to(repo))
            if descriptor(root / name) != wanted[key]:
                raise ValueError('retained Mx2 result identity changed: ' + key)
            references.add(key)
        if root != result:
            for name, expected in json.loads((root / 'collection.json').read_bytes())['files'].items():
                if descriptor(root / name) != expected:
                    raise ValueError('retained Mx2 static evidence changed: ' + name)
                references.add(str((root / name).relative_to(repo)))
    for target, original, derive in (
            ('windows_q8_apack_probe.py', 'windows_q8_row_probe.py', derive_probe),
            ('windows_q8_apack_metrics.py', 'windows_q8_row_metrics.py', derive_metrics),
            ('q8_apack_benchmark.cmake', 'q8_row_benchmark.cmake', derive_cmake),
            ('q8_apack_benchmark.cpp', 'q8_row_benchmark.cpp', derive_benchmark)):
        args = [(tools / original).read_bytes()]
        if target.endswith('.cpp'):
            args.append((tools / 'q8_apack_controls.cpp').read_bytes())
        if (tools / target).read_bytes() != derive(*args):
            raise ValueError('exact packing experiment derivation required: ' + target)
    names = set(parent['tools']) | {parent_path.name, 'q8_apack_benchmark.cpp',
        'q8_apack_benchmark.cmake', 'q8_apack_controls.cpp', 'q8-apack-parent-identities.json'}
    pending = ['windows_q8_apack_probe.py', 'test_windows_q8_apack.py', 'freeze_windows_q8_apack.py']
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited: continue
        visited.add(name); names.add(name)
        for node in ast.walk(ast.parse((tools / name).read_bytes())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                local = (module or '').split('.')[0] + '.py'
                if (tools / local).is_file() and local not in visited: pending.append(local)
    original = (tools / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
    protocol = {key: value for key, value in parent.items()
        if key not in ('tools', 'repository_files', 'limits', 'parent_protocol', 'control_source', 'candidate_source')}
    protocol.update(schema_version=9, parent_protocol=descriptor(parent_path), order=ORDER, kernel_order=KERNEL_ORDER,
        control_source=identity(control_source(original)), candidate_source=identity(transform(original)),
        packing_blocks=65536, packing_payload_bytes=2097152, packing_scales_bytes=262144,
        packing_worker_cap_bytes=67108864,
        tools={name: descriptor(tools / name) for name in sorted(names)},
        repository_files={name: descriptor(repo / name) for name in sorted(references)},
        limits=[
            'One bounded per-worker per-dispatch A packing comparison. Original Mx4 control and every failed prior prediction remain unchanged.',
            'Production 2x4 geometry/full-region/tails/worker ownership and ordered scale/updot/FMA/hsum operations stay exact; only A payload/scale storage changes.',
            'Copy each touched A row once per worker. Scratch at most64MiB/worker, checked before multiplication; original helper fallback on invalid/oversized scratch or allocation failure; matching aligned free by scope destructor.',
            'All packing/allocation/free remain inside every timed dispatch, with no cross-call cache. Extra memory/copy overhead may outweigh reuse; no causal bottleneck or process-RSS/reservation claim.',
            'Twelve actual same-implementation transport controls save all65536scale inputs and2097152payloadbytes each. Require padded two-row/nonzero-start source immutability, destination guards, lookup-bit equality and ten allocation-bound controls.',
            'Same22shapes,12fresh1/2/4worker off/on/on/off processes;control/apack/apack/control,onewarmup/eighttimedcalls. Original clocks/observer/ABI/compiler/ISA/contraction unchanged.',
            'Require1056completebyteexactvectors,12half tables,12packing output pairs,36support rejections,95owned/reapedcommands and all original source/compiler/dependency/object/execution bindings.',
            'Original5percent speedup/10percent drift for every12primary×4fourworker comparisons and separate5percent observer/10percent drift for every48primary/phase comparisons remain required.',
            'Numerical/observer failure withholds corresponding interpretation. No static/workflow/nonprimary/one-two-worker substitute for a failed original gate.',
            'No SDK/model/public/lifecycle/RTF/package/desktop/release qualification follows this synthesized model-free comparison. Original DER,deferred human review,freshmachine,platform,signing/main gates remain.',
        ])
    target = tools / 'windows-q8-apack.protocol.json'
    target.write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'control': protocol['control_source'],
        'candidate': protocol['candidate_source'], 'tools': len(names), 'repository_files': len(references),
        'local_import_closure': len(visited)}))


if __name__ == '__main__': main()
