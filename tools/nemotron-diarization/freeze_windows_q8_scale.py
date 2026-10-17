"""Freeze one A-scale conversion change against current production Mx4."""
import ast
import hashlib
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from q8_scale_source import control_source, transform
from q8_scale_experiment_source import derive_benchmark, derive_cmake, derive_metrics, derive_probe
from windows_q8_scale_metrics import ORDER, KERNEL_ORDER


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent_path = tools / 'windows-q8-row.protocol.json'
    if descriptor(parent_path) != {'bytes': 866554,
            'sha256': '8bab9e5a60b3a1da49de1fadb4cb7050ca7eccab6a8827984c741c558af2152e'}:
        raise ValueError('exact original one-row protocol required')
    parent = json.loads(parent_path.read_text())
    for group, base in (('tools', tools), ('repository_files', repo)):
        for name, expected in parent[group].items():
            if descriptor(base / name) != expected:
                raise ValueError('original frozen experiment input changed: ' + name)
    references = set(parent['repository_files']) | {
        '.github/workflows/diarization-windows-q8-scale.yml',
        'docs/research/nemotron-windows-q8-scale.md',
        'docs/research/nemotron-windows-q8-row-results.md'}
    result = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-row'
    for name, wanted in {
            'collection.json': {'bytes': 213899,
                'sha256': '213cd26a9737f0c7ac55a44f76114ed6a4b910a143222d0772e65a2c42b31a2a'},
            'collection-qualification.json': {'bytes': 1755889,
                'sha256': '49b79121268ba89cc456a991b86bc2caff94e54d7b057410988ca6d3f02f349d'}}.items():
        if descriptor(result / name) != wanted:
            raise ValueError('exact retained row result identity required: ' + name)
        references.add(str((result / name).relative_to(repo)))
    for target, original, derive in (
            ('windows_q8_scale_probe.py', 'windows_q8_row_probe.py', derive_probe),
            ('windows_q8_scale_metrics.py', 'windows_q8_row_metrics.py', derive_metrics),
            ('q8_scale_benchmark.cmake', 'q8_row_benchmark.cmake', derive_cmake),
            ('q8_scale_benchmark.cpp', 'q8_row_benchmark.cpp', derive_benchmark)):
        if (tools / target).read_bytes() != derive((tools / original).read_bytes()):
            raise ValueError('exact A-scale experiment derivation required: ' + target)
    names = set(parent['tools']) | {parent_path.name, 'q8_scale_benchmark.cpp', 'q8_scale_benchmark.cmake'}
    pending = ['windows_q8_scale_probe.py', 'test_windows_q8_scale.py', 'freeze_windows_q8_scale.py']
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
    protocol.update(schema_version=7, parent_protocol=descriptor(parent_path), order=ORDER, kernel_order=KERNEL_ORDER,
        control_source=identity(control_source(original)), candidate_source=identity(transform(original)),
        scale_half_entries=65536, scale_half_lanes=4, scale_half_bytes=1048576,
        tools={name: descriptor(tools / name) for name in sorted(names)},
        repository_files={name: descriptor(repo / name) for name in sorted(references)},
        limits=[
            'One fixed model-free A-scale lookup-to-F16C comparison against exact production2x4 Mx4. Production and all prior failed predicates remain unchanged.',
            'A private clone of the canonical Mx4 helper changes only its name and A-scale broadcast expression. Original helpers,2x4 geometry,ownership,tails,ordered scale multiplication,updot,madd,hsum and compiler/ISA/contraction stay exact.',
            'The actual candidate broadcast helper is probed for all65536 half inputs and four lanes in each of twelve processes; retain full outputs. Independent reference requires63488 finite values and both infinities bit-exact, all NaNs remain NaNs and all four lanes byte-identical. No NaN payload equivalence claim.',
            'Same22 shapes,12 fresh1/2/4-worker off/on/on/off processes; control/scale/scale/control, one warmup and eight timed calls. No clock or worker observer placement changes.',
            'Require all1056 complete vectors byte-exact,12 original half tables,12 complete four-lane scale tables,36 support rejections and all95 observed/reaped commands. Retain every failed observation.',
            'Original5percent speedup and10percent drift apply to every primary in all four-worker processes; original5percent observer and10percent drift apply to all48 primary/position comparisons.',
            'Withhold corresponding interpretation on numerical/observer failure. Source lookup removal, generated instruction/stack counts and timing observations do not establish measured traffic or a causal bottleneck.',
            'Prior row result and manifest identities are bound, not replayed by this producer; original failure remains failed.',
            'No SDK/model/public/lifecycle/RTF/package/desktop/release qualification follows a model-free pass. Original DER,deferred human review,fresh-machine,platform and signing/main gates stay open.'])
    target = tools / 'windows-q8-scale.protocol.json'
    target.write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'control': protocol['control_source'],
        'candidate': protocol['candidate_source'], 'tools': len(names), 'repository_files': len(references),
        'local_import_closure': len(visited)}))


if __name__ == '__main__': main()
