"""Freeze one packed-B-scale2x2 helper against exact production Mx4."""
import ast
import hashlib
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from q8_mx2_source import control_source, transform
from q8_mx2_experiment_source import derive_benchmark, derive_cmake, derive_metrics, derive_probe
from windows_q8_mx2_metrics import ORDER, KERNEL_ORDER


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
        '.github/workflows/diarization-windows-q8-mx2.yml',
        'docs/research/nemotron-windows-q8-mx2.md',
        'docs/research/nemotron-windows-q8-row-results.md',
        'docs/research/nemotron-windows-q8-row-loop-inspection.md',
        'docs/research/nemotron-windows-q8-scale-results.md',
        'docs/research/nemotron-windows-q8-tile-v2-results.md'}
    result = repo / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-row'
    for name, wanted in {
            'collection.json': {'bytes': 213899,
                'sha256': '213cd26a9737f0c7ac55a44f76114ed6a4b910a143222d0772e65a2c42b31a2a'},
            'collection-qualification.json': {'bytes': 1755889,
                'sha256': '49b79121268ba89cc456a991b86bc2caff94e54d7b057410988ca6d3f02f349d'}}.items():
        if descriptor(result / name) != wanted:
            raise ValueError('exact retained row result identity required: ' + name)
        references.add(str((result / name).relative_to(repo)))
    scale_result = result.with_name('2026-10-17-q8-scale')
    for name, wanted in {
            'collection.json': {'bytes':217686,'sha256':'30c58b5e202161214373176ff9602c67ce33482f73519fed242a7011a485f392'},
            'collection-qualification.json': {'bytes':1761168,'sha256':'bbf21c7cac61d58c5f9983035533f7a2e55bf4903225666dc87f3c35c708d323'}}.items():
        if descriptor(scale_result/name) != wanted:
            raise ValueError('exact retained A-scale result required: '+name)
        references.add(str((scale_result/name).relative_to(repo)))
    for inspection, wanted in [
            (result.with_name('2026-10-17-q8-row-loop-inspection'), {'bytes':2409,'sha256':'51dcd2e621769519d5d2d2767125621debfb74f64dc96e845828ffb7f5443d66'}),
            (result.with_name('2026-10-17-q8-scale-loop-inspection'), {'bytes':2444,'sha256':'83a47173855c03f84fc4f47cb0798045180659079f44f47da2d2847b712c5e11'})]:
        if descriptor(inspection/'collection.json') != wanted:
            raise ValueError('exact complete static receipt required')
        for name, expected in json.loads((inspection/'collection.json').read_text())['files'].items():
            if descriptor(inspection/name) != expected:
                raise ValueError('static evidence changed: '+name)
            references.add(str((inspection/name).relative_to(repo)))
        references.add(str((inspection/'collection.json').relative_to(repo)))
    for target, original, derive in (
            ('windows_q8_mx2_probe.py', 'windows_q8_row_probe.py', derive_probe),
            ('windows_q8_mx2_metrics.py', 'windows_q8_row_metrics.py', derive_metrics),
            ('q8_mx2_benchmark.cmake', 'q8_row_benchmark.cmake', derive_cmake),
            ('q8_mx2_benchmark.cpp', 'q8_row_benchmark.cpp', derive_benchmark)):
        if (tools / target).read_bytes() != derive((tools / original).read_bytes()):
            raise ValueError('exact packed-B-scale experiment derivation required: ' + target)
    names = set(parent['tools']) | {parent_path.name, 'q8_mx2_benchmark.cpp', 'q8_mx2_benchmark.cmake'}
    pending = ['windows_q8_mx2_probe.py', 'test_windows_q8_mx2.py', 'freeze_windows_q8_mx2.py']
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
    protocol.update(schema_version=8, parent_protocol=descriptor(parent_path), order=ORDER, kernel_order=KERNEL_ORDER,
        control_source=identity(control_source(original)), candidate_source=identity(transform(original)),
        tools={name: descriptor(tools / name) for name in sorted(names)},
        repository_files={name: descriptor(repo / name) for name in sorted(references)},
        limits=[
            'One fixed packed-B-scale2x2 private helper comparison against exact production2x4 Mx4. Original generic scalar-scale2x2 and every subsequent failed experiment remain failed.',
            'Only private helper name/two-column geometry/packing and omitted columns2/3 change. Fullregionm>=2n>=4,me=m/2*2,ne=n/4*4 and original tails/helpers remain exact. A lookup,ordered scale multiplication,updot,madd,hsum,ISA/compiler/contraction stay unchanged.',
            'Four source accumulators/two B vectors retain B reuse across two A rows, but repeat A work and double tiles/change ownership under original duty formula. Separate staticprediction zero explicit RSPvectoroperands in unique4FMAloop; normalize two iterations for8outputs. No traffic/CPUcause claim.',
            'Same22shapes,12fresh1/2/4worker off/on/on/off processes;control/mx2/mx2/control,onewarmup/eighttimedcalls. Clocks/support/observer/ABI untouched.',
            'Require1056completebyteexactvectors,12completehalftables,36supportrejections,95owned/reapedcommands,complete source/compiler/dependency/object bindings and every observation.',
            'Original5percent speedup/10percent drift required for all12primaries in every four-worker process; original5percent observer/10percent drift required for all48primary/position comparisons.',
            'Numerical/observer failure withholds corresponding interpretation. Staticprediction cannot replace/redefine numerical/timing/observer requirements; workflow success alone is not qualification.',
            'Prior row/A-scale result identities and complete17member static receipts are bound, not oldvector-replayed here. Row1 has0explicitstackoperands/90normalizedinstructions despite failedspeed; staticcounts alone cannot choose production.',
            'No SDK/model/public/lifecycle/RTF/package/desktop/release qualification follows a model-free pass. Original DER,deferred human review,freshmachine,platform,signing/main gates remain.',
])
    target = tools / 'windows-q8-mx2.protocol.json'
    target.write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'control': protocol['control_source'],
        'candidate': protocol['candidate_source'], 'tools': len(names), 'repository_files': len(references),
        'local_import_closure': len(visited)}))


if __name__ == '__main__': main()
