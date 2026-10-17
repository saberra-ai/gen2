"""Freeze one new Q8 dispatch, admitted model geometries and all actual parents."""
import ast
import hashlib
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from pooled_runtime import BOUNDED_PROFILE, load_profile
from q8_mx4_source import ORIGINAL, PATH, transform
from q8_mx4_geometry import MODEL_RESULT, admit_geometry, derive_harness, derive_project
from windows_q8_mx4_metrics import ORDER, SHAPES, PRIMARY_SHAPES, REPETITIONS, GUARD_CODES


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent_path = tools / 'windows-q8-tile-v2.protocol.json'
    if descriptor(parent_path) != {'bytes':449295,'sha256':'e4266c71fa4ed4556f1a7d4c47d25d357f3ddc1cc52050d82ae0cdd883d636da'}:
        raise ValueError('exact preceding Q8 protocol required')
    parent = json.loads(parent_path.read_text())
    for group, root in [('tools',tools),('repository_files',repo)]:
        for name, expected in parent[group].items():
            if descriptor(root/name) != expected: raise ValueError('preceding frozen input changed: '+name)
    model_protocol = tools / 'windows-matrix-batch-profile.protocol.json'
    if descriptor(model_protocol) != {'bytes':564099,'sha256':'04e3b9a4dc3376ed577b7e3bc5df9b314b7d8c7e8350b5115394d51fe70aa225'}:
        raise ValueError('exact model geometry producer protocol required')
    references = set(parent['repository_files']) | {
        '.github/workflows/diarization-windows-q8-mx4.yml',
        'docs/research/nemotron-windows-q8-mx4.md'}
    for base, expected in [
            ('benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-tile-v2',
             {'bytes':62590,'sha256':'192908b8555ca39deffc9a703703270dff39c278fbdcfc95691c5d6ba873a2e4'}),
            (MODEL_RESULT, {'bytes':67052,'sha256':'dd44361053387a87afeff82537d93873f1753d634bb57917c2e8db8e127fa8ea'})]:
        manifest_path = repo/base/'collection.json'
        if descriptor(manifest_path) != expected: raise ValueError('exact actual parent manifest required')
        manifest = json.loads(manifest_path.read_text())['files']
        for name, value in manifest.items():
            if descriptor(repo/base/name) != value: raise ValueError('actual preceding payload changed: '+name)
        references |= {base+'/'+name for name in manifest} | {base+'/collection.json'}
    geometry = admit_geometry(json.loads((repo/MODEL_RESULT/'collection-qualification.json').read_text()))
    for target, origin, derive in [
            ('q8_mx4_benchmark.cpp','q8_tile_benchmark_v2.cpp',derive_harness),
            ('q8_mx4_benchmark.cmake','q8_tile_benchmark_v2.cmake',derive_project)]:
        if (tools/target).read_bytes() != derive((tools/origin).read_bytes()):
            raise ValueError('exact harness/project derivation required')
    fixture = tools/'fixtures/q8-tile-sgemm.cpp'
    audit = json.loads((repo/'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-math-errno/evidence-control/platform-llama-materialized-audit.json').read_text())
    if descriptor(fixture) != ORIGINAL or audit['files'][PATH] != ORIGINAL or len(audit['files']) != 3425:
        raise ValueError('exact original SGEMM and source map required')
    candidate = transform(fixture.read_bytes())
    names = set(parent['tools']) | {parent_path.name,model_protocol.name,'q8_mx4_benchmark.cpp','q8_mx4_benchmark.cmake'}
    pending = ['windows_q8_mx4_probe.py','test_windows_q8_mx4.py','freeze_windows_q8_mx4.py']
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited: continue
        visited.add(name); names.add(name)
        for node in ast.walk(ast.parse((tools/name).read_bytes())):
            imports = ([node.module] if isinstance(node,ast.ImportFrom) else
                       [item.name for item in node.names] if isinstance(node,ast.Import) else [])
            for module in imports:
                local = (module or '').split('.')[0]+'.py'
                if (tools/local).is_file() and local not in visited: pending.append(local)
    _, lock = load_profile(BOUNDED_PROFILE)
    protocol = {'schema_version':3,'parent_protocol':descriptor(parent_path),'model_protocol':descriptor(model_protocol),
        'model_geometry_admission':geometry, 'guard_codes':GUARD_CODES, 'half_table_entries':65536,
        'half_table_bytes':262144,'finite_half_entries':63488,'order':ORDER,'shapes':SHAPES,
        'primary_shapes':PRIMARY_SHAPES,'timed_calls_per_shape':REPETITIONS,
        'native_revision':lock['runtime']['revision'],'llama_revision':lock['runtime']['llama_revision'],
        'original_source':ORIGINAL,'candidate_source':{'bytes':len(candidate),'sha256':hashlib.sha256(candidate).hexdigest()},
        'tools':{name:descriptor(tools/name) for name in sorted(names)},
        'repository_files':{name:descriptor(repo/name) for name in sorted(references)},
        'limits':['Model-free Windows Q8 kernel control; no SDK/model/public/lifecycle/release qualification.',
            'Only Q8 Windows AVX2/F16C dispatch changes to existing gemmMx4<2>; original arithmetic helpers and tails remain unchanged.',
            'Original compiler/linker, contraction off, platform/Release flags and portable AVX2/FMA/F16C; candidate compile guard must be enabled.',
            'Actual bounded model profiles determine primary minimum/median/maximum distinct columns before kernel timing; synthesized values are not captured model tensor payloads.',
            'Twelve fresh kernel processes,36 exact support rejections,12 complete half tables and264 finite nonzero byte-exact vectors; padded strides, input/output guards and repeat checks remain mandatory.',
            'Original 10-percent drift and 5-percent improvement predicates apply separately to every one of12 primary shapes on four workers. Retain all22 shapes and all timings without dropping outliers.',
            'Any useful kernel still requires full SDK/model parity and production/native/public/lifecycle/performance/package qualification; original quality, human, desktop and signing gates remain.']}
    target = tools/'windows-q8-mx4.protocol.json'
    target.write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'protocol':descriptor(target),'tools':len(names),'repository_files':len(references),'local_import_closure':len(visited)}))


if __name__ == '__main__': main()
