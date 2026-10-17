"""Freeze the existing kernels and a separate same-process worker observer."""
import ast
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from q8_worker_timing_source import derive_probe
from windows_q8_worker_timing_metrics import ORDER, KERNEL_ORDER


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent_path = tools/'windows-q8-mx4.protocol.json'
    if descriptor(parent_path) != {'bytes': 649877, 'sha256': '593d4fe427d82813fd9108247e9aae51a719b72a47159b6854b05641faad9d44'}:
        raise ValueError('exact captured-geometry parent protocol required')
    parent = json.loads(parent_path.read_text())
    for group, base in [('tools', tools), ('repository_files', repo)]:
        for name, expected in parent[group].items():
            if descriptor(base/name) != expected:
                raise ValueError('preceding frozen input changed: '+name)
    references = set(parent['repository_files']) | {
        '.github/workflows/diarization-windows-q8-worker-timing.yml',
        'docs/research/nemotron-windows-q8-worker-timing.md',
        'docs/research/nemotron-windows-q8-mx4-results.md',
        'docs/research/nemotron-windows-q8-loop-inspection.md'}
    for name, expected in [
        ('2026-10-17-q8-mx4', {'bytes': 90509, 'sha256': '04b1990d3e6b92fa88d9ef0a59546465514d484c49300be6dbe66310ddeeb9db'}),
        ('2026-10-17-q8-loop-inspection', {'bytes': 179, 'sha256': 'eea736246bcc60c53d6662bb85f81650b0c6871c690f7117eb2b09dc1a018f4c'})]:
        base = 'benches/results/nemotron-diarization/windows-x86_64/'+name
        path = repo/base/'collection.json'
        if descriptor(path) != expected:
            raise ValueError('exact preceding actual collection required')
        manifest = json.loads(path.read_text())['files']
        for relative, value in manifest.items():
            if descriptor(repo/base/relative) != value:
                raise ValueError('preceding actual payload changed: '+relative)
        references |= {base+'/'+relative for relative in manifest} | {base+'/collection.json'}
    if (tools/'windows_q8_worker_timing_probe.py').read_bytes() != derive_probe((tools/'windows_q8_mx4_probe.py').read_bytes()):
        raise ValueError('exact construction-preserving producer derivation required')
    if (tools/'q8_worker_timing_benchmark.cmake').read_bytes() != (tools/'q8_mx4_benchmark.cmake').read_bytes().replace(
            b'project(gen2_q8_mx4_control', b'project(gen2_q8_worker_timing_control'):
        raise ValueError('only project name may differ in CMake inputs')
    names = set(parent['tools']) | {parent_path.name, 'q8_worker_timing_benchmark.cpp', 'q8_worker_timing_benchmark.cmake'}
    pending = ['windows_q8_worker_timing_probe.py', 'test_windows_q8_worker_timing.py', 'freeze_windows_q8_worker_timing.py']
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        names.add(name)
        for node in ast.walk(ast.parse((tools/name).read_bytes())):
            imports = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            for module in imports:
                local = (module or '').split('.')[0]+'.py'
                if (tools/local).is_file() and local not in visited:
                    pending.append(local)
    protocol = {key: value for key, value in parent.items() if key not in ('tools', 'repository_files', 'limits', 'parent_protocol')}
    protocol.update(schema_version=4, parent_protocol=descriptor(parent_path), order=ORDER, kernel_order=KERNEL_ORDER,
        worker_observer='Two worker steady-clock reads per enabled kernel call; disabled mode performs no worker clock reads.',
        tools={name: descriptor(tools/name) for name in sorted(names)},
        repository_files={name: descriptor(repo/name) for name in sorted(references)},
        limits=[
            'Same two prior kernels, exact original compiler/linker/flags/contraction and source/ABI/ISA admissions; no new arithmetic or production candidate.',
            'Twelve fresh worker1/2/4 off/on/on/off processes; every shape runs control/mx4/mx4/control within one persistent pool, one warmup and eight timed calls per phase.',
            'All22 ordered shapes,1056 full byte-exact vectors,12 complete half tables and36 exact support rejections; preserve all95 commands and owned cleanup evidence.',
            'Whole-dispatch and enabled per-worker start/end timestamps are wall time including preemption and observer costs; no CPU-time or causal claim.',
            'Original10percent control-drift and5percent speedup predicates apply to every primary in all4 four-worker processes; all48 primary/position observer pairs require10percent control drift and at-most5percent overhead.',
            'Keep every raw observation and failed predicate; a failed observer bound withholds bounded interpretation; original failed kernel results stay failed.',
            'No SDK/model/public/lifecycle/RTF/production/package/desktop/release qualification. Quality, deferred human review, fresh-machine and signing gates remain.'])
    target = tools/'windows-q8-worker-timing.protocol.json'
    target.write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({'protocol': descriptor(target), 'tools': len(names), 'repository_files': len(references), 'local_import_closure': len(visited)}))


if __name__ == '__main__': main()
