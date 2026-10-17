"""Freeze only the versioned process-ownership correction to the experiment."""
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from windows_compiler_v2_sources import derive_runner


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent = json.loads((tools / 'windows-compiler-diagnostic.protocol.json').read_text())
    for group, root in [('tools', tools), ('repository_files', repo)]:
        for name, expected in parent[group].items():
            if descriptor(root / name) != expected:
                raise ValueError('original compiler experiment changed: ' + name)
    if (tools / 'windows_compiler_diagnostic_v2.py').read_bytes() != derive_runner(
            (tools / 'windows_compiler_diagnostic.py').read_bytes()):
        raise ValueError('versioned controller exceeds exact ownership correction')
    receipt = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-job-failure'
    collection = json.loads((repo / receipt / 'collection.json').read_text())
    for name, expected in collection['files'].items():
        if descriptor(repo / receipt / name) != expected:
            raise ValueError('actual first-job failure receipt changed: ' + name)
    tool_names = set(parent['tools']) | {'windows-compiler-diagnostic.protocol.json',
        'windows_job_v2.py', 'test_windows_job_v2.py', 'windows_compiler_diagnostic_v2.py',
        'windows_compiler_v2_sources.py', 'test_windows_compiler_diagnostic_v2.py',
        'freeze_windows_compiler_diagnostic_v2.py'}
    repo_names = set(parent['repository_files']) | {receipt + '/' + name for name in collection['files']}
    repo_names |= {receipt + '/collection.json', '.github/workflows/diarization-windows-compiler-diagnostic-v2.yml',
        'docs/research/nemotron-windows-compiler-diagnostic-v2.md', 'docs/research/nemotron-windows-compiler-job-failure.md'}
    p = {'schema_version': 1, 'order': parent['order'], 'actual_pcm_sha256': parent['actual_pcm_sha256'],
         'variants': parent['variants'],
         'tools': {name: descriptor(tools / name) for name in sorted(tool_names)},
         'repository_files': {name: descriptor(repo / name) for name in sorted(repo_names)},
         'limits': parent['limits'] + [
             'Versioned process-ownership correction only; original inference/build compiler selection and fixed timing rule unchanged.',
             'Retain observed/reaped worker exit codes and a bounded membership-scoped PID/image snapshot before cleanup.',
             'Drain owned background descendants after normal exit too; require worker reaped, Job empty and handle closed before returning status.',
             'Snapshot races/query errors are recorded; no outside-Job process image query or PID-based descendant termination.']}
    target = tools / 'windows-compiler-diagnostic-v2.protocol.json'
    target.write_text(json.dumps(p, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'tools': len(p['tools']),
                      'repository_files': len(p['repository_files'])}))


if __name__ == '__main__':
    main()
