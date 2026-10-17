"""Freeze the compiler comparison without modifying previous producer inputs."""
import json
from pathlib import Path

from build_valid_mel_candidate import descriptor
from windows_compiler_metrics import ORDER
from windows_worker_metrics import PCM_SHA256


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    parent_path = tools / 'windows-session-profile.protocol.json'
    parent = json.loads(parent_path.read_text())
    for group, root in [('tools', tools), ('repository_files', repo)]:
        for name, expected in parent[group].items():
            if descriptor(root / name) != expected:
                raise ValueError('retained diagnostic source differs: ' + name)
    worker = json.loads((tools / 'windows-worker-diagnostic-v2.protocol.json').read_text())
    receipt = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-session-profile'
    collection = json.loads((repo / receipt / 'collection.json').read_text())
    for name, expected in collection['files'].items():
        if descriptor(repo / receipt / name) != expected:
            raise ValueError('session result receipt differs: ' + name)
    tool_names = set(parent['tools']) | {'windows-session-profile.protocol.json',
        'windows_compiler_diagnostic.py', 'windows_compiler_metrics.py',
        'windows_job.py', 'test_windows_job.py',
        'test_windows_compiler_diagnostic.py', 'freeze_windows_compiler_diagnostic.py'}
    repo_names = set(parent['repository_files']) | {receipt + '/' + name for name in collection['files']}
    repo_names |= {receipt + '/collection.json', '.github/workflows/diarization-windows-compiler-diagnostic.yml',
        'docs/research/nemotron-windows-compiler-diagnostic.md', 'docs/research/nemotron-windows-session-results.md'}
    p = {'schema_version': 1, 'order': ORDER, 'actual_pcm_sha256': PCM_SHA256,
         'variants': {'4': worker['variants']['4']},
         'tools': {name: descriptor(tools / name) for name in sorted(tool_names)},
         'repository_files': {name: descriptor(repo / name) for name in sorted(repo_names)},
         'limits': ['Diagnostic native compiler/toolchain comparison only; no supported release/public/lifecycle/package qualification.',
             'Identical admitted cap4 diagnostic inference sources and MSVC-built probe source/compiler; only SDK compiler selection differs.',
             'Fresh MSVC/clang-cl SDKs; fixed MSVC,clang-cl,clang-cl,MSVC fresh-process order on one host.',
             'Portable AVX2/FMA/F16C baseline, four-worker policy, original geometry/PCM/model/full-output1e-6 limits unchanged.',
             'Keep individual timings and control drift; no averaging, new RTF tolerance, cross-host timing promotion or release gate changes.',
             'Every child runs in a gated Windows kill-on-close Job; terminal success requires the entire process tree reaped.',
             '1800s SDK-build timeout;600s other-child timeout. Ambient compiler overrides,NEMO_SPEECH_TIMING,GH_TOKEN,GITHUB_TOKEN removed from child mappings.']}
    target = tools / 'windows-compiler-diagnostic.protocol.json'
    target.write_text(json.dumps(p, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'tools': len(p['tools']),
                      'repository_files': len(p['repository_files'])}))


if __name__ == '__main__':
    main()
