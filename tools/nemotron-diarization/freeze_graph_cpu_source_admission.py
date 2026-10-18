"""Bind a source-only Linux check to the committed graph diagnostic and process helpers."""
import json
from pathlib import Path
import subprocess

from graph_cpu_source_admission import NATIVE_SOURCE, NATIVE_PROTOCOL, PROCESS_HELPER, START_FLOOR, admit_native, descriptor


def main():
    tools = Path(__file__).resolve().parent
    repo = tools.parents[1]
    native_path = tools / 'windows-graph-cpu-profile.protocol.json'
    native = admit_native(native_path.read_bytes())
    for group, root in [('tools', tools), ('repository_files', repo)]:
        for name, wanted in native[group].items():
            if descriptor(root / name) != wanted:
                raise ValueError('native preparation changed: ' + name)
    for name in ['remote_q8_apack_replay.py', 'remote_thread_cpu_replay.py',
                 'test_remote_q8_apack_replay.py', 'test_remote_thread_cpu_replay.py']:
        raw = (tools / name).read_bytes()
        committed = subprocess.check_output(['git', 'show',
            NATIVE_SOURCE + ':tools/nemotron-diarization/' + name], cwd=repo)
        if raw != committed:
            raise ValueError('existing process helper/control changed: ' + name)
    if descriptor(tools / 'remote_q8_apack_replay.py') != PROCESS_HELPER:
        raise ValueError('original owned-process helper required')
    names = ['graph_cpu_source_admission.py', 'test_graph_cpu_source_admission.py',
             'freeze_graph_cpu_source_admission.py', 'remote_q8_apack_replay.py',
             'remote_thread_cpu_replay.py', 'test_remote_q8_apack_replay.py',
             'test_remote_thread_cpu_replay.py']
    files = ['tools/nemotron-diarization/' + name for name in names] + [
        '.github/workflows/diarization-graph-cpu-source.yml',
        'docs/research/nemotron-graph-cpu-source-admission.md']
    value = {'schema_version': 1, 'native_source': NATIVE_SOURCE,
        'native_protocol': NATIVE_PROTOCOL, 'minimum_free_bytes': START_FLOOR,
        'expected_acquisition_commands': 8, 'expected_composition_commands': 11,
        'expected_materialized_files': 3425, 'compiler_sdk_model_execution': False,
        'files': {name: descriptor(repo / name) for name in sorted(files)},
        'limits': ['Only source acquisition, exact patch composition and CMake script-mode materialization.',
                   'Original 8 GiB source-only floor before acquisition and every registered command; no native SDK/model build, load or inference.',
                   'One bounded outer POSIX group owns all stage children, with original inherited-child I/O and deadlines.',
                   'All native protocol inputs remain bound before project imports and each stage operation.',
                   'Source proof alone supplies no Windows graph/API, compiler, parity, observer timing, performance or release pass.']}
    target = tools / 'graph-cpu-source-admission.protocol.json'
    target.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'protocol': descriptor(target), 'files': len(files), 'compiled_sdk_or_model': False}))


if __name__ == '__main__':
    main()
