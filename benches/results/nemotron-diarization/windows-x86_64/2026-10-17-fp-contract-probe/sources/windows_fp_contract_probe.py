"""Small actual Windows compiler observation; no SDK build or speech model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from frozen_git_sources import verify_git_sources
from platform_native import msvc_tool
from qualification_storage import atomic_json
from windows_job_v2 import run_in_job

VARIANTS = ('msvc-default', 'clang-default', 'clang-no-contract')
COMMON_FLAGS = ['/nologo', '/O2', '/Ob2', '/DNDEBUG', '/std:c++17', '/MD', '/arch:AVX2', '/EHsc']
INPUTS = [0x3f800001, 0x3f7ffffe, 0xbf800000]
FUSED = 0xa8800000
REFERENCE = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-compiler-parity-failure/collection-qualification.json'
HOST = '[ordered]@{avx2=[System.Runtime.Intrinsics.X86.Avx2]::IsSupported;fma=[System.Runtime.Intrinsics.X86.Fma]::IsSupported;cpu=@(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)} | ConvertTo-Json -Depth 4 -Compress'
BOOTSTRAP = 'import json,os;from platform_native import msvc_environment;print(json.dumps(msvc_environment(dict(os.environ))))'


def descriptor(path):
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def environment(values):
    forbidden = {'CL', '_CL_', 'CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS',
                 'GH_TOKEN', 'GITHUB_TOKEN', 'NEMO_SPEECH_TIMING'}
    return {key: value for key, value in values.items() if key.upper() not in forbidden}


def compile_command(compiler, source, directory, variant):
    if variant not in VARIANTS:
        raise ValueError('explicit arithmetic observation variant required')
    flags = COMMON_FLAGS + (['/clang:-ffp-contract=off'] if variant == 'clang-no-contract' else [])
    return [str(compiler), *flags, str(source), '/Fo' + str(directory / 'probe.obj'),
            '/Fe' + str(directory / 'probe.exe')]


def observation(text, variant):
    if variant not in VARIANTS:
        raise ValueError('explicit arithmetic observation variant required')
    value = json.loads(text)
    fields = {'inputs', 'ordinary', 'separate', 'intrinsic', 'mxcsr', 'msc_ver',
              'clang_major', 'fma_macro', 'avx2_macro'}
    if (not isinstance(value, dict) or set(value) != fields
            or value['inputs'] != INPUTS
            or any(type(item) is not int for item in value['inputs'])
            or any(type(value[key]) is not int for key in fields - {'inputs'})
            or value['separate'] != 0 or value['intrinsic'] != FUSED
            or value['ordinary'] not in (0, FUSED)
            or not 0 <= value['mxcsr'] <= 0xffff
            or value['msc_ver'] <= 0 or value['avx2_macro'] != 1
            or value['fma_macro'] not in (0, 1)
            or (variant == 'msvc-default' and value['clang_major'] != 0)
            or (variant != 'msvc-default' and value['clang_major'] != 19)
            or (variant == 'clang-no-contract' and value['ordinary'] != 0)):
        raise ValueError('invalid actual arithmetic control observation')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64', 'x86_64'):
        parser.error('actual Windows x64 required')
    tools = Path(__file__).resolve().parent
    repo, output = tools.parents[1], args.output.resolve()
    if output.exists() or output.is_symlink() or output.is_relative_to(repo) or repo.is_relative_to(output):
        parser.error('fresh output disjoint from repository required')
    protocol_path = tools / 'windows-fp-contract-probe.protocol.json'
    protocol = protocol_path.read_bytes()
    p = json.loads(protocol)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    expected = {**{'tools/nemotron-diarization/' + name: value for name, value in p['tools'].items()},
                **p['repository_files']}

    def frozen():
        if protocol_path.read_bytes() != protocol:
            raise ValueError('arithmetic observation protocol changed')
        for name, value in expected.items():
            if descriptor(repo / name) != value:
                raise ValueError('arithmetic observation source changed: ' + name)

    frozen()
    verify_git_sources(repo, commit, expected)
    if subprocess.check_output(['git', 'show', commit + ':tools/nemotron-diarization/' + protocol_path.name],
                               cwd=repo, env={**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}) != protocol:
        raise ValueError('committed arithmetic observation protocol required')
    output.mkdir()
    (output / 'protocol.json').write_bytes(protocol)
    q = {'schema_version': 1, 'source_commit': commit, 'protocol': descriptor(protocol_path),
         'probe_completed': False, 'commands': [], 'variants': {}, 'model_or_sdk_calls': 0,
         'release_qualified': False, 'package_qualified': False, 'public_or_lifecycle_qualified': False,
         'limits': ['Scalar compiler observation only; it cannot establish the cause of speech-model divergence or performance.']}
    persist = lambda: atomic_json(output / 'qualification.json', q)
    persist()

    def run(name, command, env, private=False, cwd=output):
        frozen()
        record = {'name': name, 'command': list(map(str, command)), 'process_tree_reaped': False}
        q['commands'].append(record)
        persist()
        try:
            with (tempfile.TemporaryFile() if private else (output / (name + '.stdout.log')).open('xb')) as out, \
                 (tempfile.TemporaryFile() if private else (output / (name + '.stderr.log')).open('xb')) as err:
                result = run_in_job(record['command'], cwd=cwd, env=env, stdout=out, stderr=err, timeout=60)
                record.update(returncode=result.returncode, process_tree_reaped=result.process_tree_reaped,
                              job_observation=result.job_observation)
                result.check_returncode()
                if private:
                    out.seek(0)
                    data = out.read(4 * 1024**2 + 1)
                    if len(data) > 4 * 1024**2:
                        raise ValueError('private compiler environment exceeds bound')
                    return data
        except BaseException as error:
            record.update(error=repr(error), process_tree_reaped=getattr(error, 'process_tree_reaped', record['process_tree_reaped']),
                          job_observation=getattr(error, 'job_observation', record.get('job_observation')))
            raise
        finally:
            persist()

    try:
        original = environment(dict(os.environ))
        run('host', ['pwsh', '-NoProfile', '-NonInteractive', '-Command', HOST], original)
        q['host'] = json.loads((output / 'host.stdout.log').read_text(encoding='utf-8-sig'))
        if q['host']['avx2'] is not True or q['host']['fma'] is not True:
            raise ValueError('actual AVX2/FMA host required before executing probe')
        private = run('compiler-environment', [sys.executable, '-B', '-c', BOOTSTRAP], original, private=True, cwd=tools)
        env = environment(json.loads(private))
        reference = json.loads((repo / REFERENCE).read_text())
        compilers = {'msvc': msvc_tool(env, 'cl.exe'), 'clang-cl': msvc_tool(env, 'clang-cl.exe')}
        dumpbin = msvc_tool(env, 'dumpbin.exe')
        q['dumpbin'] = {'path': str(dumpbin), **descriptor(dumpbin)}
        q['compilers'] = {kind: {'path': str(path), **descriptor(path)} for kind, path in compilers.items()}
        for kind, path in compilers.items():
            if descriptor(path) != reference['builds'][kind]['compiler']['compiler_binary']:
                raise ValueError('exact compiler binary from failed diagnostic required')
        for variant in VARIANTS:
            frozen()
            kind = 'msvc' if variant == 'msvc-default' else 'clang-cl'
            compiler = compilers[kind]
            directory = output / variant
            directory.mkdir()
            command = compile_command(compiler, tools / 'fp_contract_probe.cpp', directory, variant)
            if kind == 'clang-cl':
                run(variant + '-driver', [*command, '-###'], env)
            run(variant + '-build', command, env)
            run(variant + '-disassembly', [dumpbin, '/nologo', '/disasm', '/symbols', directory / 'probe.obj'], env)
            observed = []
            binaries = {name: descriptor(directory / name) for name in ('probe.exe', 'probe.obj')}
            for repeat in range(3):
                run(variant + '-run-' + str(repeat), [directory / 'probe.exe'], env)
                observed.append(observation((output / (variant + '-run-' + str(repeat) + '.stdout.log')).read_text(), variant))
            if observed != [observed[0]] * 3 or binaries != {name: descriptor(directory / name) for name in binaries}:
                raise ValueError('stable repeated scalar observations and binaries required')
            q['variants'][variant] = {'observation': observed[0], 'repetitions': 3, 'binaries': binaries}
            persist()
        values = {kind: value['observation'] for kind, value in q['variants'].items()}
        # Sticky exception flags may reflect expression evaluation. Compare only
        # rounding/masking/FTZ/DAZ controls, retaining the complete raw register.
        if len({value['mxcsr'] & 0xffc0 for value in values.values()}) != 1:
            raise ValueError('matched floating-point control state required')
        q['scalar_contraction_difference_observed'] = (values['msvc-default']['ordinary'] == 0
            and values['clang-default']['ordinary'] == FUSED and values['clang-no-contract']['ordinary'] == 0)
        frozen()
        if any(descriptor(path) != {key: q['compilers'][kind][key] for key in ('bytes', 'sha256')}
               for kind, path in compilers.items()):
            raise ValueError('compiler binary changed during scalar observation')
        q['probe_completed'] = True
    except BaseException as error:
        q['error'] = repr(error)
        raise
    finally:
        persist()


if __name__ == '__main__':
    main()
