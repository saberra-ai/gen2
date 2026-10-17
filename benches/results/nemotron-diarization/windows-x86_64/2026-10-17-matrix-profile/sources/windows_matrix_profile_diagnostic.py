"""Fixed off/on/on/off matrix-path profile; no runtime or release qualification."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import struct
import sys

from artifacts import acquire, matches
from build_valid_mel_candidate import descriptor, source_inventory
from frozen_git_sources import verify_git_sources
from ggml_matrix_profile import MAX_LOG_BYTES, environment, summarize
from platform_native import capture, msvc_probe_command
from profile_windows_sessions import HOST_SCRIPT
from qualification_storage import atomic_json
from windows_matrix_profile_compiler import Diagnostic
from windows_matrix_profile_metrics import (ORDER, PCM_SHA256, admit_profile_audit, compare_native,
    profile_signature, telemetry, timing_evidence)
from windows_compiler_metrics import admit_compiler_paths
from windows_session_timing import file_inventory
from windows_matrix_profile_report import save_profile


REFERENCE = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/evidence-4/case-0-cap-4.json'


def verify_control_evidence(binary, control):
    if (descriptor(binary) != control['binary']
            or any(descriptor(Path(name)) != value for name, value in control['inputs'].items())
            or any(descriptor(Path(name)) != value for name, value in control['outputs_and_logs'].items())):
        raise ValueError('earlier matrix control inputs or evidence changed')


def model_free_controls(runner, frozen):
    """Validate actual Windows SGEMM/conversion/fallback before model acquisition."""
    source=runner.tools/'ggml_matrix_profile_control.cpp'
    binary=runner.output/'matrix-profile-control.exe'
    libraries=[]
    for stem in ('ggml','ggml-base','ggml-cpu'):
        path,=runner.build_directory.rglob('gen2_nemo_windows_'+stem+'.lib')
        libraries.append(path)
    inputs={str(path):descriptor(path)for path in [source,*libraries]}
    command=msvc_probe_command(runner.environment,[source],
        [runner.build_directory/'_deps/llama.cpp/ggml/include'],libraries[0],binary)+libraries[1:]
    if Path(command[0]).resolve(strict=True)!=runner.msvc:
        raise ValueError('same admitted MSVC control compiler required')
    runner.run('matrix-control-build',command,runner.output)
    binary_identity=descriptor(binary)
    mapping={key:value for key,value in runner.environment.items()if key.upper()!='PATH'}
    old_path={key.upper():value for key,value in runner.environment.items()}['PATH']
    mapping['PATH']=str(runner.prefix/'bin')+os.pathsep+old_path
    rows,reference,output_bindings=[],{},{}
    for threads in (1,2,4):
        for index,enabled in enumerate(ORDER):
            frozen()
            if (descriptor(binary)!=binary_identity or any(descriptor(Path(name))!=value for name,value in inputs.items())
                    or any(descriptor(Path(name))!=value for name,value in output_bindings.items())):
                raise ValueError('model-free matrix inputs or earlier outputs changed')
            phase=f'matrix-control-{threads}-{index}';target=runner.output/phase
            runner.run(phase,[binary,str(threads),target],environment=environment(mapping,enabled))
            if (runner.output/(phase+'.stdout.log')).read_text().splitlines()!=[
                    f'matrix-control threads={threads} shape={shape} quantized={shape%2} columns={32 if shape<2 else 1} repeat={repeat} bytes={8192 if shape<2 else 256}'
                    for shape in range(4)for repeat in range(3)]:
                raise ValueError('all twelve model-free matrix graph calls required')
            path=runner.output/(phase+'.log')
            if path.stat().st_size>MAX_LOG_BYTES:raise ValueError('bounded model-free matrix log required')
            observed=summarize(path.read_text(),enabled,expected_graphs=12,expected_threads=threads,expected_matrices=1)
            observed.pop('cpu')
            if enabled:
                if len({g['pool']for g in observed['graphs']})!=1:raise ValueError('one persistent model-free pool required')
                for graph_index,graph in enumerate(observed['graphs']):
                    shape=graph_index//3
                    for worker in graph['workers']:
                        matrix,=worker['matrices']
                        if (worker['fused']!=1 or matrix['route']!=[2,4,5,5][shape]
                                or matrix['types']!=[8 if shape%2 else 0,0,0]
                                or matrix['a']!=[64,64,1,1] or matrix['b']!=[64,32 if shape<2 else 1,1,1]
                                or matrix['first'][:3]!=([1,1,0]if shape==0 else[1,0,1])
                                or matrix['second'][:3]!=([1,1,0]if shape==1 else[1,0,1]if shape==3 else[0,0,0])
                                or matrix['conversion'][0]!=shape%2):
                            raise ValueError('actual matrix route/type/shape/attempt/conversion/fusion mismatch')
            outputs={}
            for shape in range(4):
                for repeat in range(3):
                    output=Path(str(target)+f'.{shape}.{repeat}.f32');raw=output.read_bytes()
                    if len(raw)!=(8192 if shape<2 else 256):raise ValueError('complete model-free matrix vector required')
                    values=[v for v,in struct.iter_unpack('<f',raw)]
                    if not all(math.isfinite(v)for v in values)or not any(v!=0 for v in values):
                        raise ValueError('finite nonzero model-free matrix output required')
                    if shape not in reference:reference[shape]=raw
                    if raw!=reference[shape]:raise ValueError('model-free matrix outputs differ per shape')
                    outputs[output.name]=descriptor(output);output_bindings[str(output)]=descriptor(output)
            for suffix in ('.stdout.log','.log'):
                path=runner.output/(phase+suffix);output_bindings[str(path)]=descriptor(path)
            rows.append({'threads':threads,'index':index,'enabled':enabled,'profile':observed,'outputs':outputs})
    frozen()
    if (descriptor(binary)!=binary_identity or any(descriptor(Path(name))!=value for name,value in inputs.items())
            or any(descriptor(Path(name))!=value for name,value in output_bindings.items())):
        raise ValueError('model-free matrix inputs or outputs changed')
    return {'cases':rows,'inputs':inputs,'binary':binary_identity,'outputs_and_logs':output_bindings,
            'limits':'Twelve actual Windows processes and 144 per-shape byte-exact vectors; no model or release claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64', 'x86_64'):
        parser.error('actual Windows x86_64 CPU profile required')
    tools = Path(__file__).resolve().parent
    repo, source, output = tools.parents[1], args.source.resolve(strict=True), args.output.resolve()
    if (args.jobs < 1 or output.exists() or output.is_symlink() or output.is_relative_to(repo)
            or output.is_relative_to(source) or source.is_relative_to(output)):
        parser.error('fresh disjoint output outside repository/native source required')
    protocol_path = tools / 'windows-matrix-profile.protocol.json'
    raw_protocol = protocol_path.read_bytes()
    p = json.loads(raw_protocol)
    commit = capture(['git', 'rev-parse', 'HEAD'], cwd=repo)

    def frozen():
        if protocol_path.read_bytes() != raw_protocol or p['order'] != ORDER or p['actual_pcm_sha256'] != PCM_SHA256:
            raise ValueError('CPU profile frozen protocol/order/PCM changed')
        for group, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('CPU profile frozen input differs: ' + name)

    frozen()
    expected = {**{'tools/nemotron-diarization/' + name: value for name, value in p['tools'].items()}, **p['repository_files']}
    verify_git_sources(repo, commit, expected)
    if subprocess.check_output(['git', 'show', commit + ':tools/nemotron-diarization/' + protocol_path.name],
            cwd=repo, env={**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}) != raw_protocol:
        raise ValueError('committed CPU profile protocol required')
    output.mkdir()
    (output / 'protocol.json').write_bytes(raw_protocol)
    q = {'schema_version': 1, 'source_commit': commit, 'protocol': descriptor(protocol_path),
        'diagnostic_completed': False, 'release_qualified': False, 'package_qualified': False,
        'public_or_lifecycle_qualified': False, 'order': ORDER, 'build': {}, 'cases': [], 'limits': p['limits']}
    persist = lambda: atomic_json(output / 'qualification.json', q)
    persist()
    runner = None
    try:
        evidence = output / 'evidence'
        evidence.mkdir()
        runner = Diagnostic(output / 'source', output / 'sdk', evidence, args.jobs, 'clang-cl', p, frozen)
        q['build'] = {'construction_completed': False, 'compiler': str(runner.compiler),
            'compiler_binary': runner.compiler_identity, 'probe_compiler': {'path': str(runner.msvc), **runner.probe_compiler_identity}}
        for root, origin, revision in [(runner.source, source, runner.lock['runtime']['revision']),
                (runner.source / 'llama.cpp', source / 'llama.cpp', runner.lock['runtime']['llama_revision'])]:
            name = 'native' if root == runner.source else 'llama'
            runner.run('clone-' + name, ['git', 'clone', '--no-hardlinks', '--no-checkout', origin, root])
            runner.run('checkout-' + name, ['git', '-C', root, 'checkout', '--detach', revision])
        runner.verify_source()
        admission = runner.build()
        q['build'].update(construction_completed=True, reported_version=runner.reported_version,
            compiler_admission=admission, native_probe_source=runner.native_probe_source,
            installed_files=file_inventory(runner.prefix), compiler_audit=descriptor(evidence / 'platform-llama-materialized-audit.json'),
            exact_profile_source=admit_profile_audit(runner.compiler_audit),
            windows_private_names=runner.naming, build_compatibility=runner.compatibility)
        q['build']['compiler_evidence_files'] = {name: descriptor(evidence / name) for name in
            ['compiler-C.cmake', 'compiler-CXX.cmake', 'compiler-admission.json', 'CMakeCache.txt',
             'compile_commands.json', 'platform-llama-materialized-audit.json']}
        persist()

        def unchanged():
            frozen()
            if 'assets' in q and file_inventory(output / 'assets') != q['assets']:
                raise ValueError('CPU profile assets changed')
            runner.verify_sources(runner.build_directory)
            if (file_inventory(runner.prefix) != q['build']['installed_files']
                    or any(descriptor(evidence / name) != value for name, value in q['build']['compiler_evidence_files'].items())
                    or descriptor(runner.compiler) != runner.compiler_identity or descriptor(runner.msvc) != runner.probe_compiler_identity
                    or descriptor(runner.linker) != runner.linker_identity
                    or admit_compiler_paths(runner.compiler, runner.compiler_aliases) != runner.compiler_aliases
                    or source_inventory(runner.build_directory / '_deps/llama.cpp') != runner.compiler_audit['files']):
                raise ValueError('matrix profile SDK/compiler/source changed')
            if 'model_free_controls' in q:
                verify_control_evidence(evidence / 'matrix-profile-control.exe', q['model_free_controls'])

        unchanged()
        runner.run('receiver-host', ['pwsh', '-NoProfile', '-NonInteractive', '-Command', HOST_SCRIPT])
        q['host'] = json.loads((evidence / 'receiver-host.stdout.log').read_text(encoding='utf-8-sig'))
        if q['host']['avx2'] is not True or q['host']['fma'] is not True:
            raise ValueError('actual AVX2/FMA receiver required before compute')
        q['model_free_controls'] = model_free_controls(runner, unchanged)
        unchanged()
        persist()
        assets = output / 'assets'
        assets.mkdir()
        for item in runner.lock['artifacts']:
            if item['role'] != 'reference':
                frozen()
                acquire(item, assets)
                if not matches(assets / item['name'], item):
                    raise ValueError('CPU profile asset differs')
        q['assets'] = file_inventory(assets)
        reference_path = repo / REFERENCE
        reference, reference_raw = json.loads(reference_path.read_text()), Path(str(reference_path) + '.f32').read_bytes()
        signatures = []
        model_outputs = {}
        for index, enabled in enumerate(ORDER):
            unchanged()
            if any(descriptor(Path(name))!=value for name,value in model_outputs.items()):
                raise ValueError('earlier model outputs/profile/log changed')
            if file_inventory(assets) != q['assets']:
                raise ValueError('CPU profile assets changed')
            phase = 'case-' + str(index)
            target = evidence / (phase + '.json')
            case = {'index': index, 'enabled': enabled, 'command_started': True}
            q['cases'].append(case)
            persist()
            runner.run(phase, [runner.probe, assets / 'Nemotron-3-Diarization.q8_0.gguf',
                assets / 'ami_en2002d_2132.wav', '1', '-1', target], environment=environment(runner.environment, enabled))
            actual, raw = json.loads(target.read_text()), Path(str(target) + '.f32').read_bytes()
            log_path = evidence / (phase + '.log')
            if log_path.stat().st_size > MAX_LOG_BYTES:
                raise ValueError('bounded CPU profiling log required')
            log = log_path.read_text()
            case.update(result=actual, result_file=descriptor(target), probabilities=descriptor(Path(str(target) + '.f32')),
                        workers=telemetry(log, 4))
            persist()
            case['comparison'] = compare_native(reference, actual, reference_raw, raw)
            if raw != reference_raw or type(actual['push_count']) is not int or actual['push_count'] != 58:
                raise ValueError('byte-exact complete 58-push CPU profiling output required')
            observed = summarize(log, enabled, expected_graphs=p['expected_graphs'], expected_threads=4,
                                 expected_matrices=p['expected_matrices'])
            if enabled:
                if len({graph['pool'] for graph in observed['graphs']}) != 1:
                    raise ValueError('one admitted persistent CPU pool required')
                signatures.append(profile_signature(observed))
            profile_path=evidence/(phase+'.matrix-profile.json')
            case['profile'] = save_profile(profile_path, observed)
            for path in (target,Path(str(target)+'.f32'),log_path,evidence/(phase+'.stdout.log'),profile_path):
                model_outputs[str(path)]=descriptor(path)
            persist()
        unchanged()
        if any(descriptor(Path(name))!=value for name,value in model_outputs.items()):
            raise ValueError('saved complete model evidence changed')
        q['model_output_bindings']=model_outputs
        if signatures[0] != signatures[1]:
            raise ValueError('repeated actual graph/fusion execution differs')
        q['timing_comparison'] = timing_evidence(q['cases'])
        q['diagnostic_completed'] = True
    except BaseException as error:
        q['error'] = repr(error)
        raise
    finally:
        if runner is not None:
            q['commands'] = runner.commands
        persist()


if __name__ == '__main__':
    main()
