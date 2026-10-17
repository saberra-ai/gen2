"""Fresh MSVC/clang-cl native comparison with identical admitted inference source."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from artifacts import acquire, matches
from build_valid_mel_candidate import descriptor, source_inventory
from frozen_git_sources import verify_git_sources
from platform_native import capture, msvc_tool
from profile_windows_sessions import HOST_SCRIPT
from qualification_storage import atomic_json
from windows_job_v2 import run_in_job
from windows_q8_model_metrics import (ORDER, admit_compiler_paths, build_command, command_compiler,
                                      compiler_environment, compiler_evidence, compiler_kind, timing_evidence)
from windows_session_timing import file_inventory, profiling_environment, summarize
from windows_q8_model_compiler import Diagnostic
from windows_q8_model_controls import kernel_evidence, model_free_controls, verify_bindings, MAX_LOG_BYTES
from windows_q8_model_metrics import PCM_SHA256, compare_native, telemetry, admit_q8_audit, derive_probe, version


REFERENCE = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/evidence-4/case-0-cap-4.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64', 'x86_64'):
        parser.error('actual Windows x86_64 compiler diagnostic required')
    tools = Path(__file__).resolve().parent
    repo, source, output = tools.parents[1], args.source.resolve(strict=True), args.output.resolve()
    if (args.jobs < 1 or output.exists() or output.is_symlink() or output.is_relative_to(repo)
            or output.is_relative_to(source) or source.is_relative_to(output)):
        parser.error('fresh disjoint output outside repository/native source and positive jobs required')
    protocol_path = tools / 'windows-q8-model.protocol.json'
    raw_protocol = protocol_path.read_bytes()
    p = json.loads(raw_protocol)
    commit = capture(['git', 'rev-parse', 'HEAD'], cwd=repo)

    def frozen():
        if protocol_path.read_bytes() != raw_protocol or p['order'] != ORDER or p['actual_pcm_sha256'] != PCM_SHA256:
            raise ValueError('compiler diagnostic frozen protocol/order/PCM changed')
        for group, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('compiler diagnostic frozen input differs: ' + name)

    frozen()
    expected = {**{'tools/nemotron-diarization/' + name: value for name, value in p['tools'].items()}, **p['repository_files']}
    verify_git_sources(repo, commit, expected)
    if subprocess.check_output(['git', 'show', commit + ':tools/nemotron-diarization/' + protocol_path.name],
            cwd=repo, env={**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}) != raw_protocol:
        raise ValueError('committed compiler diagnostic protocol required')
    output.mkdir()
    (output / 'protocol.json').write_bytes(raw_protocol)
    q = {'schema_version': 1, 'source_commit': commit, 'protocol': descriptor(protocol_path),
        'diagnostic_completed': False, 'release_qualified': False, 'package_qualified': False,
        'public_or_lifecycle_qualified': False, 'order': ORDER, 'builds': {}, 'cases': [], 'limits': p['limits']}
    persist = lambda: atomic_json(output / 'qualification.json', q)
    persist()
    runners = {}
    try:
        for kind in ('control', 'mx4'):
            evidence = output / ('evidence-' + kind)
            evidence.mkdir()
            runner = Diagnostic(output / ('source-' + kind), output / ('sdk-' + kind), evidence, args.jobs, 'clang-cl', p, frozen, kind)
            runners[kind] = runner
            q['builds'][kind] = {'construction_completed': False, 'compiler': str(runner.compiler),
                'compiler_binary': runner.compiler_identity,
                'probe_compiler': {'path': str(runner.msvc), **runner.probe_compiler_identity}}
            try:
                for root, origin, revision in [(runner.source, source, runner.lock['runtime']['revision']),
                        (runner.source / 'llama.cpp', source / 'llama.cpp', runner.lock['runtime']['llama_revision'])]:
                    name = 'native' if root == runner.source else 'llama'
                    runner.run('clone-' + name, ['git', 'clone', '--no-hardlinks', '--no-checkout', origin, root])
                    runner.run('checkout-' + name, ['git', '-C', root, 'checkout', '--detach', revision])
                runner.verify_source()
                admission = runner.build()
                exact_source = admit_q8_audit(runner.compiler_audit, kind)
                kernel = kernel_evidence(runner, json.loads((evidence / 'compile_commands.json').read_text()))
                q['builds'][kind].update(construction_completed=True, reported_version=runner.reported_version,
                    compiler_admission=admission, exact_source=exact_source, kernel=kernel,
                    native_probe_source=runner.native_probe_source, installed_files=file_inventory(runner.prefix),
                    compiler_audit=descriptor(evidence / 'platform-llama-materialized-audit.json'),
                    windows_private_names=runner.naming, build_compatibility=runner.compatibility)
                q['builds'][kind]['compiler_evidence_files'] = {name: descriptor(evidence / name) for name in
                    ['compiler-C.cmake', 'compiler-CXX.cmake', 'compiler-admission.json',
                     'CMakeCache.txt', 'compile_commands.json', 'platform-llama-materialized-audit.json']}
            finally:
                q.setdefault('commands', {})[kind] = runner.commands
                persist()
        if q['builds']['control']['probe_compiler'] != q['builds']['mx4']['probe_compiler']:
            raise ValueError('identical MSVC probe compiler required for both SDKs')
        original_probe = (tools / 'native_stream_probe_valid_mel.cpp').read_bytes()
        for kind, runner in runners.items():
            if (runner.output / 'native_stream_probe_bounded.cpp').read_bytes() != derive_probe(original_probe, 4, kind):
                raise ValueError('only exact diagnostic probe version literal may differ')
        host_runner = runners['control']
        host_runner.run('receiver-host', ['pwsh', '-NoProfile', '-NonInteractive', '-Command', HOST_SCRIPT])
        q['host'] = json.loads((host_runner.output / 'receiver-host.stdout.log').read_text(encoding='utf-8-sig'))
        if q['host']['avx2'] is not True or q['host']['fma'] is not True:
            raise ValueError('actual AVX2/FMA receiver required before model admission')
        def unchanged():
            frozen()
            if 'assets' in q and file_inventory(output / 'assets') != q['assets']:
                raise ValueError('compiler diagnostic assets changed')
            for kind, runner in runners.items():
                runner.verify_sources(runner.build_directory)
                verify_bindings(q['builds'][kind]['kernel']['bindings'])
                if 'model_free_controls' in q['builds'][kind]:
                    verify_bindings(q['builds'][kind]['model_free_controls']['bindings'])
                if any(descriptor(runner.output / name) != expected
                        for name, expected in q['builds'][kind]['compiler_evidence_files'].items()):
                    raise ValueError('compiler diagnostic provenance changed')
                if (file_inventory(runner.prefix) != q['builds'][kind]['installed_files']
                        or descriptor(runner.compiler) != runner.compiler_identity
                        or descriptor(runner.msvc) != runner.probe_compiler_identity
                        or descriptor(runner.linker) != runner.linker_identity
                        or admit_compiler_paths(runner.compiler, runner.compiler_aliases) != runner.compiler_aliases
                        or source_inventory(runner.build_directory / '_deps/llama.cpp') != runner.compiler_audit['files']):
                    raise ValueError('compiler diagnostic SDK/compiler/source changed')

            verify_bindings(model_outputs)

        graph_reference, model_outputs = None, {}
        for kind, runner in runners.items():
            controls, graph_reference = model_free_controls(runner, unchanged, graph_reference)
            q['builds'][kind]['model_free_controls'] = controls
            persist()
        unchanged()
        assets = output / 'assets'
        assets.mkdir()
        for item in runners['control'].lock['artifacts']:
            if item['role'] != 'reference':
                frozen()
                acquire(item, assets)
                if not matches(assets / item['name'], item):
                    raise ValueError('compiler diagnostic asset differs: ' + item['name'])
        q['assets'] = file_inventory(assets)
        reference_path = repo / REFERENCE
        reference, reference_raw = json.loads(reference_path.read_text()), Path(str(reference_path) + '.f32').read_bytes()

        for index, kind in enumerate(ORDER):
            unchanged()
            runner = runners[kind]
            phase = 'case-' + str(index) + '-' + kind
            target = runner.output / (phase + '.json')
            case = {'index': index, 'variant': kind, 'command_started': True}
            q['cases'].append(case)
            persist()
            runner.run(phase, [runner.probe, assets / 'Nemotron-3-Diarization.q8_0.gguf',
                assets / 'ami_en2002d_2132.wav', '1', '-1', target],
                environment=profiling_environment(runner.environment, 'control'))
            actual, raw = json.loads(target.read_text()), Path(str(target) + '.f32').read_bytes()
            if (runner.output / (phase + '.log')).stat().st_size > MAX_LOG_BYTES:
                raise ValueError('bounded full-model log required')
            log = (runner.output / (phase + '.log')).read_text()
            case.update(result=actual, result_file=descriptor(target), probabilities=descriptor(Path(str(target) + '.f32')),
                workers=telemetry(log, 4), timing_control=summarize(log, 'control', actual['elapsed_micros']))
            persist()
            if case['workers']['effective_threads'] != 4 or actual.get('push_count') != 58 or type(actual.get('push_count')) is not int:
                raise ValueError('fixed four-worker/58-push compiler diagnostic required')
            case['comparison'] = compare_native(reference, actual, reference_raw, raw, kind)
            for path in (target, Path(str(target) + '.f32'), runner.output / (phase + '.log'), runner.output / (phase + '.stdout.log')):
                model_outputs[str(path)] = descriptor(path)
            persist()
        unchanged()
        q['model_output_bindings'] = model_outputs
        q['timing_comparison'] = timing_evidence(q['cases'])
        q['diagnostic_completed'] = True
    except BaseException as error:
        q['error'] = repr(error)
        raise
    finally:
        for kind, runner in runners.items():
            q.setdefault('commands', {})[kind] = runner.commands
        persist()


if __name__ == '__main__':
    main()
