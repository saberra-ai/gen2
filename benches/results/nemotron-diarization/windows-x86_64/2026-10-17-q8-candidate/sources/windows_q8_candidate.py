"""Fresh bounded Windows Q8 SDK, full native/public parity, lifecycle and RTF."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

from artifacts import acquire, matches
from build_valid_mel_candidate import descriptor, source_inventory
from frozen_git_sources import verify_git_sources
from platform_native import Qualification as BaseQualification, capture, msvc_tool
from platform_bounded_windows_native import Qualification as BoundedQualification
from pooled_runtime import BOUNDED_PROFILE, load_profile, profile_provenance
from profile_windows_sessions import HOST_SCRIPT
from qualification_storage import atomic_json
from windows_job_v2 import run_in_job
from windows_q8_candidate_build import Qualification as BuildQualification
from windows_q8_candidate_controls import model_free_controls
from windows_q8_candidate_metrics import (admit_compiler_paths, command_compiler, compiler_environment,
    compiler_evidence, build_command, admit_candidate_audit, compare_candidate_pair, both_pairs_realtime, PCM_SHA256)
from windows_q8_candidate_sources import load_recipe, install_q8_patch, RECIPE_ID
from windows_q8_candidate_verify import verify_completed_source
from windows_q8_model_controls import kernel_evidence, verify_bindings, MAX_LOG_BYTES
from windows_session_timing import file_inventory, profiling_environment

ORDER = ['native', 'public', 'public', 'native']
PROTOCOL = 'windows-q8-candidate.protocol.json'
LIFECYCLE = sorted({
    'dropping_a_diarization_child_keeps_the_parent_pipeline_active',
    'silence_short_tail_and_idle_deadline_have_explicit_terminal_results',
    'repeated_reload_full_queue_cancel_and_dropped_stream_release_resources',
    'streamed_uneven_pcm_matches_recording_and_finish_is_idempotent',
    'native_admission_cancellation_and_failed_load_release_reservations',
    'verified_provisioned_cache_is_reused_offline',
    'real_diarizer_matches_native_reference_and_resets_recordings'})


def require_uninstrumented(log):
    if any(marker in log for marker in ('[DEBUG-worker-v1]', '[timing]', '[gen2-cpu-', '[gen2-matrix-')):
        raise ValueError('production candidate must not emit diagnostic telemetry')


def require_host(host):
    cpus = host.get('cpu', [])
    if isinstance(cpus, dict):
        cpus = [cpus]
    if (host.get('avx2') is not True or host.get('fma') is not True or not cpus
            or any(type(cpu.get('NumberOfLogicalProcessors')) is not int
                   or cpu['NumberOfLogicalProcessors'] < 1 for cpu in cpus)
            or sum(cpu['NumberOfLogicalProcessors'] for cpu in cpus) < 4):
        raise ValueError('actual AVX2/FMA host with at least four logical processors required')


class Qualification(BuildQualification):
    def __init__(self, source, prefix, output, jobs, protocol, frozen):
        BaseQualification.__init__(self, source, prefix, output, jobs, True)
        self.lock_path, self.lock = load_profile(BOUNDED_PROFILE)
        self.recipe_path, self.recipe = load_recipe()
        self.environment = {**compiler_environment(profiling_environment(self.environment, 'control')),
                            'GIT_LFS_SKIP_SMUDGE': '1'}
        self.protocol, self.frozen = protocol, frozen
        self.kind = 'clang-cl'
        self.msvc, self.compiler = [msvc_tool(self.environment, name) for name in ('cl.exe', 'clang-cl.exe')]
        self.probe_compiler_identity, self.compiler_identity = descriptor(self.msvc), descriptor(self.compiler)
        if (self.compiler_identity != self.recipe['compiler']['binary']
                or self.probe_compiler_identity != self.recipe['compiler']['probe_compiler_binary']):
            raise ValueError('original model and probe compilers required')
        self.native_probe_source = self.compiler_audit = self.reported_version = None
        self.guard = None
        self.bindings = {}

    def verify_protocol(self):
        self.frozen()
        if self.native_probe_source is not None:
            from platform_bounded_metrics import derive_native_probe
            if (self.output / 'native_stream_probe_bounded.cpp').read_bytes() != derive_native_probe(
                    (self.tools / 'native_stream_probe_valid_mel.cpp').read_bytes()):
                raise ValueError('unchanged bounded native probe required')

    def run(self, name, command, cwd=None, environment=None):
        self.verify_protocol()
        if self.guard is not None:
            self.guard()
        command = list(map(str, command))
        if name == 'sdk-build':
            command = build_command(command, self.msvc, self.compiler, self.kind)
        if name == 'windows-provenance':
            index = command.index('-Compiler') + 1
            if command[index] != str(self.msvc):
                raise ValueError('original provenance compiler selection required')
            command[index] = str(self.compiler)
        if name in ('native-probe-build', 'matrix-control-build') and (
                Path(command[0]).resolve(strict=True) != self.msvc
                or descriptor(self.msvc) != self.probe_compiler_identity):
            raise ValueError('fixed original MSVC probe/graph compiler required')
        row = {'name': name, 'command': command, 'cwd': str(cwd or self.repo),
               'process_supervision': 'Windows kill-on-close Job v2 with scoped snapshot and owned-background cleanup',
               'process_tree_reaped': False}
        self.commands.append(row)
        atomic_json(self.output / 'commands.json', self.commands)
        print('START ' + name, flush=True)
        started = time.monotonic()
        try:
            with (self.output / (name + '.stdout.log')).open('xb') as out, (self.output / (name + '.log')).open('xb') as err:
                result = run_in_job(command, cwd=cwd or self.repo,
                    env=environment if environment is not None else self.environment,
                    stdout=out, stderr=err, timeout=1800 if name in ('sdk-build', 'gen2-build') else 600)
            row.update(returncode=result.returncode, process_tree_reaped=result.process_tree_reaped,
                       job_observation=result.job_observation)
            result.check_returncode()
            if self.guard is not None:
                self.guard()
                self.seal_outputs()
        except BaseException as error:
            row['error'] = repr(error)
            row['process_tree_reaped'] = getattr(error, 'process_tree_reaped', row['process_tree_reaped'])
            row['job_observation'] = getattr(error, 'job_observation', row.get('job_observation'))
            raise
        finally:
            row['elapsed_seconds'] = time.monotonic() - started
            atomic_json(self.output / 'commands.json', self.commands)
        print('PASS ' + name, flush=True)

    def seal_outputs(self):
        verify_bindings(self.bindings)
        for name, identity in file_inventory(self.output).items():
            if name != 'commands.json':
                self.bindings[str(self.output / name)] = identity
        verify_bindings(self.bindings)

    def verify_source(self):
        BoundedQualification.verify_source(self)
        install_q8_patch(self.source, self.output, 'mx4')
        verify_completed_source(self.source, None, self.lock, self.patch_path, True)
        shutil.copyfile(self.recipe_path, self.output / 'build-recipe.json')

    def build(self):
        self.run('compiler-version', [self.compiler, '--version'])
        super().build()
        compiler_files = {}
        for language in ('C', 'CXX'):
            path, = self.build_directory.glob('CMakeFiles/*/CMake' + language + 'Compiler.cmake')
            target = self.output / ('compiler-' + language + '.cmake')
            shutil.copyfile(path, target)
            compiler_files[language] = target.read_text()
        database = json.loads((self.output / 'compile_commands.json').read_text())
        self.compiler_aliases = admit_compiler_paths(self.compiler,
            {command_compiler(row['command']) for row in database})
        admission = compiler_evidence((self.output / 'CMakeCache.txt').read_text(), database,
            compiler_files, self.compiler, self.kind, self.compiler_aliases)
        self.linker = Path(admission['linker']).resolve(strict=True)
        self.linker_identity = descriptor(self.linker)
        admission.update(compiler_binary=self.compiler_identity, compiler_path_aliases=self.compiler_aliases,
                         linker_binary=self.linker_identity)
        atomic_json(self.output / 'compiler-admission.json', admission)
        return admission, admit_candidate_audit(self.compiler_audit), database


def run_histories(runner, assets, executable, q, persist):
    cases = q['cases']
    for index, kind in enumerate(ORDER):
        phase = 'case-' + str(index) + '-' + kind
        target = runner.output / (phase + '.json')
        raw_path = Path(str(target) + '.f32')
        case = {'index': index, 'kind': kind, 'command_started': True}
        cases.append(case)
        persist()
        if kind == 'native':
            command = [runner.probe, assets / 'Nemotron-3-Diarization.q8_0.gguf',
                       assets / 'ami_en2002d_2132.wav', '1', '-1', target]
        else:
            command = [executable, assets, runner.library, assets / 'ami_en2002d_2132.wav', '1', raw_path]
        runner.run(phase, command)
        stdout, log = [runner.output / (phase + suffix) for suffix in ('.stdout.log', '.log')]
        if max(stdout.stat().st_size, log.stat().st_size) > MAX_LOG_BYTES:
            raise ValueError('bounded complete-history logs required')
        require_uninstrumented(log.read_text())
        if kind == 'public':
            atomic_json(target, json.loads(stdout.read_text()))
        case.update(result=json.loads(target.read_text()), result_file=descriptor(target), probabilities=descriptor(raw_path),
                    result_path=target.name, probability_path=raw_path.name)
        runner.seal_outputs()
        persist()
        if index in (1, 3):
            first, second = cases[index - 1:index + 1]
            native, public = (first, second) if first['kind'] == 'native' else (second, first)
            comparison = compare_candidate_pair(public['result'], native['result'],
                (runner.output / native['probability_path']).read_bytes(),
                (runner.output / public['probability_path']).read_bytes())
            q['comparisons'].append(comparison)
            persist()
    runner.guard()


def run_lifecycle(runner, assets, native, q, persist):
    reference = runner.output / 'reference'
    reference.mkdir()
    atomic_json(reference / 'ami.json', {'segments': [
        {'start': s['start_seconds'], 'end': s['end_seconds'], 'speaker': s['speaker']}
        for s in native['result']['turns']]})
    shutil.copyfile(runner.output / native['probability_path'], reference / 'ami.json.f32')
    runner.seal_outputs()
    env = {**runner.environment, 'GEN2_NEMOTRON_ASSETS': str(assets), 'GEN2_NEMO_LIBRARY': str(runner.library),
           'GEN2_NEMOTRON_REFERENCE': str(reference), 'GEN2_NEMO_EXECUTION': 'cpu'}
    command = ['cargo', 'test', '--locked', '--release', '--no-default-features', '--features',
               'backend-diar-nemo', '--test', 'diarization_native', '--']
    runner.run('native-lifecycle-list', [*command, '--ignored', '--list'], environment=env)
    listed = {line.removesuffix(': test') for line in (runner.output / 'native-lifecycle-list.stdout.log').read_text().splitlines()
              if line.endswith(': test')}
    if not set(LIFECYCLE) <= listed:
        raise ValueError('all seven original lifecycle tests must be compiled')
    for index, test in enumerate(LIFECYCLE):
        phase = 'native-lifecycle-' + str(index)
        runner.run(phase, [*command, '--ignored', '--exact', test, '--test-threads=1'], environment=env)
        stdout = (runner.output / (phase + '.stdout.log')).read_text()
        require_uninstrumented((runner.output / (phase + '.log')).read_text())
        if (not re.search(r'^test ' + re.escape(test) + r' \.\.\. ok$', stdout, re.MULTILINE)
                or '1 passed; 0 failed; 0 ignored;' not in stdout):
            raise ValueError('required lifecycle test did not execute: ' + test)
        q['completed_lifecycle_tests'].append(test)
        persist()
    runner.guard()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64', 'x86_64'):
        parser.error('actual Windows x86_64 production qualification required')
    tools = Path(__file__).resolve().parent
    repo, origin, output = tools.parents[1], args.source.resolve(strict=True), args.output.resolve()
    if (args.jobs < 1 or output.exists() or args.output.is_symlink() or output.is_relative_to(repo)
            or output.is_relative_to(origin) or origin.is_relative_to(output)):
        parser.error('fresh disjoint output outside repository/native source and positive jobs required')
    protocol_path = tools / PROTOCOL
    raw_protocol = protocol_path.read_bytes()
    p = json.loads(raw_protocol)
    recipe_path, recipe = load_recipe()
    lock_path, lock = load_profile(BOUNDED_PROFILE)
    commit = capture(['git', 'rev-parse', 'HEAD'], cwd=repo)

    def frozen():
        if (protocol_path.read_bytes() != raw_protocol or p['order'] != ORDER
                or p['actual_pcm_sha256'] != PCM_SHA256
                or p['required_lifecycle_tests'] != LIFECYCLE or p['build_recipe'] != descriptor(recipe_path)
                or p['runtime_provenance'] != profile_provenance(lock_path, lock)):
            raise ValueError('frozen candidate protocol/order/recipe/lifecycle differs')
        for group, root in [('tools', tools), ('repository_files', repo)]:
            for name, expected in p[group].items():
                if descriptor(root / name) != expected:
                    raise ValueError('frozen candidate input differs: ' + name)

    frozen()
    expected = {**{'tools/nemotron-diarization/' + name: value for name, value in p['tools'].items()}, **p['repository_files']}
    verify_git_sources(repo, commit, expected)
    if subprocess.check_output(['git', 'show', commit + ':tools/nemotron-diarization/' + PROTOCOL], cwd=repo,
            env={**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_NO_REPLACE_OBJECTS': '1'}) != raw_protocol:
        raise ValueError('committed candidate protocol required')
    output.mkdir()
    (output / 'protocol.json').write_bytes(raw_protocol)
    q = {**profile_provenance(lock_path, lock), 'schema_version': 1, 'source_commit': commit,
         'protocol': descriptor(protocol_path), 'build_recipe_id': RECIPE_ID, 'build_recipe': recipe,
         'qualification_completed': False, 'development_native_gates_passed': False,
         'functional_lifecycle_gates_passed': False, 'realtime_gate_passed': False,
         'graph_input_payloads_fully_retained': False, 'release_qualified': False, 'package_qualified': False,
         'order': ORDER, 'cases': [], 'comparisons': [], 'completed_lifecycle_tests': [], 'limits': p['limits']}
    persist = lambda: atomic_json(output / 'qualification.json', q)
    persist()
    runner = None
    try:
        evidence = output / 'evidence'
        evidence.mkdir()
        runner = Qualification(output / 'source', output / 'sdk', evidence, args.jobs, p, frozen)
        for root, source, revision in [(runner.source, origin, lock['runtime']['revision']),
                (runner.source / 'llama.cpp', origin / 'llama.cpp', lock['runtime']['llama_revision'])]:
            name = 'native' if root == runner.source else 'llama'
            runner.run('clone-' + name, ['git', 'clone', '--no-hardlinks', '--no-checkout', source, root])
            runner.run('checkout-' + name, ['git', '-C', root, 'checkout', '--detach', revision])
        runner.verify_source()
        admission, source_admission, database = runner.build()
        target = Path(runner.environment.get('CARGO_TARGET_DIR', repo / 'target'))
        if not target.is_absolute():
            target = repo / target
        executable = target / 'release/examples/nemotron_valid_mel_stream.exe'
        tests = list((target / 'release/deps').glob('diarization_native-*.exe'))
        if not executable.is_file() or len(tests) != 1:
            raise ValueError('one built public example and lifecycle executable required')
        retained = evidence / 'public-binaries'
        retained.mkdir()
        public_bindings = {}
        for path in (executable, *tests):
            copy = retained / path.name
            shutil.copyfile(path, copy)
            identity = descriptor(path)
            if descriptor(copy) != identity:
                raise ValueError('retained public executable differs')
            public_bindings.update({str(path): identity, str(copy): identity})
        q['build'] = {'compiler_admission': admission, 'exact_source': source_admission,
            'reported_version': runner.reported_version, 'installed_files': file_inventory(runner.prefix),
            'native_probe_source': runner.native_probe_source, 'windows_private_names': runner.naming,
            'build_compatibility': runner.compatibility, 'public_executables': public_bindings}
        runner.run('receiver-host', ['pwsh', '-NoProfile', '-NonInteractive', '-Command', HOST_SCRIPT])
        q['host'] = json.loads((evidence / 'receiver-host.stdout.log').read_text(encoding='utf-8-sig'))
        require_host(q['host'])
        assets = output / 'assets'

        def unchanged():
            runner.verify_protocol()
            verify_completed_source(runner.source, runner.build_directory, runner.lock, runner.patch_path, True)
            verify_bindings(runner.bindings)
            verify_bindings(q['build']['public_executables'])
            if (file_inventory(runner.prefix) != q['build']['installed_files']
                    or source_inventory(runner.build_directory / '_deps/llama.cpp') != runner.compiler_audit['files']
                    or descriptor(runner.compiler) != runner.compiler_identity
                    or descriptor(runner.msvc) != runner.probe_compiler_identity
                    or descriptor(runner.linker) != runner.linker_identity
                    or admit_compiler_paths(runner.compiler, runner.compiler_aliases) != runner.compiler_aliases):
                raise ValueError('candidate SDK/compiler/source changed')
            if 'assets' in q and file_inventory(assets) != q['assets']:
                raise ValueError('candidate assets changed')
            for key in ('kernel', 'model_free_controls'):
                if key in q['build']:
                    verify_bindings(q['build'][key]['bindings'])

        runner.guard = unchanged
        runner.seal_outputs()
        q['build']['kernel'] = kernel_evidence(runner, database)
        runner.seal_outputs()
        q['build']['model_free_controls'], _ = model_free_controls(runner, unchanged)
        q['graph_input_payloads_fully_retained'] = True
        runner.seal_outputs()
        unchanged()
        persist()
        assets.mkdir()
        for item in lock['artifacts']:
            if item['role'] != 'reference':
                unchanged()
                acquire(item, assets)
                if not matches(assets / item['name'], item):
                    raise ValueError('candidate asset differs: ' + item['name'])
        q['assets'] = file_inventory(assets)
        persist()
        runner.run('windows-default-loader-control', [sys.executable,
                   tools / 'windows_loader_control.py', runner.library])
        run_histories(runner, assets, executable, q, persist)
        q['realtime_gate_passed'] = both_pairs_realtime(q['comparisons'])
        persist()
        run_lifecycle(runner, assets, q['cases'][0], q, persist)
        unchanged()
        q['functional_lifecycle_gates_passed'] = (len(q['comparisons']) == 2
            and all(c['passed'] is True and c['probabilities_bit_exact'] is True for c in q['comparisons'])
            and q['completed_lifecycle_tests'] == LIFECYCLE)
        q['qualification_completed'] = True
        persist()
        if not q['functional_lifecycle_gates_passed'] or not q['realtime_gate_passed']:
            raise ValueError('all four original native/public RTFs must be below one and all lifecycle gates pass')
        q['development_native_gates_passed'] = True
    except BaseException as error:
        q['error'] = repr(error)
        raise
    finally:
        if runner is not None:
            q['commands'] = runner.commands
            q['retained_output_bindings'] = runner.bindings
        persist()


if __name__ == '__main__':
    main()
