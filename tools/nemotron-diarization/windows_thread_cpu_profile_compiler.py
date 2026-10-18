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
from windows_thread_cpu_profile_metrics import (ORDER, admit_compiler_paths, build_command, command_compiler,
                                      compiler_environment, compiler_evidence, compiler_kind, timing_evidence)
from windows_session_timing import file_inventory, profiling_environment, summarize
from windows_thread_cpu_profile_worker import Diagnostic as WorkerDiagnostic
from windows_thread_cpu_profile_metrics import PCM_SHA256, compare_native, telemetry


REFERENCE = 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-worker-matrix/evidence-4/case-0-cap-4.json'


class Diagnostic(WorkerDiagnostic):
    def __init__(self, source, prefix, output, jobs, kind, protocol, frozen):
        super().__init__(source, prefix, output, jobs, 4, protocol, frozen)
        self.environment = compiler_environment(profiling_environment(self.environment, 'control'))
        self.kind = compiler_kind(kind)
        self.msvc = msvc_tool(self.environment, 'cl.exe')
        self.probe_compiler_identity = descriptor(self.msvc)
        self.compiler = msvc_tool(self.environment, 'cl.exe' if kind == 'msvc' else 'clang-cl.exe')
        self.compiler_identity = descriptor(self.compiler)

    def run(self, name, command, cwd=None, environment=None):
        self.frozen()
        command = list(map(str, command))
        if name == 'sdk-build':
            command = build_command(command, self.msvc, self.compiler, self.kind)
        if name == 'windows-provenance':
            index = command.index('-Compiler') + 1
            if command[index] != str(self.msvc):
                raise ValueError('original provenance compiler differs')
            command[index] = str(self.compiler)
        if name == 'native-probe-build' and (Path(command[0]).resolve(strict=True) != self.msvc
                or descriptor(self.msvc) != self.probe_compiler_identity):
            raise ValueError('fixed MSVC probe compiler required')
        record = {'name': name, 'command': command, 'cwd': str(cwd or self.repo),
                  'process_supervision': 'Windows kill-on-close Job v2 with scoped snapshot and owned-background cleanup', 'process_tree_reaped': False}
        self.commands.append(record)
        atomic_json(self.output / 'commands.json', self.commands)
        print('START ' + self.kind + ' ' + name, flush=True)
        started = time.monotonic()
        try:
            with (self.output / (name + '.stdout.log')).open('xb') as out, (self.output / (name + '.log')).open('xb') as err:
                result = run_in_job(command, cwd=cwd or self.repo,
                    env=environment if environment is not None else self.environment,
                    stdout=out, stderr=err, timeout=1800 if name == 'sdk-build' else 600)
            record['returncode'] = result.returncode
            record['process_tree_reaped'] = result.process_tree_reaped
            record['job_observation'] = result.job_observation
            result.check_returncode()
        except BaseException as error:
            record['error'] = repr(error)
            record['process_tree_reaped'] = getattr(error, 'process_tree_reaped', record['process_tree_reaped'])
            record['job_observation'] = getattr(error, 'job_observation', record.get('job_observation'))
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            atomic_json(self.output / 'commands.json', self.commands)
        print('PASS ' + self.kind + ' ' + name, flush=True)

    def build(self):
        if self.kind == 'clang-cl':
            self.run('compiler-version', [self.compiler, '--version'])
        super().build()
        compiler_files = {}
        for language in ('C', 'CXX'):
            path, = self.build_directory.glob('CMakeFiles/*/CMake' + language + 'Compiler.cmake')
            target = self.output / ('compiler-' + language + '.cmake')
            shutil.copy2(path, target)
            compiler_files[language] = target.read_text()
        database = json.loads((self.output / 'compile_commands.json').read_text())
        self.compiler_aliases = admit_compiler_paths(self.compiler,
            {command_compiler(row['command']) for row in database})
        evidence = compiler_evidence((self.output / 'CMakeCache.txt').read_text(),
            database, compiler_files, self.compiler, self.kind, self.compiler_aliases)
        evidence['compiler_binary'] = self.compiler_identity
        evidence['compiler_path_aliases'] = self.compiler_aliases
        self.linker = Path(evidence['linker']).resolve(strict=True)
        self.linker_identity = descriptor(self.linker)
        evidence['linker_binary'] = self.linker_identity
        if descriptor(self.compiler) != self.compiler_identity:
            raise ValueError('compiler binary changed during construction')
        atomic_json(self.output / 'compiler-admission.json', evidence)
        return evidence
