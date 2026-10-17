"""Execution-order, failure preservation and source-freeze controls; no SDK/model."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from build_valid_mel_candidate import descriptor
from windows_q8_candidate import (Qualification, ORDER, LIFECYCLE, require_host, require_uninstrumented,
                                  run_histories, run_lifecycle)
from windows_q8_candidate_producer_sources import derive_build
from windows_q8_candidate_metrics import BOUNDED_RESULT, both_pairs_realtime
from windows_q8_model_controls import verify_bindings

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]


class HistoryFixture:
    """Replay already retained vectors; synthetic timings test admission only."""
    def __init__(self, output, elapsed=59_000_000, truncate_at=None):
        self.output, self.elapsed, self.truncate_at = output, elapsed, truncate_at
        self.probe, self.library = output / 'probe.exe', output / 'library.dll'
        self.bindings, self.phases = {}, []
        self.environment = {}
        root = REPO / BOUNDED_RESULT
        self.native = json.loads((root / 'native-ami.json').read_text())
        self.public = json.loads((root / 'public.json').read_text())
        self.raw = (root / 'native-ami.json.f32').read_bytes()

    def guard(self):
        verify_bindings(self.bindings)

    def seal_outputs(self):
        Qualification.seal_outputs(self)

    def run(self, phase, command, environment=None):
        self.guard()
        self.phases.append(phase)
        stdout, log = [self.output / (phase + suffix) for suffix in ('.stdout.log', '.log')]
        log.write_text('ordinary production diagnostics\n')
        if phase.startswith('case-'):
            native = phase.endswith('-native')
            record = copy.deepcopy(self.native if native else self.public)
            record['elapsed_micros'] = self.elapsed
            raw = self.raw[:-4] if len(self.phases) - 1 == self.truncate_at else self.raw
            if native:
                target = Path(command[-1])
                target.write_text(json.dumps(record))
                Path(str(target) + '.f32').write_bytes(raw)
                stdout.write_text('')
            else:
                Path(command[-1]).write_bytes(raw)
                stdout.write_text(json.dumps(record))
        elif phase == 'native-lifecycle-list':
            stdout.write_text('\n'.join(test + ': test' for test in LIFECYCLE))
        else:
            test = command[command.index('--exact') + 1]
            stdout.write_text('test ' + test + ' ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n')
        self.guard()
        self.seal_outputs()


class Q8CandidateProducerTest(unittest.TestCase):
    def test_build_is_exact_original_method_with_candidate_verifier(self):
        raw = (TOOLS / 'platform_bounded_windows_native.py').read_bytes()
        derived = derive_build(raw)
        self.assertEqual(derived, (TOOLS / 'windows_q8_candidate_build.py').read_bytes())
        self.assertIn(b'from windows_q8_candidate_verify import verify_completed_source', derived)
        self.assertIn(b"'backend-diar-nemo', '--example', 'nemotron_valid_mel_stream'", derived)
        with self.assertRaises(ValueError): derive_build(raw.replace(b'def build(self):', b'def missing(self):'))
        with self.assertRaises(ValueError): derive_build(raw.replace(b'verify_completed_source(', b'unrecorded_verify('))

    def test_requires_actual_avx2_fma_and_four_logical_processors(self):
        host = {'avx2': True, 'fma': True, 'cpu': [{'NumberOfLogicalProcessors': 4}]}
        require_host(host)
        require_host({**host, 'cpu': {'NumberOfLogicalProcessors': 4}})
        for bad in ({**host, 'avx2': False}, {**host, 'fma': 'true'}, {**host, 'cpu': []},
                    {**host, 'cpu': [{'NumberOfLogicalProcessors': 3}]},
                    {**host, 'cpu': [{'NumberOfLogicalProcessors': True}]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError): require_host(bad)

    def test_production_rejects_each_telemetry_family(self):
        require_uninstrumented('ordinary model log')
        for marker in ('[DEBUG-worker-v1]', '[timing]', '[gen2-cpu-detail]', '[gen2-matrix-batch]'):
            with self.subTest(marker=marker), self.assertRaises(ValueError): require_uninstrumented(marker)

    def test_four_fresh_histories_and_all_seven_lifecycle_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = HistoryFixture(Path(folder))
            q = {'cases': [], 'comparisons': [], 'completed_lifecycle_tests': []}
            persists = []
            run_histories(runner, Path(folder) / 'assets', Path(folder) / 'public.exe', q, lambda: persists.append(True))
            self.assertEqual([c['kind'] for c in q['cases']], ORDER)
            self.assertEqual(runner.phases, ['case-0-native', 'case-1-public', 'case-2-public', 'case-3-native'])
            self.assertTrue(both_pairs_realtime(q['comparisons']))
            self.assertEqual(len(q['comparisons']), 2)
            self.assertTrue(all(c['probabilities']['bytes'] == 192000 for c in q['cases']))
            run_lifecycle(runner, Path(folder) / 'assets', q['cases'][0], q, lambda: persists.append(True))
            self.assertEqual(q['completed_lifecycle_tests'], LIFECYCLE)
            self.assertEqual(len(runner.phases), 12)
            runner.guard()
            self.assertTrue(persists)

    def test_slow_histories_are_preserved_and_lifecycle_still_executes(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = HistoryFixture(Path(folder), elapsed=61_000_000)
            q = {'cases': [], 'comparisons': [], 'completed_lifecycle_tests': []}
            run_histories(runner, Path(folder) / 'assets', Path(folder) / 'public.exe', q, lambda: None)
            self.assertFalse(both_pairs_realtime(q['comparisons']))
            self.assertEqual([c['result']['elapsed_micros'] for c in q['cases']], [61_000_000] * 4)
            run_lifecycle(runner, Path(folder) / 'assets', q['cases'][0], q, lambda: None)
            self.assertEqual(q['completed_lifecycle_tests'], LIFECYCLE)

    def test_incomplete_probability_history_rejects_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = HistoryFixture(Path(folder), truncate_at=1)
            q = {'cases': [], 'comparisons': []}
            with self.assertRaises(ValueError):
                run_histories(runner, Path(folder) / 'assets', Path(folder) / 'public.exe', q, lambda: None)
            self.assertEqual(len(runner.phases), 2)
            self.assertEqual(len(q['cases']), 2)
            self.assertEqual(len(q['comparisons']), 0)
            self.assertEqual(q['cases'][1]['probabilities']['bytes'], 191996)

    def test_sealed_completed_output_mutation_and_symlink_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = HistoryFixture(Path(folder))
            target = runner.output / 'earlier-result.f32'; target.write_bytes(b'original')
            runner.seal_outputs()
            (runner.output / 'commands.json').write_text('mutable command progress')
            runner.seal_outputs()
            self.assertNotIn(str(runner.output / 'commands.json'), runner.bindings)
            target.write_bytes(b'changed')
            with self.assertRaises(ValueError): runner.seal_outputs()
            target.write_bytes(b'original')
            link = runner.output / 'link'
            try: link.symlink_to(target)
            except OSError: self.skipTest('symlink creation unavailable')
            with self.assertRaises(ValueError): runner.seal_outputs()

    def test_lifecycle_missing_listed_or_unexecuted_test_rejects(self):
        for mode in ('missing', 'unexecuted'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                runner = HistoryFixture(Path(folder))
                original_run = runner.run
                def run(name, command, environment=None):
                    original_run(name, command, environment)
                    path = runner.output / (name + '.stdout.log')
                    if name == ('native-lifecycle-list' if mode == 'missing' else 'native-lifecycle-0'):
                        # A controlled command writes a bad result before the caller inspects it.
                        path.write_text('0 tests\n' if mode == 'missing' else 'test result: ok. 0 passed; 0 failed; 1 ignored;\n')
                        runner.bindings[str(path)] = descriptor(path)
                runner.run = run
                probability = runner.output / 'reference-source.f32'; probability.write_bytes(runner.raw)
                q = {'completed_lifecycle_tests': []}
                case = {'result': runner.native, 'probability_path': probability.name}
                with self.assertRaises(ValueError):
                    run_lifecycle(runner, Path(folder) / 'assets', case, q, lambda: None)
                self.assertFalse(q['completed_lifecycle_tests'])

    def test_owned_command_keeps_actual_exit_and_cleanup_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = Qualification.__new__(Qualification)
            runner.output = runner.repo = Path(folder)
            runner.commands = []; runner.guard = None; runner.environment = {}
            runner.verify_protocol = lambda: None
            observation = {'process_tree_reaped': True, 'job_handle_closed': True, 'cleanup_errors': []}
            def execute(*args, **kwargs):
                result = subprocess.CompletedProcess(args[0], 17)
                result.process_tree_reaped, result.job_observation = True, observation
                return result
            with patch('windows_q8_candidate.run_in_job', side_effect=execute):
                with self.assertRaises(subprocess.CalledProcessError): runner.run('controlled-failure', ['controlled.exe'])
            self.assertEqual(runner.commands[0]['returncode'], 17)
            self.assertTrue(runner.commands[0]['process_tree_reaped'])
            self.assertEqual(runner.commands[0]['job_observation'], observation)
            self.assertEqual(json.loads((runner.output / 'commands.json').read_text()), runner.commands)

    def test_protocol_covers_actual_imports_source_and_receipt_inputs(self):
        from freeze_windows_q8_candidate import inputs
        p = inputs(TOOLS)
        actual = json.loads((TOOLS / 'windows-q8-candidate.protocol.json').read_text())
        self.assertEqual(actual, p)
        for name in ('windows_q8_candidate.py', 'windows_q8_candidate_build.py',
                     'windows_q8_candidate_producer_sources.py', 'test_windows_q8_candidate_producer.py',
                     'test_windows_job_v2.py', 'windows_job_v2.py', 'windows-q8-candidate.recipe.json'):
            self.assertIn(name, p['tools'])
        for name in (BOUNDED_RESULT + '/platform-llama-materialized-audit.json',
                     'src/diarization/ffi.rs', 'examples/nemotron_stream.rs', 'tests/diarization_native.rs'):
            self.assertIn(name, p['repository_files'])
        self.assertEqual(p['order'], ORDER)
        self.assertEqual(p['required_lifecycle_tests'], LIFECYCLE)


if __name__ == '__main__': unittest.main()
