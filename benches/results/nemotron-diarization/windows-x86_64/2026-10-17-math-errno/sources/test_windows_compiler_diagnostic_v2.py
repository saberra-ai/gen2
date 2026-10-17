"""Controller-only derivation and retained status/cleanup observations."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from windows_compiler_diagnostic_v2 import Diagnostic
from windows_compiler_v2_sources import derive_runner


class CompilerV2Test(unittest.TestCase):
    def test_only_exact_controller_sites_change(self):
        tools = Path(__file__).resolve().parent
        original = (tools / 'windows_compiler_diagnostic.py').read_bytes()
        self.assertEqual(derive_runner(original), (tools / 'windows_compiler_diagnostic_v2.py').read_bytes())
        for invalid in [original + original, original.replace(b'from windows_job import run_in_job', b'unknown')]:
            with self.assertRaises(ValueError):
                derive_runner(invalid)

    def test_observed_status_and_cleanup_survive_failed_commands(self):
        reaped = subprocess.CompletedProcess(['fixture-only'], 9)
        reaped.process_tree_reaped = True
        reaped.job_observation = {'observed_worker_exit_code': 9, 'process_tree_reaped': True}
        timeout = subprocess.TimeoutExpired(['fixture-only'], 2)
        timeout.process_tree_reaped = True
        timeout.job_observation = {'observed_worker_exit_code': None, 'process_tree_reaped': True}
        uncertain = RuntimeError('unverified cleanup')
        uncertain.process_tree_reaped = False
        uncertain.job_observation = {'observed_worker_exit_code': 0, 'process_tree_reaped': False}
        for value, error_type in [(reaped, subprocess.CalledProcessError), (timeout, subprocess.TimeoutExpired),
                                   (uncertain, RuntimeError)]:
            with self.subTest(failure=type(value).__name__), tempfile.TemporaryDirectory() as temporary:
                runner = object.__new__(Diagnostic)
                runner.output = runner.repo = Path(temporary)
                runner.kind, runner.commands, runner.environment = 'msvc', [], {}
                runner.frozen = lambda: None
                arguments = {'side_effect': value} if isinstance(value, BaseException) else {'return_value': value}
                with patch('windows_compiler_diagnostic_v2.run_in_job', **arguments), self.assertRaises(error_type):
                    runner.run('fixture-command', ['fixture-only'])
                record = runner.commands[0]
                self.assertEqual(record['job_observation'], value.job_observation)
                self.assertEqual(record['process_tree_reaped'], value.process_tree_reaped)
                self.assertIn('error', record)
                if isinstance(value, subprocess.CompletedProcess):
                    self.assertEqual(record['returncode'], 9)


if __name__ == '__main__':
    unittest.main()
