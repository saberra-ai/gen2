"""Sampling-target safety and an actual model-free macOS sampling control."""
import os
from pathlib import Path
import platform
import signal
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from profile_macos_portable_cpu import CONTROL_SOURCE, sample_target, traced_pid


class SamplingTargetTests(unittest.TestCase):
    def test_redacted_sample_path_still_requires_exact_target_uuid(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stderr, report = root / 'stderr', root / 'sample'
            stderr.write_text('dyld[123]: <ABC-123> /tmp/target\n')
            for reported_uuid, expected_error in [('ABC-123', False), ('DEF-456', True)]:
                report.write_text('Process: target [123]\nPath: /private/*/target\nBinary Images:\n'
                                  '0x1000 - 0x2000 +target (0) <' + reported_uuid + '> /private/*/target\n')
                result = {}
                with patch('profile_macos_portable_cpu.subprocess.check_output', return_value='/tmp/target'), \
                        patch('profile_macos_portable_cpu.subprocess.run',
                              return_value=SimpleNamespace(returncode=0, stdout='', stderr='')):
                    sample_target(stderr, Path('/tmp/target'), report, result,
                                  threading.Event(), duration=1)
                self.assertEqual('error' in result, expected_error, result)

    def test_ambiguous_trace_cannot_select_a_process(self):
        self.assertIsNone(traced_pid('ordinary output'))
        self.assertEqual(traced_pid('dyld[123]: <ABC-123> /tmp/target\n'), 123)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            traced_pid('dyld[123]: <ABC> /tmp/target\ndyld[456]: <DEF> /tmp/other\n')

    def test_stale_pid_cannot_sample_another_executable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stderr = root / 'stderr'
            stderr.write_text('dyld[123]: <ABC> /tmp/target\n')
            result = {}
            with patch('profile_macos_portable_cpu.subprocess.check_output', return_value='/tmp/other'), \
                    patch('profile_macos_portable_cpu.subprocess.run') as sampler:
                sample_target(stderr, Path('/tmp/target'), root / 'sample', result,
                              threading.Event(), duration=1)
            self.assertIn('executable differs', result['error'])
            sampler.assert_not_called()

    def test_process_identity_timeout_stops_before_sampling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stderr = root / 'stderr'
            stderr.write_text('dyld[123]: <ABC> /tmp/target\n')
            result = {}
            with patch('profile_macos_portable_cpu.subprocess.check_output',
                       side_effect=subprocess.TimeoutExpired(['ps'], 5)) as identity, \
                    patch('profile_macos_portable_cpu.subprocess.run') as sampler:
                sample_target(stderr, Path('/tmp/target'), root / 'sample', result,
                              threading.Event(), duration=1)
            self.assertIn('TimeoutExpired', result['error'])
            self.assertEqual(identity.call_args.kwargs['timeout'], 5)
            sampler.assert_not_called()

    @unittest.skipUnless(platform.system() == 'Darwin' and platform.machine() == 'arm64',
                         'actual Apple Silicon loader and sampler control')
    def test_actual_sandboxed_control_is_sampled_without_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            source, target, stderr = root / 'control.c', root / 'control', root / 'stderr'
            source.write_text(CONTROL_SOURCE)
            subprocess.run(['/usr/bin/clang', '-arch', 'arm64', str(source), '-o', str(target)], check=True)
            with stderr.open('wb') as log:
                process = subprocess.Popen(['/usr/bin/sandbox-exec', '-p',
                    '(version 1)(allow default)(deny network*)', '/usr/bin/env',
                    'DYLD_PRINT_LIBRARIES=1', str(target)], stdout=subprocess.DEVNULL,
                    stderr=log, start_new_session=True)
                try:
                    result = {}
                    sample_target(stderr, target, root / 'sample.txt', result,
                                  threading.Event(), duration=1)
                    self.assertNotIn('error', result, result)
                    self.assertEqual(result['pid'], process.pid)
                    self.assertEqual(result['executable'], str(target))
                    self.assertEqual(result['returncode'], 0)
                    self.assertGreater(result['report']['bytes'], 0)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
