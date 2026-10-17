"""Synthetic receiver controls; no Docker, compiler, Windows API or model calls."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

from build_windows_failed_load_probe import validate_observer
from platform_bounded_metrics import VERSION
from qualify_bounded_windows_package import Qualification, prepare_model_free_inputs, LIFECYCLE_GATES
from windows_failed_load_metrics import validate_failed_loads
import test_stream_comparison_valid_mel as fixtures


def failed_loads():
    return {'failed_model_construction_attempts': 101, 'valid_models_loaded': 0, 'frontend_or_neural_calls': 0,
            'thread_growth_allowance': 2, 'maximum_requested_settling_wait_milliseconds': 100,
            'immediate_thread_counts': [5] * 101, 'threads_after_each_failure': [3] * 101,
            'requested_settling_wait_milliseconds': [10] * 101}


class BoundedWindowsReceiverTests(unittest.TestCase):
    def test_complete_failed_load_history_rejects_growth_truncation_and_false_zeroes(self):
        value = failed_loads()
        self.assertEqual(validate_failed_loads(value), value)
        for field, replacement in [('valid_models_loaded', False), ('frontend_or_neural_calls', 1),
                ('immediate_thread_counts', [3] * 100), ('threads_after_each_failure', [3] + [6] * 100),
                ('requested_settling_wait_milliseconds', [0] * 101), ('thread_growth_allowance', 3)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_failed_loads({**value, field: replacement})

    def test_observer_requires_known_worker_presence_retirement_and_valid_counts(self):
        value = {'known_worker_observed': True, 'known_worker_retired': True, 'valid_models_loaded': 0,
                 'frontend_or_neural_calls': 0, 'baseline_threads': 1, 'active_threads': 2,
                 'settled_threads': 1, 'requested_retirement_wait_ms': 10}
        validate_observer(value)
        for name, bad in [('known_worker_observed', False), ('known_worker_retired', False),
                          ('active_threads', 0), ('valid_models_loaded', False), ('requested_retirement_wait_ms', 110)]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_observer({**value, name: bad})

    def test_model_free_copy_excludes_assets_and_extra_mounts_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            (source / 'assets').mkdir(parents=True)
            (source / 'assets/model.gguf').write_bytes(b'unit fixture, no weights')
            (source / 'reference').mkdir()
            copied = root / 'copy'
            shutil.copytree(source, copied)
            prepare_model_free_inputs(copied, ())
            self.assertFalse((copied / 'assets').exists())
            self.assertFalse((copied / 'reference').exists())
            self.assertEqual((source / 'assets/model.gguf').read_bytes(), b'unit fixture, no weights')
            with self.assertRaisesRegex(ValueError, 'asset mounts'):
                prepare_model_free_inputs(copied, [('outside', 'C:\\chat')])
            (copied / 'unexpected.GGUF').write_bytes(b'unit fixture')
            with self.assertRaisesRegex(ValueError, 'model payload'):
                prepare_model_free_inputs(copied, ())

    def receiver_fixture(self, root, elapsed=59000000, history=None):
        receiver = Qualification.__new__(Qualification)
        receiver.output = root
        receiver.work = root / 'work'
        receiver.work.mkdir()
        receiver.runtime = root / 'runtime'
        receiver.package = {'entry_library': 'private.dll'}
        receiver.central_crt = None
        receiver.tools = Path(__file__).resolve().parent
        receiver.receipt = {'offline_container_gates_passed': False, 'realtime_gate_passed': False,
                            'failed_load_cleanup_passed': False}
        raw = struct.pack('<f', .5) * 6000 * 8
        public, native, _ = fixtures.ValidMelReference().fixture()
        pcm = json.loads((receiver.tools / 'bounded-windows-native.protocol.json').read_text())['cases']['actual_pcm_sha256']
        shared = {'runtime_version': VERSION, 'input_pcm_f32le_sha256': pcm, 'input_repeats': 1,
                  'audio_duration_seconds': 60, 'native_frame_count': 6000, 'elapsed_micros': elapsed}
        public.update(**shared, backend='cpu', device=None, accepted_samples=960000,
            retained_probabilities={'first_frame': 0, 'rows': 6000}, retained_probability_sha256=hashlib.sha256(raw).hexdigest())
        native.update(**shared, requested_gpu_index=-1, audio_samples=960000, retained_first_frame=0, retained_rows=6000)
        calls = []

        def container(name, executable, arguments, **kwargs):
            calls.append((name, executable, arguments, kwargs))
            results = root / name / 'results'
            results.mkdir(parents=True)
            if name == 'failed-loads':
                (results / 'child.stdout.log').write_text(json.dumps(failed_loads() if history is None else history))
            elif name == 'native':
                (results / 'native.json').write_text(json.dumps(native))
                (results / 'native.json.f32').write_bytes(raw)
            elif name == 'public':
                (results / 'child.stdout.log').write_text(json.dumps(public))
                (results / 'public.f32').write_bytes(raw)
            elif name == 'lifecycle-list':
                (results / 'child.stdout.log').write_text(''.join(name + ': test\n' for name in sorted(LIFECYCLE_GATES)))
            elif name == 'lifecycle':
                (results / 'child.stdout.log').write_text(''.join('test ' + name + ' ... ok\n' for name in sorted(LIFECYCLE_GATES)))
            elif name == 'missing-private-library':
                (results / 'child.stderr.log').write_text('os error 126')
            else:
                raise AssertionError('unexpected synthetic stage: ' + name)
            return results
        receiver.container = container
        return receiver, calls

    def test_failed_load_control_precedes_all_valid_model_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            receiver, calls = self.receiver_fixture(Path(directory))
            with patch('qualify_bounded_windows_package.verify', return_value=receiver.package):
                receiver.exercise()
            self.assertEqual(calls[0][0], 'failed-loads')
            self.assertEqual(calls[0][2], [VERSION, 'C:\\results\\missing-model'])
            self.assertIs(calls[0][3]['model_free'], True)
            self.assertIs(receiver.receipt['failed_load_cleanup_passed'], True)
            self.assertIs(receiver.receipt['offline_container_gates_passed'], True)
            self.assertIs(receiver.receipt['realtime_gate_passed'], True)

    def test_observed_slow_rtf_fails_after_retaining_other_runtime_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            receiver, calls = self.receiver_fixture(Path(directory), elapsed=60000000)
            with patch('qualify_bounded_windows_package.verify', return_value=receiver.package):
                with self.assertRaisesRegex(ValueError, 'RTF must each be below one'):
                    receiver.exercise()
            self.assertEqual(calls[-1][0], 'missing-private-library')
            self.assertIs(receiver.receipt['offline_container_gates_passed'], False)
            self.assertIs(receiver.receipt['realtime_gate_passed'], False)
            self.assertIn('native_public_comparison', receiver.receipt)

    def test_growing_failed_load_history_stops_before_native_inference(self):
        history = failed_loads()
        history['threads_after_each_failure'][-1] = 6
        with tempfile.TemporaryDirectory() as directory:
            receiver, calls = self.receiver_fixture(Path(directory), history=history)
            with self.assertRaisesRegex(ValueError, 'growing'):
                receiver.exercise()
            self.assertEqual([call[0] for call in calls], ['failed-loads'])
            self.assertIs(receiver.receipt['offline_container_gates_passed'], False)


if __name__ == '__main__':
    unittest.main()
