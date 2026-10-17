import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from linux_chat_audit import CRITICAL_SYMBOLS, audit_results, ggml_bindings
from qualify_linux_package import RUNTIME_PATH, loader_paths


class LinuxChatAuditTests(unittest.TestCase):
    def test_every_native_ggml_binding_must_stay_in_the_diarization_package(self):
        names = {'libgen2_nemo_linux_native.so', 'libgen2_nemo_linux_ggml-base.so.0.25.1'}
        source = RUNTIME_PATH + '/lib/libgen2_nemo_linux_native.so'
        target = RUNTIME_PATH + '/lib/libgen2_nemo_linux_ggml-base.so.0.25.1'
        trace = '\n'.join(f" 12: binding file {source} [0] to {target} [0]: normal symbol `{symbol}'" for symbol in CRITICAL_SYMBOLS)
        self.assertEqual(len(ggml_bindings(trace, names)), len(CRITICAL_SYMBOLS))
        self.assertEqual(len(ggml_bindings(trace.replace(RUNTIME_PATH, '/work/bin/../relocated runtime ü'), names)), len(CRITICAL_SYMBOLS))
        for changed in [trace.replace(target, '/work/bin/chat-coexistence'),
                        trace.replace(target, '/usr/lib/libggml-base.so.0'),
                        trace.splitlines()[0], '']:
            with self.assertRaises(ValueError):
                ggml_bindings(changed, names)

    def test_chat_openmp_runtime_must_load_from_its_own_declared_test_payload(self):
        runtime = '/work/chat/lib/libgomp.so.1'
        self.assertEqual(loader_paths('12: calling init: ' + runtime, [], [runtime])['bundled'], [runtime])
        with self.assertRaises(ValueError):
            loader_paths('12: calling init: /usr/lib/libgomp.so.1', [], [runtime])

    def fixture(self, root):
        raw = bytes(3 * 8 * 4)
        native = {'audio_samples': 320, 'audio_duration_seconds': .02, 'native_frame_count': 3,
                  'geometry': {'chunk': 9, 'right': 4, 'left': 0, 'fifo': 264, 'cache': 264, 'update': 222},
                  'input_pcm_f32le_sha256': 'fixture-pcm', 'requested_gpu_index': -1,
                  'retained_first_frame': 0, 'retained_rows': 3, 'turns': [], 'elapsed_micros': 1000}
        public = {**native, 'accepted_samples': 320, 'complete': True, 'input_repeats': 1,
                  'retained_probabilities': {'first_frame': 0, 'rows': 3},
                  'retained_probability_sha256': hashlib.sha256(raw).hexdigest(),
                  'model_revision': 'f667ed73aee57d40cc39428eb768b4fd87a0a29e',
                  'runtime_version': 'nemo-speech-asr 0.2.0', 'backend': 'cpu', 'device': None, 'api_call_micros': [0, 100]}
        fingerprints = {'executable': 'fixed binary', 'chat_model': 'fixed model'}
        system = {'gpu_layers': 0, 'threads': 2, 'threads_batch': 2, 'ctx_size': 1024}
        chat = {'text': 'hello', 'api_call_micros': [10, 20], 'offload': {'layers': 0, 'total': 29, 'backend': None, 'device': None}}
        summaries = {}
        for order in ['isolated', 'chat-first', 'diar-first']:
            directory = root / order
            directory.mkdir()
            stages = ['isolated'] if order == 'isolated' else ['both_loaded', 'concurrent', 'after_shutdown', 'after_reload'] + (['before_diarizer'] if order == 'chat-first' else [])
            summary = {'order': order, 'fingerprints': fingerprints, 'settings': {'system': system},
                       'prompt': 'Reply with the single word hello.',
                       'generation': {'max_tokens': 8, 'temperature': 0., 'seed': 0},
                       'chat_outputs': {stage: copy.deepcopy(chat) for stage in stages},
                       'concurrent_intervals': {'chat_micros': [10, 20], 'diarization_micros': [0, 100]},
                       'reservation_after_shutdown_mb': 0}
            summaries[order] = summary
            (directory / 'summary.json').write_text(json.dumps(summary))
            if order != 'isolated':
                for stage in ['first', 'concurrent', 'reloaded']:
                    (directory / (stage + '.json')).write_text(json.dumps(public))
                    (directory / (stage + '.f32')).write_bytes(raw)
        return native, raw, fingerprints, summaries

    def test_changed_chat_missing_stages_nonconcurrent_execution_and_pcm_drift_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native, raw, fingerprints, summaries = self.fixture(root)
            self.assertEqual(len(audit_results(root, native, raw, fingerprints)['comparisons']), 6)
            path = root / 'diar-first/summary.json'
            original = summaries['diar-first']
            mutations = []
            changed = copy.deepcopy(original); changed['chat_outputs']['concurrent']['text'] = 'different'; mutations.append(changed)
            changed = copy.deepcopy(original); del changed['chat_outputs']['after_shutdown']; mutations.append(changed)
            changed = copy.deepcopy(original); changed['concurrent_intervals']['chat_micros'] = [101, 102]; mutations.append(changed)
            changed = copy.deepcopy(original); changed['reservation_after_shutdown_mb'] = 1024; mutations.append(changed)
            changed = copy.deepcopy(original); changed['fingerprints'] = {'executable': 'other'}; mutations.append(changed)
            changed = copy.deepcopy(original); changed['chat_outputs']['both_loaded']['offload']['layers'] = 1; mutations.append(changed)
            for changed in mutations:
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    audit_results(root, native, raw, fingerprints)
            # Preparation/serialization wrapper windows overlap, but the
            # actual diarize/generate calls do not. This must never qualify.
            changed = copy.deepcopy(original)
            changed['concurrent_intervals']['chat_micros'] = [101, 102]
            changed['chat_outputs']['concurrent']['api_call_micros'] = [101, 102]
            changed['wrapper_intervals'] = {'chat': [0, 200], 'diarization': [0, 300]}
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'concurrently'):
                audit_results(root, native, raw, fingerprints)
            # An earlier wrapper-based summary could claim overlap even though
            # the separately exported inference windows are disjoint.
            changed = copy.deepcopy(original)
            changed['chat_outputs']['concurrent']['api_call_micros'] = [101, 102]
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'actual API observation'):
                audit_results(root, native, raw, fingerprints)
            path.write_text(json.dumps(original))
            concurrent = root / 'diar-first/concurrent.json'
            value = json.loads(concurrent.read_text())
            value['api_call_micros'] = [30, 40]
            concurrent.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'actual API observation'):
                audit_results(root, native, raw, fingerprints)
            value['api_call_micros'] = [0, 100]
            concurrent.write_text(json.dumps(value))
            run = root / 'diar-first/first.json'
            value = json.loads(run.read_text()); value['input_pcm_f32le_sha256'] = 'other-pcm'; run.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'PCM'):
                audit_results(root, native, raw, fingerprints)


if __name__ == '__main__':
    unittest.main()
