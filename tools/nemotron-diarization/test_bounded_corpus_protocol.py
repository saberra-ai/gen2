"""The bounded profile must keep every original slice/oracle/gate and its budget."""
import hashlib
import json
from pathlib import Path
import unittest


class BoundedCorpusProtocolTests(unittest.TestCase):
    def test_exact_corpus_budget_and_original_quality_contract(self):
        tools = Path(__file__).parent
        current = json.loads((tools / 'bounded-corpus.protocol.json').read_text())
        original = json.loads((tools / 'corpus.protocol.json').read_text())
        previous = json.loads((tools / 'valid-mel-corpus.protocol.json').read_text())
        self.assertEqual(current['quality_gates'], original['scoring']['per_slice_gates'])
        self.assertEqual(current['legacy_scoring'], original['scoring'])
        self.assertEqual(current['storage'], previous['storage'])
        self.assertEqual(current['command_timeout_seconds'], previous['command_timeout_seconds'])
        self.assertEqual(current['expected_fresh_child_processes'], 16)
        self.assertEqual(current['expected_scored_pairs'], 8)
        self.assertEqual(current['scoring_environment'], previous['scoring_environment'])
        for name, wanted in previous['inventory'].items():
            if name.startswith(('corpus/', 'oracle/')) or name in ('corpus', 'oracle'):
                self.assertEqual(current['inventory'][name], wanted)
        self.assertEqual(sum(name.endswith('-fp32.npy') for name in current['inventory']), 8)
        self.assertEqual(sum(name.endswith('.reference.json') for name in current['inventory']), 8)

    def test_qualified_helpers_and_exact_tools_bound(self):
        tools = Path(__file__).parent
        current = json.loads((tools / 'bounded-corpus.protocol.json').read_text())
        qualified = json.loads((tools / 'macos-bounded-pnola-app.protocol.json').read_text())
        for name in ('load-probe', 'native-stream', 'public-stream'):
            self.assertEqual(current['inventory']['cpu/bin/' + name], qualified['work_inventory']['bin/' + name])
        self.assertEqual(current['inventory']['assets/Nemotron-3-Diarization.q8_0.gguf'], qualified['work_inventory']['assets/model.gguf'])
        for name in ('qualify_bounded_corpus.py', 'bounded_corpus_metrics.py', 'compare_stream_bounded.py',
                     'score_valid_mel_corpus.py', 'test_bounded_corpus_protocol.py', 'corpus.protocol.json'):
            raw = (tools / name).read_bytes()
            self.assertEqual(current['tools'][name], {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})


if __name__ == '__main__':
    unittest.main()
