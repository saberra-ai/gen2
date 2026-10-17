"""Reject confounded tile transforms, incomplete vectors and weak timing claims."""
import hashlib
import json
from pathlib import Path
import struct
import unittest

from q8_tile_source import ORIGINAL, PATH, transform, CLASS, OLD, NEW
from windows_q8_tile_metrics import ORDER, REPETITIONS, SHAPES, timing_evidence, validate_output


class Q8TileTest(unittest.TestCase):
    def fixture(self, variant='control', workers=4):
        rows, raw = [], []
        for index, (m,n,k) in enumerate(SHAPES):
            rows.append({'shape': index, 'm':m, 'n':n, 'k':k, 'workers':workers, 'workers_started':workers,
                'unsupported_support_controls':3,
                'variant':variant, 'bytes':m*n*4, 'warmups':1, 'repetitions':REPETITIONS, 'timed_ns':[1000]*REPETITIONS})
            raw.append(struct.pack('<f', .25) * (m*n))
        return rows, raw

    def test_exact_canonical_source_transform_changes_only_one_dispatch(self):
        root = Path(__file__).resolve().parents[2] / 'benches/results/nemotron-diarization/windows-x86_64/2026-10-17-math-errno'
        audit = json.loads((root/'evidence-control/platform-llama-materialized-audit.json').read_text())
        self.assertEqual(audit['files'][PATH], ORIGINAL)
        raw = (Path(__file__).resolve().parent / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
        changed = transform(raw)
        self.assertEqual(changed, raw[:raw.index(CLASS)] + raw[raw.index(CLASS):].replace(OLD, NEW, 1))
        self.assertEqual({'bytes':len(changed),'sha256':hashlib.sha256(changed).hexdigest()},
            {'bytes':163351,'sha256':'304359cacc2bf8dd2fcdd5e07812ebb4df3930d69101f7366bc6b2664b71f195'})
        for raw in (b'', CLASS+OLD, b'unrelated'):
            with self.assertRaises(ValueError): transform(raw)

    def test_all_shape_vectors_and_worker_counts_validate(self):
        for workers in (1,2,4):
            for variant in ('control','tile2'):
                rows, raw = self.fixture(variant, workers)
                value = validate_output('\n'.join(map(json.dumps,rows)), variant, workers, raw)
                self.assertEqual(len(value['outputs']),7)

    def test_partial_reordered_wrong_shape_and_worker_rows_reject(self):
        rows, raw = self.fixture()
        invalid = [rows[:-1], list(reversed(rows)), [{**rows[0],'workers_started':3},*rows[1:]],
            [{**rows[0],'unsupported_support_controls':2},*rows[1:]],
            [{**rows[0],'workers':True},*rows[1:]], [{**rows[0],'m':2},*rows[1:]],
            [{**rows[0],'variant':'tile2'},*rows[1:]], [{**rows[0],'extra':1},*rows[1:]]]
        for changed in invalid:
            with self.assertRaises(ValueError): validate_output('\n'.join(map(json.dumps,changed)),'control',4,raw)

    def test_nonfinite_zero_unwritten_and_truncated_vectors_reject(self):
        rows, raw = self.fixture()
        for first in [raw[0][:-1], struct.pack('<2f',0,0), struct.pack('<2f',float('nan'),1),
                      struct.pack('<2f',float('inf'),1),struct.pack('<2f',2**80,1)]:
            with self.assertRaises(ValueError): validate_output('\n'.join(map(json.dumps,rows)),'control',4,[first,*raw[1:]])

    def test_bad_or_incomplete_timing_rows_reject(self):
        rows, raw = self.fixture()
        for times in [[1]*7,[0]*8,[True]*8,[-1]*8,[1.0]*8,[600000000001]*8]:
            changed = [{**rows[0],'timed_ns':times},*rows[1:]]
            with self.assertRaises(ValueError): validate_output('\n'.join(map(json.dumps,changed)),'control',4,raw)

    def cases(self, times):
        result=[]
        for variant,time in zip(ORDER,times):
            rows,_=self.fixture(variant)
            for row in rows: row['timed_ns']=[time]*REPETITIONS
            result.append({'variant':variant,'workers':4,'observed':{'rows':rows}})
        return result

    def test_integer_predicates_include_both_surrounding_controls(self):
        self.assertTrue(timing_evidence(self.cases([1000,950,950,1100]))['kernel_speedup_prediction_supported'])
        self.assertFalse(timing_evidence(self.cases([1000,951,950,1000]))['kernel_speedup_prediction_supported'])
        self.assertFalse(timing_evidence(self.cases([1000,900,900,1101]))['kernel_speedup_prediction_supported'])
        self.assertFalse(timing_evidence(self.cases([1100,1000,1000,1000]))['kernel_speedup_prediction_supported'])

    def test_every_primary_shape_must_pass_without_averaging(self):
        cases=self.cases([1000,900,900,1000])
        cases[1]['observed']['rows'][5]['timed_ns']=[1000]*REPETITIONS
        self.assertFalse(timing_evidence(cases)['kernel_speedup_prediction_supported'])
        cases[1]['workers']=2
        with self.assertRaises(ValueError): timing_evidence(cases)


if __name__=='__main__': unittest.main()
