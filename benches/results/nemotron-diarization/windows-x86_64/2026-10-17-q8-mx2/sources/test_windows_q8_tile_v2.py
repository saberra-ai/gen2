"""Independent half conversion and versioned support rejection controls."""
import json
import math
import struct
import unittest

from windows_q8_tile_metrics_v2 import GUARD_CODES, REPETITIONS, SHAPES, admit_guard, admit_half_table, validate_output


def half_table():
    return b''.join(struct.pack('<f', struct.unpack('<e', struct.pack('<H', index))[0]) for index in range(65536))


class Q8TileV2Test(unittest.TestCase):
    def test_all_binary16_values_and_signed_zero_are_checked(self):
        raw=half_table()
        result=admit_half_table(raw)
        self.assertEqual(result['entries'],65536)
        self.assertEqual(result['finite_entries_verified'],63488)
        self.assertEqual(raw[32768*4:32768*4+4],struct.pack('<I',0x80000000))

    def test_truncated_zero_and_corrupted_finite_tables_reject(self):
        raw=half_table()
        for changed in [raw[:-1],bytes(len(raw)),raw[:4]+struct.pack('<f',1.0)+raw[8:],
                        raw[:32768*4]+bytes(4)+raw[32769*4:]]:
            with self.assertRaises(ValueError):admit_half_table(changed)

    def test_infinity_and_nan_classification_are_not_ignored(self):
        raw=half_table()
        for index in [0x7c00,0xfc00,0x7c01,0xfe00]:
            changed=raw[:index*4]+struct.pack('<f',1.0)+raw[(index+1)*4:]
            with self.assertRaises(ValueError):admit_half_table(changed)

    def test_each_guard_requires_its_exact_exit_and_only_its_diagnostic(self):
        for name,code in GUARD_CODES.items():
            stderr=('GEN2_Q8_UNSUPPORTED_'+name+'\n').encode()
            self.assertTrue(admit_guard(name,code,b'',stderr)['rejection_observed'])
            self.assertTrue(admit_guard(name,code,b'',stderr.replace(b'\n',b'\r\n'))['rejection_observed'])
            for actual,out,err in [(0,b'',stderr),(code+1,b'',stderr),(True,b'',stderr),
                                   (code,b'extra',stderr),(code,b'',b''),(code,b'',stderr+b'extra')]:
                with self.assertRaises(ValueError):admit_guard(name,actual,out,err)
        with self.assertRaises(ValueError):admit_guard('unknown',73,b'',b'')

    def fixture(self,variant='control',workers=4):
        rows=[];outputs=[]
        for index,(m,n,k) in enumerate(SHAPES):
            rows.append({'shape':index,'m':m,'n':n,'k':k,'workers':workers,'workers_started':workers,
                         'half_table_entries':65536,'finite_half_entries_checked':63488,
                         'variant':variant,'bytes':m*n*4,'warmups':1,'repetitions':REPETITIONS,'timed_ns':[1000]*REPETITIONS})
            outputs.append(struct.pack('<f',.25)*(m*n))
        return rows,outputs

    def test_all_original_shapes_and_workers_keep_complete_output_admission(self):
        for variant in ('control','tile2'):
            for workers in (1,2,4):
                rows,outputs=self.fixture(variant,workers)
                result=validate_output('\n'.join(map(json.dumps,rows)),variant,workers,outputs)
                self.assertEqual(len(result['outputs']),7)

    def test_missing_wrong_or_old_harness_admission_fields_reject(self):
        rows,outputs=self.fixture()
        for patch in [{'half_table_entries':65535},{'finite_half_entries_checked':False},
                      {'unsupported_support_controls':3}]:
            changed=[{**rows[0],**patch},*rows[1:]]
            with self.assertRaises(ValueError):validate_output('\n'.join(map(json.dumps,changed)),'control',4,outputs)
        rows[0].pop('half_table_entries')
        with self.assertRaises(ValueError):validate_output('\n'.join(map(json.dumps,rows)),'control',4,outputs)

    def test_reordered_rows_and_unwritten_or_nonfinite_vectors_still_reject(self):
        rows,outputs=self.fixture()
        with self.assertRaises(ValueError):validate_output('\n'.join(map(json.dumps,reversed(rows))),'control',4,outputs)
        for raw in [outputs[0][:-1],struct.pack('<2f',2**80,1),struct.pack('<2f',math.nan,1)]:
            with self.assertRaises(ValueError):validate_output('\n'.join(map(json.dumps,rows)),'control',4,[raw,*outputs[1:]])


if __name__=='__main__':unittest.main()
