import hashlib
from pathlib import Path
import unittest
from ggml_matrix_profile import environment,summarize
from ggml_matrix_profile_source import transform
from test_ggml_cpu_profile import fixture as cpu_fixture


def fixture():
    rows=cpu_fixture()
    for thread in range(4):
        rows.append(f'[DEBUG-matrix-paths-v1] matrix pool=0x1234 generation=8 thread={thread} node=2 types=0,0,0 names=f32,f32,f32 a=64,64,1,1 b=64,32,1,1 c=64,32,1,1 sa=4,256,16384,16384 sb=4,256,8192,8192 sc=4,256,8192,8192 dot=0 route=2 first=1,1,0,55 second=0,0,0,0 conversion=0,0 inner_wait_us=0 chunks=0,0 kernel_us=60')
    return rows


class MatrixProfileTest(unittest.TestCase):
    def parse(self,rows):return summarize('\n'.join(rows),True,expected_graphs=1,expected_threads=4,expected_matrices=1)

    def test_complete_worker_walls_and_metadata_remain_separate(self):
        result=self.parse(fixture())
        self.assertEqual(len(result['graphs'][0]['workers']),4)
        self.assertEqual(result['graphs'][0]['workers'][0]['matrices'][0]['route'],2)
        self.assertNotIn('total_wall_us',result)
        self.parse(list(reversed(fixture())))

    def test_environment_and_disabled_modes_isolate_both_observers(self):
        self.assertEqual(environment({'gen2_diar_cpu_profile':'1','gen2_diar_matrix_profile':'1','PATH':'x'},False),{'PATH':'x'})
        self.assertEqual(environment({},True),{'GEN2_DIAR_MATRIX_PROFILE':'1'})
        self.assertFalse(summarize('',False,expected_graphs=1,expected_threads=4,expected_matrices=1)['enabled'])
        for enabled in [1,None,'1']:
            with self.assertRaises(ValueError):environment({},enabled)
        with self.assertRaises(ValueError):summarize('\n'.join(fixture()),False,expected_graphs=1,expected_threads=4,expected_matrices=1)

    def test_missing_duplicate_or_overflow_rows_reject(self):
        rows=fixture()
        for bad in [rows[:-1],rows+[rows[-1]],rows+['[DEBUG-matrix-paths-v1] overflow']]:
            with self.assertRaises(ValueError):self.parse(bad)

    def test_metadata_root_counts_and_enclosing_intervals_must_match(self):
        for old,new in [('node=2','node=7'),('kernel_us=60','kernel_us=59'),('first=1,1,0,55','first=1,1,0,65'),
                        ('c=64,32,1,1','c=63,32,1,1'),('names=f32,f32,f32','names=q8_0,f32,f32'),
                        ('route=2','route=0'),('first=1,1,0,55','first=1,0,0,0')]:
            with self.subTest(old=old),self.assertRaises(ValueError):self.parse([r.replace(old,new) if r.startswith('[DEBUG-matrix-paths-v1]') else r for r in fixture()])

    def test_converted_and_fallback_paths_preserve_actual_attempts(self):
        rows=[r.replace('types=0,0,0','types=8,0,0').replace('names=f32,f32,f32','names=q8_0,f32,f32')
               .replace('dot=0 route=2','dot=8 route=4').replace('first=1,1,0,55','first=1,0,1,1')
               .replace('second=0,0,0,0','second=1,1,0,40').replace('conversion=0,0','conversion=1,5')for r in fixture()]
        self.parse(rows)
        fallback=[r.replace('route=4','route=5').replace('second=1,1,0,40','second=1,0,1,1')
                  .replace('chunks=0,0','chunks=1,40')for r in rows]
        self.parse(fallback)
        with self.assertRaises(ValueError):self.parse([r.replace('conversion=1,5','conversion=0,5')for r in rows])
        for stage,old in [('first','1,0,1,1'),('second','1,0,1,1')]:
            with self.subTest(stage=stage),self.assertRaises(ValueError):
                self.parse([r.replace(stage+'='+old,stage+'=2,1,1,1')for r in fallback])

    def test_legitimate_worker_local_paths_and_chunk_counts_are_not_forced_equal(self):
        rows=fixture();rows[-1]=rows[-1].replace('route=2','route=5').replace('first=1,1,0,55','first=1,0,1,1').replace('chunks=0,0','chunks=1,40')
        self.parse(rows)
        rows[-1]=rows[-1].replace('sa=4,256,16384,16384','sa=4,512,32768,32768')
        with self.assertRaises(ValueError):self.parse(rows)

    def test_exact_source_fixture_transform_preserves_barriers_and_dispatch_calls(self):
        raw=(Path(__file__).parent/'fixtures/matrix-paths-ggml-cpu.c').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),'d4dd2c161a3f5c0ed45216683de99dd9b13b93330d81c992f4c4163fea904a04')
        changed=transform(raw)
        self.assertEqual(hashlib.sha256(changed).hexdigest(),'93551fb37e0ec9ced9c4f3f0adfe87c4d04a873aec8fb23338417296bbb3f8f2')
        self.assertEqual(raw.count(b'ggml_barrier('),changed.count(b'ggml_barrier('))
        self.assertEqual(changed.count(b'if (!gen2_matrix_sgemm('),2)
        self.assertEqual(raw.count(b'ggml_cpu_extra_compute_forward(params, tensor)'),changed.count(b'ggml_cpu_extra_compute_forward(params, tensor)'))
        self.assertEqual(raw.count(b'ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan)'),changed.count(b'ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan)'))
        self.assertEqual(changed.count(b'gen2_matrix_current = gen2_saved_matrix;'),2)
        with self.assertRaises(ValueError):transform(b'wrong source')


if __name__=='__main__':unittest.main()
