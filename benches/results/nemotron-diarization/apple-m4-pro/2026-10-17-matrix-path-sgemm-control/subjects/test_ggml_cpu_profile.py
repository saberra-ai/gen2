"""Reject incomplete/overlapping profiling evidence before interpreting it."""
import unittest

from ggml_cpu_profile import environment, summarize
from ggml_cpu_profile_source import transform


def fixture():
    lines = []
    for thread in range(4):
        lines += [f'[DEBUG-cpu-ops-v1] graph pool=0x1234 generation=8 thread={thread} threads=4 nodes=5 roots=2 fused=1 skipped=2 topology=0123456789abcdef execution=fedcba9876543210 elapsed_us=100 final_wait_us=4',
                  f'[DEBUG-cpu-ops-v1] op pool=0x1234 generation=8 thread={thread} id=29 name=MUL_MAT roots=1 fused=0 kernel_us=60 barrier_us=5',
                  f'[DEBUG-cpu-ops-v1] op pool=0x1234 generation=8 thread={thread} id=25 name=RMS_NORM roots=1 fused=1 kernel_us=20 barrier_us=1']
    return lines


class CpuProfileTest(unittest.TestCase):
    def summarize(self, lines, threads=4):
        return summarize('\n'.join(lines), True, expected_graphs=1, expected_threads=threads)

    def test_environment_removes_inherited_case_variants(self):
        self.assertEqual(environment({'gen2_diar_cpu_profile': '1', 'PATH': '/bin'}, False), {'PATH': '/bin'})
        self.assertEqual(environment({'GEN2_DIAR_CPU_PROFILE': 'bad'}, True), {'GEN2_DIAR_CPU_PROFILE': '1'})
        for value in ('1', 1, None):
            with self.assertRaises(ValueError):
                environment({}, value)

    def test_complete_workers_keep_overlapping_times_separate(self):
        result = self.summarize(fixture())
        self.assertEqual(result['graph_count'], 1)
        self.assertEqual(len(result['graphs'][0]['workers']), 4)
        self.assertEqual(result['graphs'][0]['workers'][0]['fused'], 1)
        self.assertNotIn('total_wall_us', result)
        self.assertEqual(self.summarize(list(reversed(fixture())))['graph_count'], 1)

    def test_missing_duplicate_or_malformed_coverage_rejects(self):
        lines = fixture()
        for changed in [lines[:-1], lines[:3], lines + [lines[0]], lines + [lines[1]],
                        lines + ['noise [DEBUG-cpu-ops-v1] incomplete']]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.summarize(changed)

    def test_worker_topology_fusion_and_operation_identity_must_agree(self):
        for old, new in [('topology=0123456789abcdef', 'topology=0123456789abcdee'),
                         ('execution=fedcba9876543210', 'execution=fedcba9876543211'),
                         ('name=RMS_NORM', 'name=SOFT_MAX'), ('roots=1 fused=1', 'roots=1 fused=0')]:
            changed = fixture()
            changed[-3:] = [line.replace(old, new) for line in changed[-3:]]
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.summarize(changed)

    def test_enclosing_intervals_counts_and_bounds_are_enforced(self):
        for old, new in [('kernel_us=60', 'kernel_us=160'), ('nodes=5', 'nodes=4'),
                         ('threads=4', 'threads=3'), ('elapsed_us=100', 'elapsed_us=3'),
                         ('pool=0x1234', 'pool=0x0'), ('barrier_us=5', 'barrier_us=-1')]:
            changed = [line.replace(old, new) for line in fixture()]
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.summarize(changed)

    def test_globally_consistent_unknown_or_mislabeled_operation_rejects(self):
        for old, new in [('id=29 name=MUL_MAT', 'id=1023 name=MUL_MAT'),
                         ('id=29 name=MUL_MAT', 'id=23 name=MUL_MAT'),
                         ('id=25 name=RMS_NORM', 'id=25 name=SOFT_MAX')]:
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.summarize([line.replace(old, new) for line in fixture()])

    def test_control_has_no_profile_rows_and_source_transform_requires_exact_file(self):
        self.assertEqual(summarize('normal output', False, expected_graphs=3, expected_threads=4),
                         {'enabled': False, 'graphs': []})
        with self.assertRaises(ValueError):
            summarize('\n'.join(fixture()), False, expected_graphs=1, expected_threads=4)
        with self.assertRaises(ValueError):
            transform(b'wrong CPU source')


if __name__ == '__main__':
    unittest.main()
