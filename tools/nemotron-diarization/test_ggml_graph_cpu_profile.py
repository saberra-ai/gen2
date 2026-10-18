"""Compiled graph-clock insertion controls and exact worker/trace admission."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from ggml_graph_cpu_profile import environment, summarize
from ggml_graph_cpu_profile_source import BEGIN, DECLARATIONS, END, HELPERS, transform
from ggml_matrix_batch_profile_source import transform as batch_transform
from test_ggml_matrix_profile import fixture as wall_fixture

HARNESS = br'''
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#define GEN2_GRAPH_CPU_CLOCK_TEST 1
typedef struct { uint32_t dwLowDateTime, dwHighDateTime; } FILETIME;
static int mode, calls, ids;
static void * GetCurrentThread(void) { return (void *)(uintptr_t)1; }
static unsigned long GetCurrentThreadId(void) { ids++; return 42; }
static unsigned GetLastError(void) { return 123; }
static int GetThreadTimes(void * thread, FILETIME * created, FILETIME * exited, FILETIME * kernel, FILETIME * user) {
    (void)created; (void)exited;
    if (thread != (void *)(uintptr_t)1) exit(10);
    calls++;
    if ((mode == 1 && calls == 1) || (mode == 2 && calls == 2)) return 0;
    kernel->dwHighDateTime = calls == 1 ? 1 : 2;
    kernel->dwLowDateTime = calls == 1 ? 2 : 4;
    user->dwHighDateTime = calls == 1 ? 3 : 4;
    user->dwLowDateTime = calls == 1 ? 4 : 9;
    if (mode == 3 && calls == 2) {
        kernel->dwHighDateTime = 0;
        user->dwHighDateTime = 0;
    }
    if (mode == 4) {
        kernel->dwHighDateTime = 1; kernel->dwLowDateTime = 2;
        user->dwHighDateTime = 3; user->dwLowDateTime = 4;
    }
    return 1;
}
'''
FUNCTION = br'''
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    mode = atoi(argv[1]);
    const int enabled = atoi(argv[2]);
    void * tp = (void *)(uintptr_t)1;
    struct worker_state { int ith; } actual_state = {0};
    const struct worker_state * state = &actual_state;
    const int gen2_generation = 1;
'''
TAIL = br'''
    printf("calls=%d ids=%d\n", calls, ids);
    return 0;
}
'''


def fixture():
    lines = wall_fixture()
    for thread in range(4):
        lines.append(f'[DEBUG-graph-cpu-v1] graph pool=0x1234 generation=8 thread={thread} '
                     f'tid={thread + 1} kernel_100ns=0 user_100ns=0 errors=0 error_bits=0 first_error=0')
    return lines


class GraphCPUAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('clang') or shutil.which('cc')
        if not compiler:
            raise unittest.SkipTest('C compiler required for graph-clock insertion controls')
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        source = root / 'graph-clock.c'
        source.write_bytes(HARNESS + HELPERS + FUNCTION + DECLARATIONS +
                           b'    if (enabled) {\n' + BEGIN + END + b'    }\n' + TAIL)
        cls.binary = root / 'graph-clock'
        subprocess.run([compiler, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                        source, '-o', cls.binary], capture_output=True, check=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def execute(self, mode=0, enabled=True):
        return subprocess.run([self.binary, str(mode), str(int(enabled))],
                              check=True, capture_output=True, timeout=5)

    def test_two_reads_only_at_graph_boundaries_preserve_64bit_cpu_durations(self):
        result = self.execute()
        self.assertIn(b'kernel_100ns=4294967298 user_100ns=4294967301 errors=0', result.stderr)
        self.assertEqual(result.stdout.strip(), b'calls=2 ids=1')
        self.assertEqual(len(result.stderr.splitlines()), 1)

    def test_disabled_insertion_calls_no_windows_cpu_api(self):
        result = self.execute(enabled=False)
        self.assertEqual(result.stderr, b'')
        self.assertEqual(result.stdout.strip(), b'calls=0 ids=0')

    def test_api_failures_regressions_and_genuine_zero_deltas_remain_explicit(self):
        self.assertIn(b'errors=1 error_bits=1 first_error=123', self.execute(1).stderr)
        self.assertIn(b'errors=3 error_bits=3 first_error=123', self.execute(2).stderr)
        self.assertIn(b'errors=2 error_bits=2 first_error=0', self.execute(3).stderr)
        self.assertIn(b'kernel_100ns=0 user_100ns=0 errors=0', self.execute(4).stderr)


class GraphCPUTraceTest(unittest.TestCase):
    def summarize(self, lines, enabled=True):
        return summarize('\n'.join(lines), enabled, expected_graphs=1,
                         expected_threads=4, expected_matrices=1)

    def test_complete_trace_joins_cpu_to_wall_identity_without_invented_shares(self):
        value = self.summarize(fixture())
        self.assertEqual([w['graph_cpu']['tid'] for w in value['graphs'][0]['workers']], [1, 2, 3, 4])
        self.assertTrue(all(w['graph_cpu']['user_100ns'] == 0 for w in value['graphs'][0]['workers']))
        self.assertNotIn('cycles', value['graphs'][0]['workers'][0]['graph_cpu'])

    def test_incomplete_duplicate_orphan_error_mixed_and_malformed_records_reject(self):
        original = fixture()
        changes = [original[:-1], original + [original[-1]],
                   [line.replace('pool=0x1234', 'pool=0xdef') if 'graph-cpu' in line else line for line in original],
                   [line.replace('errors=0', 'errors=1') for line in original],
                   [line.replace('error_bits=0', 'error_bits=1') for line in original],
                   [line.replace('first_error=0', 'first_error=123') for line in original],
                   [line.replace('tid=2 ', 'tid=1 ') for line in original],
                   [line.replace('tid=1 ', 'tid=0 ') for line in original],
                   [line.replace('user_100ns=0', 'user_100ns=6000000001') for line in original],
                   original + ['[DEBUG-thread-cpu-v1] historical'],
                   original + ['[DEBUG-matrix-cpu-v1] historical'],
                   [line + ' trailing' if 'graph-cpu' in line else line for line in original]]
        for changed in changes:
            with self.subTest(changed=changed[-1]), self.assertRaises(ValueError):
                self.summarize(changed)

    def test_environment_and_disabled_mode_exclude_all_historical_observer_keys(self):
        old = {'gen2_diar_cpu_profile': '1', 'GEN2_DIAR_MATRIX_PROFILE': '1',
               'gen2_diar_thread_cpu_profile': '1', 'GEN2_DIAR_GRAPH_CPU_PROFILE': '1', 'PATH': '/bin'}
        self.assertEqual(environment(old, False), {'PATH': '/bin'})
        self.assertEqual(environment(old, True), {'PATH': '/bin', 'GEN2_DIAR_GRAPH_CPU_PROFILE': '1'})
        self.assertIn('gen2_diar_cpu_profile', old)
        with self.assertRaises(ValueError):
            environment({}, 1)
        self.assertEqual(self.summarize([], False)['graphs'], [])
        with self.assertRaises(ValueError):
            self.summarize(fixture(), False)

    def test_canonical_transform_keeps_all_inference_and_wall_regions_and_has_no_cycle_api(self):
        raw = (Path(__file__).parent / 'fixtures/matrix-paths-ggml-cpu.c').read_bytes()
        base, changed = batch_transform(raw), transform(raw)
        for call in [b'ggml_barrier(', b'ggml_cpu_try_fuse_ops(',
                     b'ggml_compute_forward_mul_mat_one_chunk(', b'ggml_compute_forward(&params, node)',
                     b'ggml_time_us(']:
            self.assertEqual(base.count(call), changed.count(call), call)
        self.assertEqual(changed.count(b'GetThreadTimes('), 1)
        self.assertEqual(changed.count(b'gen2_graph_clock_read();'), 2)
        self.assertNotIn(b'QueryThreadCycleTime', changed)
        self.assertIn(b'getenv("GEN2_DIAR_GRAPH_CPU_PROFILE")', changed)
        # Outside the declared additions and mode key, every baseline byte stays.
        restored = changed
        for addition in [HELPERS + b'\n', DECLARATIONS, BEGIN, END]:
            self.assertEqual(restored.count(addition), 1)
            restored = restored.replace(addition, b'', 1)
        restored = restored.replace(b'getenv("GEN2_DIAR_GRAPH_CPU_PROFILE")',
                                    b'getenv("GEN2_DIAR_MATRIX_PROFILE")')
        self.assertEqual(restored, base)
        with self.assertRaises(ValueError):
            transform(b'wrong source')


if __name__ == '__main__':
    unittest.main()
