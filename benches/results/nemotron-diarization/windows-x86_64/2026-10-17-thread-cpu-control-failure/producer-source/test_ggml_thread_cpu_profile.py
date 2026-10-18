"""Actual C API-adapter/error paths and full wall/counter identity admission."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from ggml_thread_cpu_profile import summarize,environment
from ggml_thread_cpu_profile_source import transform,HELPERS
from test_ggml_matrix_profile import fixture as wall_fixture

HARNESS=br'''
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#define GEN2_THREAD_CPU_CLOCK_TEST 1
typedef uint64_t ULONG64;
typedef struct { uint32_t dwLowDateTime, dwHighDateTime; } FILETIME;
static int mode,queries,times;
static void * GetCurrentThread(void) { return (void *)(uintptr_t)1; }
static unsigned GetLastError(void) { return 123; }
static int QueryThreadCycleTime(void * thread, ULONG64 * value) {
    if (thread!=(void *)(uintptr_t)1) exit(10);
    queries++;*value=UINT64_MAX;
    return mode!=1;
}
static int GetThreadTimes(void * thread,FILETIME * created,FILETIME * exited,FILETIME * kernel,FILETIME * user) {
    (void)created;(void)exited;
    if (thread!=(void *)(uintptr_t)1) exit(11);
    times++;kernel->dwHighDateTime=1;kernel->dwLowDateTime=2;
    user->dwHighDateTime=3;user->dwLowDateTime=4;
    return mode!=2;
}
'''
TAIL=br'''
int main(int argc,char ** argv) {
    if (argc!=2)return 2;mode=atoi(argv[1]);
    struct gen2_thread_clock sample=gen2_clock_thread();
    uint64_t delta=gen2_clock_delta(mode==3?10:4,9);
    printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %u %u %d %d\n",
        sample.cycles,sample.kernel,sample.user,delta,gen2_clock_errors,
        gen2_clock_error_bits,gen2_clock_first_error,queries,times);
    return 0;
}
'''


def fixture():
    lines=wall_fixture()
    # Existing fixture has one graph/four workers/one matrix and two operations.
    from ggml_matrix_profile import summarize as wall_summary
    base=wall_summary('\n'.join(lines),True,expected_graphs=1,expected_threads=4,expected_matrices=1)
    for graph in base['graphs']:
        for worker in graph['workers']:
            prefix=f"pool={graph['pool']:x} generation={graph['generation']} thread={worker['thread']}"
            lines.append(f"[DEBUG-thread-cpu-v1] graph {prefix} tid={worker['thread']+1} cycles=10000 kernel_100ns=0 user_100ns=0 final_wait_cycles=100 errors=0 error_bits=0 first_error=0")
            for op in worker['operations']:
                lines.append(f"[DEBUG-thread-cpu-v1] op {prefix} id={op['id']} kernel_cycles=1000 barrier_cycles=100")
            matrix,=worker['matrices']
            lines.append(f"[DEBUG-matrix-cpu-v1] matrix {prefix} node={matrix['node']} first_cycles=10 second_cycles=0 conversion_cycles=0 inner_wait_cycles=0 chunk_cycles=0 kernel_cycles=1000")
    return lines


class ThreadCPUProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=shutil.which('clang')or shutil.which('cc')
        if not compiler:raise unittest.SkipTest('C compiler required for actual clock adapter')
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        source=cls.root/'clock.c';source.write_bytes(HARNESS+HELPERS+TAIL);cls.binary=cls.root/'clock'
        subprocess.run([compiler,'-std=c11','-O2','-Wall','-Wextra','-Werror',source,'-o',cls.binary],capture_output=True,check=True,timeout=30)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def run_clock(self,mode):
        result=subprocess.run([self.binary,str(mode)],capture_output=True,check=True,timeout=5)
        return list(map(int,result.stdout.split()))
    def test_actual_adapter_preserves64bit_units_and_successful_zero_clock_granularity(self):
        self.assertEqual(self.run_clock(0),[2**64-1,2**32+2,3*2**32+4,5,0,0,0,1,1])
    def test_api_errors_and_regressions_are_explicit(self):
        self.assertEqual(self.run_clock(1)[0],0)
        self.assertEqual(self.run_clock(1)[4:7],[1,1,123])
        self.assertEqual(self.run_clock(2)[1:3],[0,0]);self.assertEqual(self.run_clock(2)[4:7],[1,2,123])
        self.assertEqual(self.run_clock(3)[3:7],[0,1,4,0])
    def test_complete_trace_retains_separate_clock_units(self):
        value=summarize('\n'.join(fixture()),True,expected_graphs=1,expected_threads=4,expected_matrices=1)
        self.assertEqual(value['graphs'][0]['workers'][0]['thread_cpu']['user_100ns'],0)
        self.assertEqual(value['graphs'][0]['workers'][0]['matrices'][0]['kernel_cycles'],1000)
    def test_missing_duplicate_orphan_bad_error_and_enclosure_reject(self):
        original=fixture()
        for changed in [original[:-1],original+[original[-1]],
                [line.replace('errors=0','errors=1')for line in original],
                [line.replace('cycles=10000 ','cycles=1 ')for line in original],
                [line.replace('first_cycles=10','first_cycles=2000')for line in original],
                [line.replace('inner_wait_cycles=0','inner_wait_cycles=1')for line in original],
                [line.replace('tid=2 ','tid=1 ')for line in original],
                original+['[DEBUG-matrix-cpu-v1] malformed']]:
            with self.subTest(changed=changed[-1]),self.assertRaises(ValueError):
                summarize('\n'.join(changed),True,expected_graphs=1,expected_threads=4,expected_matrices=1)
    def test_modes_and_full_source_guard_preserve_barriers_and_inference_calls(self):
        self.assertEqual(environment({'gen2_diar_matrix_profile':'1'},False),{})
        with self.assertRaises(ValueError):environment({},1)
        with self.assertRaises(ValueError):summarize('\n'.join(fixture()),False,expected_graphs=1,expected_threads=4,expected_matrices=1)
        raw=(Path(__file__).parent/'fixtures/matrix-paths-ggml-cpu.c').read_bytes();changed=transform(raw)
        for call in [b'ggml_barrier(',b'ggml_cpu_try_fuse_ops(',b'ggml_compute_forward_mul_mat_one_chunk(',b'ggml_compute_forward(&params, node)']:
            self.assertEqual(raw.count(call),changed.count(call))
        with self.assertRaises(ValueError):transform(b'wrong source')


class ThreadCPUEmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_ggml_matrix_batch_profile import HARNESS as BASE,WRAPPERS,TAIL as END
        compiler=shutil.which('clang')or shutil.which('cc')
        if not compiler:raise unittest.SkipTest('C compiler required for actual emission')
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.binary=cls.root/'emission'
        raw=(Path(__file__).parent/'fixtures/matrix-paths-ggml-cpu.c').read_bytes();changed=transform(raw)
        first=changed.index(b'struct gen2_matrix_stage {')
        last=changed.index(b'static _Thread_local struct gen2_matrix_record * gen2_matrix_current;',first)
        records=changed[first:last]
        first=changed.index(b'    if (gen2_profile) {\n        if (gen2_matrix_overflow)')
        last=changed.index(b'\n#ifdef GGML_USE_CPU_RISCV64_SPACEMIT',first)
        emit=changed[first:last]
        wrappers=WRAPPERS.replace(b'    va_list arguments;',br'''
    if (strstr(format,"[DEBUG-matrix-cpu-v1] matrix")) {
        if (failure==5)return -1;
        if (failure==6)return (int)count;
    }
    va_list arguments;''')
        wrappers=wrappers.replace(b'        value->kernel_us = INT64_MAX;',br'''        value->kernel_us = INT64_MAX;
        value->stage[0].cycles=UINT64_MAX;value->stage[1].cycles=UINT64_MAX;
        value->conversion_cycles=UINT64_MAX;value->inner_wait_cycles=UINT64_MAX;
        value->chunk_cycles=UINT64_MAX;value->kernel_cycles=UINT64_MAX;''')
        source=cls.root/'emission.c';source.write_bytes(BASE+records+wrappers+emit+END)
        subprocess.run([compiler,'-std=c11','-O2',source,'-o',cls.binary],check=True,capture_output=True,timeout=30)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def execute(self,failure,count=256,enabled=True):
        return subprocess.run([self.binary,str(failure),str(count),str(int(enabled))],check=True,capture_output=True,timeout=5)
    def test_256_extreme_rows_retain_both_complete_records_in_one_write(self):
        result=self.execute(0)
        self.assertEqual(len(result.stderr.splitlines()),512)
        self.assertEqual(result.stderr.count(b'kernel_cycles=18446744073709551615'),256)
        self.assertLess(len(result.stderr),256*4096)
        self.assertEqual(result.stdout,b'allocations=1 frees=1 writes=1\n')
    def test_all_old_and_new_emission_failures_reject_without_partial_flush(self):
        for failure in range(1,7):
            with self.subTest(failure=failure):
                result=self.execute(failure)
                self.assertEqual(result.stderr,b'[DEBUG-matrix-paths-v1] emission-error\n')
    def test_disabled_and_empty_emission_avoid_allocation_and_output(self):
        for count,enabled in [(0,True),(256,False)]:
            result=self.execute(0,count,enabled)
            self.assertEqual(result.stderr,b'');self.assertEqual(result.stdout,b'allocations=0 frees=0 writes=0\n')


if __name__=='__main__':unittest.main()
