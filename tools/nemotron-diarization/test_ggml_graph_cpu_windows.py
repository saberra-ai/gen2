"""Run the exact C graph-clock insertions against real Windows APIs, without GGML."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from ggml_graph_cpu_profile import FIELDS, GRAPH
from ggml_graph_cpu_profile_source import BEGIN, DECLARATIONS, END, HELPERS

HEADER = br'''
#define WIN32_LEAN_AND_MEAN
#define _WIN32_WINNT 0x0600
#include <windows.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
'''
FUNCTION = br'''
int main(void) {
    void * tp = (void *)(uintptr_t)1;
    struct worker_state { int ith; } actual_state = {0};
    const struct worker_state * state = &actual_state;
    const int gen2_generation = 1;
    const unsigned long actual_tid = (unsigned long) GetCurrentThreadId();
'''
WORK = br'''
        const ULONGLONG deadline = GetTickCount64() + 40;
        volatile uint64_t work = 1;
        do {
            for (int i = 0; i < 10000; ++i) work = work * 1664525 + 1013904223;
        } while (GetTickCount64() < deadline);
'''
TAIL = br'''
    printf("thread_id=%lu\n", actual_tid);
    return gen2_graph_clock_errors ? 3 : 0;
}
'''


@unittest.skipUnless(os.name == 'nt', 'actual Windows API control requires Windows')
class GraphCPUWindowsTest(unittest.TestCase):
    def test_real_current_thread_cpu_api_compiles_runs_and_preserves_successful_counters(self):
        from platform_native import msvc_environment, msvc_tool
        environment = msvc_environment(dict(os.environ))
        compiler = msvc_tool(environment, 'clang-cl.exe')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, binary = root / 'graph-clock.c', root / 'graph-clock.exe'
            source.write_bytes(HEADER + HELPERS + FUNCTION + DECLARATIONS + BEGIN + WORK + END + TAIL)
            # No fake-API macro or replacement Windows functions are provided.
            compiled = subprocess.run([str(compiler), '/nologo', '/TC', '/clang:-std=c11', '/O2', '/W4', '/WX',
                                       str(source), 'kernel32.lib', '/link', '/OUT:' + str(binary)],
                                      cwd=root, env=environment, capture_output=True, timeout=30)
            self.assertEqual(compiled.returncode, 0,
                             (compiled.stdout + compiled.stderr).decode(errors='replace'))
            result = subprocess.run([str(binary)], cwd=root, env=environment,
                                    capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 0,
                             (result.stdout + result.stderr).decode(errors='replace'))
            lines = result.stderr.decode().splitlines()
            self.assertEqual(len(lines), 1)
            match = GRAPH.fullmatch(lines[0])
            self.assertIsNotNone(match, lines[0])
            values = dict(zip(FIELDS, map(int, match.groups()[3:]), strict=True))
            self.assertGreater(values['tid'], 0)
            self.assertEqual(result.stdout.decode().strip(), 'thread_id=' + str(values['tid']))
            self.assertEqual([values[key] for key in ['errors', 'error_bits', 'first_error']], [0, 0, 0])
            for key in ['kernel_100ns', 'user_100ns']:
                self.assertGreaterEqual(values[key], 0)
                self.assertLessEqual(values[key], 2**64 - 1)
            # Zero deltas remain valid; this short control is not a timer-resolution
            # calibration, graph benchmark, observer bound or model performance run.
            print('Actual Windows graph-clock API control:', lines[0], flush=True)


if __name__ == '__main__':
    unittest.main()
