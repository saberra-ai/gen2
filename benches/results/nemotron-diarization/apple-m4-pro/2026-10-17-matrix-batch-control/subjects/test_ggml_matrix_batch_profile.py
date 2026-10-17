"""Execute the actual emission blocks, including bounded-buffer failure paths."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from ggml_matrix_profile_source import EMIT as ORIGINAL_EMIT, HELPERS, transform as original_transform
from ggml_matrix_batch_profile_source import EMIT, transform


HARNESS = br'''
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#define GGML_USE_LLAMAFILE 0
'''

WRAPPERS = br'''
static int failure, allocations, frees, writes;
static void * observed_malloc(size_t count) {
    allocations++;
    return failure == 1 ? NULL : malloc(count);
}
static void observed_free(void * value) { frees++; free(value); }
static int observed_snprintf(char * output, size_t count, const char * format, ...) {
    if (strstr(format, "[DEBUG-matrix-paths-v1] matrix")) {
        if (failure == 2) return -1;
        if (failure == 3) return (int) count;
    }
    va_list arguments;
    va_start(arguments, format);
    int result = vsnprintf(output, count, format, arguments);
    va_end(arguments);
    return result;
}
static size_t observed_fwrite(const void * data, size_t size, size_t count, FILE * stream) {
    writes++;
    return failure == 4 ? 0 : fwrite(data, size, count, stream);
}
#define malloc observed_malloc
#define free observed_free
#define snprintf observed_snprintf
#define fwrite observed_fwrite
int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    failure = atoi(argv[1]);
    const int gen2_matrix_count = atoi(argv[2]);
    const bool gen2_profile = atoi(argv[3]) != 0;
    if (gen2_matrix_count < 0 || gen2_matrix_count > 256) return 3;
    const bool gen2_matrix_overflow = false;
    const int gen2_generation = 8;
    void * tp = (void *) (uintptr_t) 0x1234;
    struct { int ith; } worker = { 2 }, * state = &worker;
    struct gen2_matrix_record gen2_matrices[256] = {0};
    for (int i = 0; i < gen2_matrix_count; ++i) {
        struct gen2_matrix_record * value = &gen2_matrices[i];
        value->node = i; value->route = 5; value->dot = 8;
        value->stage[0].calls = 1; value->stage[0].failed = 1;
        value->stage[1].calls = 1; value->stage[1].failed = 1;
        value->chunks = UINT64_MAX; value->chunk_us = INT64_MAX;
        value->kernel_us = INT64_MAX;
        for (int t = 0; t < 3; ++t) {
            value->names[t] = "iq4_nl_4_8";
            for (int d = 0; d < 4; ++d) {
                value->ne[t][d] = INT64_MAX;
                value->nb[t][d] = UINT64_MAX;
            }
        }
    }
'''

TAIL = br'''
    printf("allocations=%d frees=%d writes=%d\n", allocations, frees, writes);
    return 0;
}
'''


class MatrixBatchProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        compiler = shutil.which('clang') or shutil.which('cc')
        if compiler is None:
            raise unittest.SkipTest('C compiler unavailable for actual emitter controls')
        for name, emission in [('original', ORIGINAL_EMIT), ('batch', EMIT)]:
            source = cls.root / (name + '.c')
            source.write_bytes(HARNESS + HELPERS + WRAPPERS + emission + TAIL)
            subprocess.run([compiler, '-std=c11', '-O2', str(source), '-o', str(cls.root / name)],
                           check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def execute(self, name, failure=0, count=256, enabled=True):
        return subprocess.run([str(self.root / name), str(failure), str(count), str(int(enabled))],
                              check=True, capture_output=True, timeout=5)

    def test_all_256_extreme_records_preserve_wire_bytes_in_one_write(self):
        original = self.execute('original')
        batch = self.execute('batch')
        self.assertEqual(original.stderr, batch.stderr)
        self.assertEqual(len(batch.stderr.splitlines()), 256)
        self.assertEqual(batch.stdout, b'allocations=1 frees=1 writes=1\n')

    def test_disabled_and_empty_graphs_do_not_allocate_or_write(self):
        for count, enabled in [(0, True), (256, False)]:
            result = self.execute('batch', count=count, enabled=enabled)
            self.assertEqual(result.stderr, b'')
            self.assertEqual(result.stdout, b'allocations=0 frees=0 writes=0\n')

    def test_allocation_format_truncation_and_short_write_emit_rejected_marker(self):
        for failure in (1, 2, 3, 4):
            with self.subTest(failure=failure):
                result = self.execute('batch', failure=failure)
                self.assertEqual(result.stderr, b'[DEBUG-matrix-paths-v1] emission-error\n')
                expected = {1: b'allocations=1 frees=0 writes=0\n',
                            2: b'allocations=1 frees=1 writes=0\n',
                            3: b'allocations=1 frees=1 writes=0\n',
                            4: b'allocations=1 frees=1 writes=1\n'}[failure]
                self.assertEqual(result.stdout, expected)

    def test_transform_changes_only_emission_after_original_final_barrier(self):
        raw = (Path(__file__).parent / 'fixtures/matrix-paths-ggml-cpu.c').read_bytes()
        original = original_transform(raw)
        changed = transform(raw)
        self.assertEqual(changed, original.replace(ORIGINAL_EMIT, EMIT))
        self.assertEqual(changed.count(b'ggml_barrier('), raw.count(b'ggml_barrier('))
        self.assertNotIn(b'char gen2_matrix_buffer[', changed)
        self.assertLess(changed.index(b'ggml_barrier(state->threadpool);',
                        changed.index(b'const int64_t gen2_before_final')),
                        changed.index(EMIT))
        with self.assertRaises(ValueError):
            transform(b'wrong source')

    def test_emission_error_marker_is_rejected_by_existing_decoder(self):
        from ggml_matrix_profile import summarize
        from test_ggml_matrix_profile import fixture
        with self.assertRaises(ValueError):
            summarize('\n'.join(fixture()) + '\n[DEBUG-matrix-paths-v1] emission-error\n', True,
                      expected_graphs=1, expected_threads=4, expected_matrices=1)


if __name__ == '__main__':
    unittest.main()
