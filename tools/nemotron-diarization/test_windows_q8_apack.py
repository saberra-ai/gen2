"""Bounded A packing, exact arithmetic and unchanged timing gates."""
import copy
import json
from pathlib import Path
import struct
import shutil
import subprocess
import sys
import tempfile
import unittest

from q8_apack_source import control_source, transform, packed_kernel, PACK, EXPORT, GUARD as PACK_GUARD
from q8_apack_experiment_source import derive_benchmark, derive_cmake, derive_metrics, derive_probe
from windows_q8_model_sources import candidate_source, GUARD
from windows_q8_apack_metrics import validate_output, timing_evidence, KERNEL_ORDER
from test_windows_q8_worker_timing import rows as parent_rows, cases as parent_cases
from q8_mx4_geometry import SHAPES, PRIMARY_SHAPES
from q8_apack_metrics import admit_packing

TOOLS = Path(__file__).resolve().parent


def relabel(rows):
    for row in rows:
        if row['variant'] == 'mx4':
            row['variant'] = 'apack'
    return rows


def cases():
    values = parent_cases()
    for case in values:
        relabel(case['observed']['rows'])
    return values


class APackKernelTest(unittest.TestCase):
    def test_actual_cpp_transport_and_bounds_with_portable_load_store_shim(self):
        if sys.platform == 'win32':
            self.skipTest('actual Windows AVX transport controls run in the frozen benchmark')
        compiler = shutil.which('clang++') or shutil.which('g++')
        if not compiler:
            self.skipTest('portable C++ compiler unavailable; Windows runs real AVX transport controls')
        harness = r'''
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
struct block_q8_0 { uint16_t d; int8_t qs[32]; };
using ggml_fp16_t = uint16_t;
struct __m256i { unsigned char bytes[32]; };
static_assert(sizeof(block_q8_0)==34, "same Q8 layout");
float ggml_table_f32_f16[65536];
static float unhalf(uint16_t x) { return ggml_table_f32_f16[x]; }
static __m256i _mm256_loadu_si256(const __m256i *p) {
    __m256i value; std::memcpy(&value,p,32); return value;
}
static void _mm256_storeu_si256(__m256i *p, __m256i value) { std::memcpy(p,&value,32); }
constexpr float guard = 0x1p80f;
static void save_output(const std::string &p,const void *v,size_t n) {
    std::ofstream f(p,std::ios::binary); f.write((const char*)v,n);
    if (!f) throw std::runtime_error("write failed");
}
'''
        # Compile the actual transport/bounds source with scalar memcpy intrinsics.
        # This tests indexing/guards/bounds, not Windows AVX code generation.
        body = PACK[len(PACK_GUARD):].removesuffix(b'#endif\n\n')
        exports = EXPORT[1 + len(PACK_GUARD):].removesuffix(b'#endif\n')
        harness += (body + exports + (TOOLS / 'q8_apack_controls.cpp').read_bytes()).decode()
        harness += r'''
int main(int argc,char **argv) {
    if(argc!=3) return 2;
    std::ifstream f(argv[1],std::ios::binary);
    f.read((char*)ggml_table_f32_f16,sizeof(ggml_table_f32_f16));
    if(!f) return 3;
    packing_controls(argv[2]);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, binary = root / 'control.cpp', root / 'control'
            source.write_text(harness)
            half = b''.join(struct.pack('<f', struct.unpack('<e', struct.pack('<H', i))[0]) for i in range(65536))
            (root / 'half.f32').write_bytes(half)
            subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', str(source), '-o', str(binary)],
                check=True, capture_output=True, timeout=60)
            prefix = root / 'observed'
            subprocess.run([str(binary), str(root / 'half.f32'), str(prefix)], check=True, capture_output=True, timeout=30)
            payload = (root / 'observed.apack-payload.i8').read_bytes()
            scales = (root / 'observed.apack-scales.f32').read_bytes()
            result = admit_packing(payload, scales)
            self.assertEqual(scales, half)
            self.assertEqual(result['blocks'], 65536)
            for altered_payload, altered_scales in ((payload[:-1], scales),
                    (bytes([payload[0] ^ 1]) + payload[1:], scales), (payload, scales[:-4]),
                    (payload, b'\0\0\x80\x3f' + scales[4:])):
                with self.assertRaises(ValueError): admit_packing(altered_payload, altered_scales)

    def test_control_is_exact_current_production_kernel_without_embedded_define(self):
        original = (TOOLS / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
        self.assertEqual(GUARD + control_source(original), candidate_source())
        self.assertIn(b'gemmMx4<2>(0, me, 0, ne);', control_source(original))

    def test_private_packing_keeps_all_original_arithmetic_bounds_and_ownership(self):
        original = (TOOLS / 'fixtures/q8-tile-sgemm.cpp').read_bytes()
        control = control_source(original)
        candidate = transform(original)
        helper = packed_kernel(control)
        restored = candidate.replace(helper, b'', 1).replace(PACK, b'', 1)
        self.assertTrue(restored.endswith(EXPORT))
        restored = restored[:-len(EXPORT)].replace(
            PACK_GUARD + b'#include <malloc.h>\n#endif\n', b'', 1).replace(
            b'gemmPackA<2>(0, me, 0, ne);', b'gemmMx4<2>(0, me, 0, ne);', 1).replace(
            b'GEN2_Q8_APACK_EXPECTED', b'GEN2_Q8_MX4_EXPECTED', 1)
        self.assertEqual(restored, control)
        for line in (b'if (m >= 2 && n >= 4)', b'const int64_t me = m / 2 * 2;',
                b'const int64_t ne = n / 4 * 4;', b'mnpack(me, m, 0, ne);', b'mnpack(0, m, ne, n);'):
            self.assertIn(line, candidate)
        for line in (b'int64_t xtiles = (n - n0) / 4;', b'int64_t duty = (tiles + nth - 1) / nth;',
                b'int64_t ii = m0 + job / xtiles * RM;', b'int64_t jj = n0 + job % xtiles * 4;',
                b'__m256 Cv[4][RM] = {};', b'~Release() { _aligned_free(storage); }'):
            self.assertIn(line, helper)
        # Every ordered contribution differs only in where identical A bytes load.
        for column in range(4):
            start = f'                    Cv[{column}][i] = madd('.encode()
            end = f'                                    Cv[{column}][i]);'.encode()
            expected = control[control.index(start, control.index(b'NOINLINE void gemmMx4')):]
            expected = expected[:expected.index(end) + len(end)].replace(
                b'load(A + lda * (ii + i) + l)',
                b'_mm256_load_si256(quants + size_t(ii + i - first_row) * size_t(k) + size_t(l))')
            self.assertIn(expected, helper)
        self.assertEqual(helper.count(b'_aligned_malloc('), 1)
        self.assertEqual(helper.count(b'gemmMx4<RM>(m0, m, n0, n);'), 2)
        for bad in (original[:-1], b'changed' + original):
            with self.assertRaises(ValueError): transform(bad)

    def test_all_construction_support_clock_and_metric_changes_are_exact_derivations(self):
        for target, parent, derive in (
                ('windows_q8_apack_probe.py', 'windows_q8_row_probe.py', derive_probe),
                ('windows_q8_apack_metrics.py', 'windows_q8_row_metrics.py', derive_metrics),
                ('q8_apack_benchmark.cmake', 'q8_row_benchmark.cmake', derive_cmake),
                ('q8_apack_benchmark.cpp', 'q8_row_benchmark.cpp', derive_benchmark)):
            with self.subTest(target=target):
                args = [(TOOLS / parent).read_bytes()]
                if target.endswith('.cpp'): args.append((TOOLS / 'q8_apack_controls.cpp').read_bytes())
                self.assertEqual((TOOLS / target).read_bytes(), derive(*args))
                with self.assertRaises(ValueError):
                    args[0] = b'incomplete parent'; derive(*args)

    def test_complete_phase_vectors_and_tails_reject_changed_or_missing_bytes(self):
        rows = relabel(parent_rows('on', 4))
        vectors = [struct.pack('<f', .25) * (m*n) for m, n, k in SHAPES]
        outputs = [raw for raw in vectors for _ in range(4)]
        text = '\n'.join(map(json.dumps, rows))
        self.assertEqual(len(validate_output(text, 'on', 4, outputs)['outputs']), 88)
        changed = outputs.copy(); changed[1] = b'\0\0\0\0' + changed[1][4:]
        with self.assertRaises(ValueError): validate_output(text, 'on', 4, changed)
        with self.assertRaises(ValueError): validate_output(text, 'on', 4, outputs[:-1])
        rows[1]['variant'] = 'mx4'
        with self.assertRaises(ValueError): validate_output('\n'.join(map(json.dumps, rows)), 'on', 4, outputs)

    def test_original_integer_speedup_predicate_applies_to_every_primary_process(self):
        self.assertEqual(KERNEL_ORDER, ['control', 'apack', 'apack', 'control'])
        self.assertTrue(timing_evidence(cases())['kernel_speedup_prediction_supported'])
        for shape in PRIMARY_SHAPES:
            for process in range(8, 12):
                value = cases()
                value[process]['observed']['rows'][shape*4+1]['timed_ns'] = [1901] * 8
                with self.subTest(shape=shape, process=process):
                    self.assertFalse(timing_evidence(value)['same_process_kernel_prediction_supported'])

    def test_observer_bounds_and_order_cannot_be_replaced_by_kernel_speed(self):
        value = cases()
        value[9]['observed']['rows'][4*4]['timed_ns'] = [2101] * 8
        result = timing_evidence(value)
        self.assertTrue(result['same_process_kernel_prediction_supported'])
        self.assertFalse(result['bounded_observer_interpretation_supported'])
        self.assertFalse(result['kernel_speedup_prediction_supported'])
        value = copy.deepcopy(cases()); value[0]['observed']['rows'][0]['worker_intervals_ns'] = [[{}]]
        with self.assertRaises(ValueError): timing_evidence(value)
        with self.assertRaises(ValueError): timing_evidence(cases()[1:])


if __name__ == '__main__': unittest.main()
