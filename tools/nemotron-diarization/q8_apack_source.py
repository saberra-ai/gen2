"""Private 2x4 Q8 helper with bounded per-dispatch A packing."""
from q8_mx4_source import transform as control_source
from q8_tile_source import CLASS, ORIGINAL, PATH
from q8_worker_timing_source import once


GUARD = b'#if defined(_WIN32) && defined(__AVX2__) && defined(__F16C__) && !defined(__AVX512F__)\n'
PACK = GUARD + b'''// Separate payload/scale arrays: 36 bytes per packed block, at most 64 MiB per worker.
static size_t gen2_q8_apack_blocks(int64_t rows, int64_t k) {
    constexpr size_t maximum = (64u * 1024u * 1024u) / 36u;
    if (rows <= 0 || k <= 0 || uint64_t(rows) > maximum / uint64_t(k))
        return 0;
    return size_t(rows) * size_t(k);
}

static void gen2_q8_apack_copy(const block_q8_0 *a, int64_t lda, int64_t first,
                               int64_t rows, int64_t k, __m256i *quants, float *scales) {
    for (int64_t row = 0; row < rows; ++row)
        for (int64_t l = 0; l < k; ++l) {
            const auto &block = a[lda * (first + row) + l];
            const size_t index = size_t(row) * size_t(k) + size_t(l);
            _mm256_storeu_si256(quants + index, _mm256_loadu_si256((const __m256i *)block.qs));
            scales[index] = unhalf(block.d);
        }
}
#endif

'''

EXPORT = b'\n' + GUARD + b'''// Diagnostic transport seams call the same packing and bound functions as the helper.
extern "C" void gen2_q8_apack_transport(const block_q8_0 *a, int64_t lda, int64_t first,
                                       int64_t rows, int64_t k, void *quants, float *scales) {
    gen2_q8_apack_copy(a, lda, first, rows, k, (__m256i *)quants, scales);
}
extern "C" size_t gen2_q8_apack_bound(int64_t rows, int64_t k) {
    return gen2_q8_apack_blocks(rows, k);
}
#endif
'''


def packed_kernel(control):
    start = b'    // Templated functions for gemm of dimensions Mx4\n'
    end = b'\n#endif\n\n    template <int RM, int RN>'
    if control.count(start) != 1 or control.count(end) != 1:
        raise ValueError('one exact canonical Mx4 helper required')
    body = control.split(start, 1)[1].split(end, 1)[0]
    body = once(body, b'void gemmMx4(', b'void gemmPackA(')
    before = b'''        for (int64_t job = start; job < end; ++job) {'''
    after = b'''        if (start >= end)
            return;
        const int64_t first_row = m0 + start / xtiles * RM;
        const int64_t last_row = m0 + ((end - 1) / xtiles + 1) * RM;
        const size_t blocks = gen2_q8_apack_blocks(last_row - first_row, k);
        if (!blocks) {
            gemmMx4<RM>(m0, m, n0, n);
            return;
        }
        void *storage = _aligned_malloc(blocks * 36u, 32u);
        if (!storage) {
            gemmMx4<RM>(m0, m, n0, n);
            return;
        }
        struct Release {
            void *storage;
            ~Release() { _aligned_free(storage); }
        } release{storage};
        auto *quants = (__m256i *)storage;
        auto *scales = (float *)(quants + blocks);
        gen2_q8_apack_copy(A, lda, first_row, last_row - first_row, k, quants, scales);
        for (int64_t job = start; job < end; ++job) {'''
    body = once(body, before, after)
    body = once(body, b'__m128 da = _mm_set1_ps(unhalf((A[lda * (ii + i) + l].d)));',
        b'__m128 da = _mm_set1_ps(scales[size_t(ii + i - first_row) * size_t(k) + size_t(l)]);')
    old = b'load(A + lda * (ii + i) + l)'
    if body.count(old) != 12:
        raise ValueError('twelve exact original A load expressions required')
    body = body.replace(old, b'_mm256_load_si256(quants + size_t(ii + i - first_row) * size_t(k) + size_t(l))')
    return GUARD + body + b'\n#endif\n\n'


def transform(raw):
    control = control_source(raw)
    result = once(control, b'#include <array>\n', b'#include <array>\n' + GUARD + b'#include <malloc.h>\n#endif\n')
    marker = b'template <typename TA, typename TB, typename TC>\n' + CLASS
    result = once(result, marker, PACK + marker)
    head, body = result.split(CLASS, 1)
    body = once(body, b'                gemmMx4<2>(0, me, 0, ne);', b'                gemmPackA<2>(0, me, 0, ne);')
    body = once(body, b'  private:\n    void mnpack(', b'  private:\n' + packed_kernel(control) + b'    void mnpack(')
    return once(head + CLASS + body, b'GEN2_Q8_MX4_EXPECTED', b'GEN2_Q8_APACK_EXPECTED') + EXPORT
