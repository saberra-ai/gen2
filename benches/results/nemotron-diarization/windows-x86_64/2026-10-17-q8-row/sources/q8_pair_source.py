"""Separate Q8 2x4 candidate: consume one B vector across two A rows."""
from q8_mx4_source import transform as control_source
from q8_tile_source import CLASS, PATH, ORIGINAL


def pair_kernel():
    """Each output retains its original l order, updot, madd and hsum."""
    prefix = b'''#if defined(_WIN32) && defined(__AVX2__) && defined(__F16C__) && !defined(__AVX512F__)
    NOINLINE void gemmPair2x4(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / 2;
        int64_t xtiles = (n - n0) / 4;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * 2;
            int64_t jj = n0 + job % xtiles * 4;
            __m256 Cv[4][2] = {};
            for (int64_t l = 0; l < k; ++l) {
                uint64_t b_delta = ((uint64_t)B[ldb * (jj + 3) + l].d << 48) | ((uint64_t)B[ldb * (jj + 2) + l].d << 32) | ((uint64_t)B[ldb * (jj + 1) + l].d << 16) | (B[ldb * (jj + 0) + l].d);
                __m128 db = _mm_cvtph_ps(_mm_set_epi64x(0, b_delta));
                const float a0 = unhalf(A[lda * (ii + 0) + l].d);
                const float a1 = unhalf(A[lda * (ii + 1) + l].d);
                __m128 da = _mm_setr_ps(a0, a1, a0, a1);
                __m256i avec0 = load(A + lda * (ii + 0) + l);
                __m256i avec1 = load(A + lda * (ii + 1) + l);
                __m256i abs0 = _mm256_sign_epi8(avec0, avec0);
                __m256i abs1 = _mm256_sign_epi8(avec1, avec1);
'''
    steps = []
    for column, shuffle in enumerate((0, 85, 170, 255)):
        steps.append(f'''                {{
                    __m256i bvec = load(B + ldb * (jj + {column}) + l);
                    __m256 dvec = _mm256_castps128_ps256(_mm_mul_ps(da, _mm_shuffle_ps(db, db, {shuffle})));
                    dvec = _mm256_permute2f128_ps(dvec, dvec, 0);
                    Cv[{column}][0] = madd(_mm256_shuffle_ps(dvec, dvec, 0),
                                         updot(abs0, _mm256_sign_epi8(bvec, avec0)), Cv[{column}][0]);
                    Cv[{column}][1] = madd(_mm256_shuffle_ps(dvec, dvec, 85),
                                         updot(abs1, _mm256_sign_epi8(bvec, avec1)), Cv[{column}][1]);
                }}
'''.encode())
    return prefix + b''.join(steps) + b'''            }
            for (int64_t j = 0; j < 4; ++j)
                for (int64_t i = 0; i < 2; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }
#endif

'''


def transform(raw):
    control = control_source(raw)
    head, body = control.split(CLASS, 1)
    call = b'                gemmMx4<2>(0, me, 0, ne);'
    insertion = b'  private:\n    void mnpack('
    if body.count(call) != 1 or body.count(insertion) != 1:
        raise ValueError('one exact Q8 dispatch and private insertion site required')
    body = body.replace(call, b'                gemmPair2x4(0, me, 0, ne);', 1)
    body = body.replace(insertion, b'  private:\n' + pair_kernel() + b'    void mnpack(', 1)
    result = head + CLASS + body
    guard = b'GEN2_Q8_MX4_EXPECTED'
    if result.count(guard) != 1:
        raise ValueError('one exact compilation guard required')
    return result.replace(guard, b'GEN2_Q8_PAIR_EXPECTED', 1)
