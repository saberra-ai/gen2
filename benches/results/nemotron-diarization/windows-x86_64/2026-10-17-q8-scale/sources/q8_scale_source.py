"""Private production-geometry Q8 helper using F16C for A half scales."""
from q8_mx4_source import transform as control_source
from q8_tile_source import CLASS, ORIGINAL, PATH


BROADCAST = b'''static inline __m128 gen2_q8_scale_broadcast(uint16_t value) {
    const __m128 converted = _mm_cvtph_ps(_mm_cvtsi32_si128(value));
    return _mm_shuffle_ps(converted, converted, 0);
}

extern "C" void gen2_q8_scale_broadcast_probe(uint16_t value, float * output) {
    _mm_storeu_ps(output, gen2_q8_scale_broadcast(value));
}

'''
OLD_SCALE = b'                    __m128 da = _mm_set1_ps(unhalf((A[lda * (ii + i) + l].d)));'
NEW_SCALE = b'                    __m128 da = gen2_q8_scale_broadcast(A[lda * (ii + i) + l].d);'


def scale_kernel(control):
    start = b'    // Templated functions for gemm of dimensions Mx4\n'
    end = b'\n#endif\n\n    template <int RM, int RN>'
    if control.count(start) != 1:
        raise ValueError('exact original Mx4 helper required')
    body = control.split(start, 1)[1].split(end, 1)[0]
    if body.count(OLD_SCALE) != 1 or body.count(b'void gemmMx4(') != 1:
        raise ValueError('exact original scale and helper name required')
    body = body.replace(b'void gemmMx4(', b'void gemmScaleMx4(', 1)
    body = body.replace(OLD_SCALE, NEW_SCALE, 1)
    return b'#if defined(_WIN32) && defined(__AVX2__) && defined(__F16C__) && !defined(__AVX512F__)\n' + body + b'\n#endif\n\n'


def transform(raw):
    control = control_source(raw)
    head, body = control.split(CLASS, 1)
    call = b'                gemmMx4<2>(0, me, 0, ne);'
    insertion = b'  private:\n    void mnpack('
    namespace = b'namespace {\n'
    if body.count(call) != 1 or body.count(insertion) != 1 or head.count(namespace) != 1:
        raise ValueError('exact production dispatch, private helper and namespace required')
    body = body.replace(call, b'                gemmScaleMx4<2>(0, me, 0, ne);', 1)
    body = body.replace(insertion, b'  private:\n' + scale_kernel(control) + b'    void mnpack(', 1)
    head = head.replace(namespace, b'#if defined(_WIN32) && defined(__AVX2__) && defined(__F16C__) && !defined(__AVX512F__)\n' + BROADCAST + b'#endif\n\n' + namespace, 1)
    result = head + CLASS + body
    if result.count(b'GEN2_Q8_MX4_EXPECTED') != 1:
        raise ValueError('exact production compile guard required')
    return result.replace(b'GEN2_Q8_MX4_EXPECTED', b'GEN2_Q8_SCALE_EXPECTED', 1)
