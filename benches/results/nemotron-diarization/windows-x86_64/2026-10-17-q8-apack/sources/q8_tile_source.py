"""One hash-guarded Q8-only AVX2 tile dispatch change; arithmetic stays intact."""
import hashlib

PATH = 'ggml/src/ggml-cpu/llamafile/sgemm.cpp'
ORIGINAL = {'bytes': 162829, 'sha256': 'e226380b05393302ba91b38d22718d9521700ee2cb7943f83ad316a7a9dcd597'}
CLASS = b'class tinyBLAS_Q0_AVX {'
OLD = b'''    void matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }
'''
NEW = b'''    void matmul(int64_t m, int64_t n) {
#if defined(_WIN32) && defined(__AVX2__) && !defined(__AVX512F__)
        if constexpr (std::is_same<TA, block_q8_0>::value &&
                      std::is_same<TB, block_q8_0>::value && std::is_same<TC, float>::value) {
            if (m >= 2 && n >= 2) {
                const int64_t me = m / 2 * 2;
                const int64_t ne = n / 2 * 2;
                gemm<2, 2>(0, me, 0, ne);
                mnpack(me, m, 0, ne);
                mnpack(0, m, ne, n);
                return;
            }
        }
#endif
        mnpack(0, m, 0, n);
    }
'''


def transform(raw):
    if {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} != ORIGINAL:
        raise ValueError('exact admitted nineteen-patch SGEMM source required')
    if raw.count(CLASS) != 1:
        raise ValueError('one original AVX quantized class required')
    prefix, body = raw.split(CLASS, 1)
    if body.count(OLD) != 1:
        raise ValueError('one original AVX class dispatch required')
    return prefix + CLASS + body.replace(OLD, NEW, 1)
