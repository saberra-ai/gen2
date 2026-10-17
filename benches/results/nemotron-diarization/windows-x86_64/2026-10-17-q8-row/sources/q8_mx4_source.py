"""One Q8 Windows AVX2/F16C dispatch candidate; original helpers remain intact."""
from q8_tile_source import CLASS, OLD, NEW as PREVIOUS, ORIGINAL, PATH, transform as previous_transform


NEW = b'''    void matmul(int64_t m, int64_t n) {
#if defined(GEN2_Q8_MX4_EXPECTED) && (!defined(_WIN32) || !defined(__AVX2__) || !defined(__F16C__) || defined(__AVX512F__))
#error exact Windows AVX2/F16C candidate compilation required
#endif
#if defined(_WIN32) && defined(__AVX2__) && defined(__F16C__) && !defined(__AVX512F__)
        if constexpr (std::is_same<TA, block_q8_0>::value &&
                      std::is_same<TB, block_q8_0>::value && std::is_same<TC, float>::value) {
            if (m >= 2 && n >= 4) {
                const int64_t me = m / 2 * 2;
                const int64_t ne = n / 4 * 4;
                gemmMx4<2>(0, me, 0, ne);
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
    # Reuse the exact canonical admission, then replace the earlier dispatch
    # entirely. The compiled result contains only the new dispatch, not both.
    result = previous_transform(raw)
    if result.count(PREVIOUS) != 1:
        raise ValueError('one exact preceding dispatch derivation required')
    return result.replace(PREVIOUS, NEW, 1)
