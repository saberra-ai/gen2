"""Private packed-B-scale 2x2 Q8 helper; production bounds and tails intact."""
from q8_mx4_source import transform as control_source
from q8_tile_source import CLASS, ORIGINAL, PATH


OLD_DELTA = b'                uint64_t b_delta = ((uint64_t)B[ldb * (jj + 3) + l].d << 48) | ((uint64_t)B[ldb * (jj + 2) + l].d << 32) | ((uint64_t)B[ldb * (jj + 1) + l].d << 16) | (B[ldb * (jj + 0) + l].d);'
NEW_DELTA = b'                uint64_t b_delta = ((uint64_t)B[ldb * (jj + 1) + l].d << 16) | (B[ldb * (jj + 0) + l].d);'


def once(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError('one exact packed-Mx4 derivation site required')
    return raw.replace(old, new, 1)


def mx2_kernel(control):
    start = b'    // Templated functions for gemm of dimensions Mx4\n'
    end = b'\n#endif\n\n    template <int RM, int RN>'
    if control.count(start) != 1:
        raise ValueError('exact canonical Mx4 helper required')
    body = control.split(start, 1)[1].split(end, 1)[0]
    for old, new in [
            (b'void gemmMx4(', b'void gemmMx2('),
            (b'int64_t xtiles = (n - n0) / 4;', b'int64_t xtiles = (n - n0) / 2;'),
            (b'int64_t jj = n0 + job % xtiles * 4;', b'int64_t jj = n0 + job % xtiles * 2;'),
            (b'__m256 Cv[4][RM] = {};', b'__m256 Cv[2][RM] = {};'),
            (OLD_DELTA, NEW_DELTA),
            (b'// Convert delta values for four blocks to float values', b'// Convert delta values for two blocks to float values'),
            (b'// Computation of product of delta values for four blocks and replicate it across 256 bit lane',
             b'// Computation of product of delta values for two blocks and replicate it across 256 bit lane'),
            (b'for (int64_t j = 0; j < 4; ++j)', b'for (int64_t j = 0; j < 2; ++j)')]:
        body = once(body, old, new)
    for column in (2, 3):
        body = once(body, f'                __m256i bvec{column} = load(B + ldb * (jj + {column}) + l);\n'.encode(), b'')
        first = f'                    Cv[{column}][i] = madd('.encode()
        last = f'                                    Cv[{column}][i]);\n'.encode()
        if body.count(first) != 1 or body.count(last) != 1:
            raise ValueError('exact unused-column accumulation required')
        a = body.index(first); b = body.index(last, a) + len(last)
        body = once(body, body[a:b], b'')
    return b'#if defined(_WIN32) && defined(__AVX2__) && defined(__F16C__) && !defined(__AVX512F__)\n' + body + b'\n#endif\n\n'


def transform(raw):
    control = control_source(raw)
    head, body = control.split(CLASS, 1)
    body = once(body, b'                gemmMx4<2>(0, me, 0, ne);', b'                gemmMx2<2>(0, me, 0, ne);')
    body = once(body, b'  private:\n    void mnpack(', b'  private:\n' + mx2_kernel(control) + b'    void mnpack(')
    return once(head + CLASS + body, b'GEN2_Q8_MX4_EXPECTED', b'GEN2_Q8_MX2_EXPECTED')
