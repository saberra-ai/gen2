"""Fixed Q8 1x4 dispatch candidate using the unchanged canonical Mx4 helper."""
from q8_mx4_source import transform as control_source
from q8_tile_source import CLASS, ORIGINAL, PATH


def transform(raw):
    control = control_source(raw)
    head, body = control.split(CLASS, 1)
    old = b'                gemmMx4<2>(0, me, 0, ne);'
    if body.count(old) != 1 or body.count(b'GEN2_Q8_MX4_EXPECTED') != 1:
        raise ValueError('exact production dispatch and compilation guard required')
    body = body.replace(old, b'                gemmMx4<1>(0, me, 0, ne);', 1)
    body = body.replace(b'GEN2_Q8_MX4_EXPECTED', b'GEN2_Q8_ROW_EXPECTED', 1)
    return head + CLASS + body
