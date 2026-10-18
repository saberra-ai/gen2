"""Independent replay of complete saved A packing outputs."""
import hashlib
from windows_q8_tile_metrics_v2 import admit_half_table


def admit_packing(payload, scales):
    expected = bytes((i + 17 * j) & 255 for i in range(65536) for j in range(32))
    if payload != expected:
        raise ValueError('complete deterministic packed payload differs')
    half = admit_half_table(scales)
    return {'blocks': 65536, 'payload_bytes': len(payload),
        'payload_sha256': hashlib.sha256(payload).hexdigest(),
        'scales_bytes': len(scales), 'scales_sha256': hashlib.sha256(scales).hexdigest(),
        'half_reference': half,
        'scope': 'Complete payload bytes and independent finite/infinity/NaN-class scale checks; actual benchmark additionally checks lookup bit identity, padded source immutability, destination guards and ten bound decisions.'}
