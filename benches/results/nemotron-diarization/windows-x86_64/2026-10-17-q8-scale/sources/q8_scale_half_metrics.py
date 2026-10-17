"""Independent binary16 reference for every lane of the actual F16C helper."""
import hashlib
from windows_q8_tile_metrics_v2 import admit_half_table


def admit_scale_half_table(raw):
    if len(raw) != 65536 * 4 * 4:
        raise ValueError('all 65536 four-lane A-scale broadcasts required')
    lanes = [b''.join(raw[i * 16 + lane * 4:i * 16 + lane * 4 + 4]
        for i in range(65536)) for lane in range(4)]
    evidence = [admit_half_table(lane) for lane in lanes]
    if any(lane != lanes[0] for lane in lanes[1:]):
        raise ValueError('all four broadcast lanes must be byte-identical')
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
        'entries': 65536, 'lanes': evidence,
        'finite_values_bit_exact': True, 'infinities_bit_exact': True,
        'nan_classification_verified': True, 'nan_payload_equivalence_claimed': False}
