"""Bounded CPU identity/parity admission with the original frozen corpus scorer."""
from bounded_cpu_metrics import VERSION
from compare_stream_bounded import compare
from qualify_pooled_native import OLD_VERSION
from score_valid_mel_corpus import score_slice as valid_mel_score


def score_slice(item, gen2, native, q8_raw, native_raw, fp32, metadata, reference, expected_pcm_digest):
    parity = compare(gen2, native, native_raw, q8_raw)
    if parity['probabilities_bit_exact'] is not True:
        raise ValueError('bounded corpus requires byte-exact native/public probabilities')
    result = valid_mel_score(item, {**gen2, 'runtime_version': OLD_VERSION},
        {**native, 'runtime_version': OLD_VERSION}, q8_raw, native_raw,
        fp32, metadata, reference, expected_pcm_digest)
    result['wrapper_parity'] = parity
    result['runtime_version'] = VERSION
    return result
