"""Full valid-row diagnostic comparisons; never an alternative quality gate."""
import numpy as np

from frontend_metrics import difference


def compare_original(probabilities, reference, samples):
    if type(samples) is not int or samples < 320:
        raise ValueError('invalid original-NeMo input sample count')
    expected = (samples // 160, 8)
    for values in [probabilities, reference]:
        if (values.shape != expected or not np.isfinite(values).all()
                or not ((0 <= values) & (values <= 1)).all()):
            raise ValueError('original-NeMo/reference valid probability rows differ')
    return difference(probabilities, reference, 1e-3)
