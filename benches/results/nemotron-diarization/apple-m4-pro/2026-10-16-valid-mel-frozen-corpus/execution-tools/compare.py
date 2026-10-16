"""Compare original FP32 and native Q8 with identical external thresholding."""
import argparse
import json
from pathlib import Path
import numpy as np

from artifacts import digest
from score import der, read_rttm


def segments(probabilities):
    result = []
    for speaker in range(8):
        active = probabilities[:, speaker] > .5
        changes = np.diff(np.r_[False, active, False].astype(int))
        result.extend({"start": start / 100, "end": end / 100, "speaker": speaker + 1}
                      for start, end in zip(np.where(changes == 1)[0], np.where(changes == -1)[0]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--rttm", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    native = json.loads(a.native.read_text())
    metadata = json.loads(a.reference.with_suffix(".json").read_text())
    fp32 = np.load(a.reference.with_suffix(".npy"), allow_pickle=False)
    q8 = np.fromfile(str(a.native) + ".f32", dtype=np.float32).reshape(-1, 8)
    samples = metadata["audio_samples"]
    # Record, don't silently normalize away, the pinned native centered-STFT tail.
    # The native finish pads half an FFT window and emits floor(N/160)+1 frames;
    # Transformers masks this exact-second fixture to N/160 valid 10 ms frames.
    if samples % 160 or fp32.shape != (samples // 160, 8):
        raise ValueError("this N0 comparison requires an exact 10 ms reference fixture")
    if q8.shape != (samples // 160 + 1, 8) or native["frames"] != len(q8):
        raise ValueError("native output does not match the investigated tail-frame rule")
    if not np.isfinite(q8).all() or not np.isfinite(fp32).all():
        raise ValueError("non-finite frame probabilities")
    duration = samples / 16000
    if native["geometry"] != metadata["geometry"]:
        raise ValueError("native/reference streaming geometry differs")
    if abs(native["audio_seconds"] - duration) > 1e-9:
        raise ValueError("native/reference audio durations differ")
    reference = read_rttm(a.rttm)
    original_score = der(reference, segments(fp32), duration)
    quantized_score = der(reference, segments(q8), duration)
    default_score = der(reference, native["segments"], duration)
    delta = quantized_score["der"] - original_score["der"]
    output = {"scope": "one licensed 60-second AMI smoke fixture; not release qualification",
              "native_sha256": digest(a.native), "reference_sha256": digest(a.reference.with_suffix('.npy')),
              "native_probabilities_sha256": digest(Path(str(a.native) + ".f32")),
              "reference_metadata_sha256": digest(a.reference.with_suffix(".json")),
              "geometry": metadata["geometry"],
              "rttm_sha256": digest(a.rttm),
              "shape_difference": {"native": list(q8.shape), "reference": list(fp32.shape),
                                   "extra_native_tail_frames": 1, "frame_seconds": .01,
                                   "scoring_uem_seconds": [0, duration],
                                   "interpretation": "native finish emits a padded terminal frame; retain it in raw evidence, clip presentation to audio duration"},
              "channel_mae_on_common_real_frames": float(np.abs(fp32 - q8[:-1]).mean()),
              "channel_max_error_on_common_real_frames": float(np.abs(fp32 - q8[:-1]).max()),
              "shared_postprocessing": "sigmoid probability > 0.5; no hysteresis/padding/min-duration",
              "fp32": original_score, "q8": quantized_score,
              "native_default_postprocessing": default_score,
              "q8_minus_fp32_absolute_der": delta,
              "q8_degradation_limit": .02, "smoke_quality_passed": delta <= .02}
    a.output.write_text(json.dumps(output, indent=2) + "\n")
    if not output["smoke_quality_passed"]:
        raise ValueError("Q8 exceeds the proposed two absolute DER point smoke degradation limit")
    print(a.output)


if __name__ == "__main__":
    main()
