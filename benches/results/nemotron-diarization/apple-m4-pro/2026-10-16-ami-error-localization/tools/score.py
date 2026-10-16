"""Overlap-inclusive frame DER, 10 ms frames, zero collar, optimal global mapping."""
import argparse
import itertools
import json
import math
from pathlib import Path


def der(reference, hypothesis, duration):
    if not math.isfinite(duration) or not 0 < duration <= 7200:
        raise ValueError("duration must be finite and between zero and two hours")
    for s in reference + hypothesis:
        if not all(math.isfinite(s[k]) for k in ["start", "end"]) or s["end"] < s["start"]:
            raise ValueError("invalid segment interval")
    ref_ids = sorted({str(s["speaker"]) for s in reference})
    hyp_ids = sorted({str(s["speaker"]) for s in hypothesis})
    n = max(len(ref_ids), len(hyp_ids))
    if n > 8:
        raise ValueError("this qualification scorer supports at most eight speakers")
    matrix = [[0. for _ in range(n)] for _ in range(n)]
    total = missed = false = paired = 0.
    # A finite recording is the explicit scoring UEM. Clip annotations at its edges.
    for frame in range(math.ceil(duration * 100)):
        start = frame / 100
        weight = min(.01, duration - start)
        midpoint = start + weight / 2
        active_ref = {ref_ids.index(str(s["speaker"])) for s in reference
                      if s["start"] <= midpoint < s["end"]}
        active_hyp = {hyp_ids.index(str(s["speaker"])) for s in hypothesis
                      if s["start"] <= midpoint < s["end"]}
        for r in active_ref:
            for h in active_hyp:
                matrix[r][h] += weight
        total += len(active_ref) * weight
        missed += max(0, len(active_ref) - len(active_hyp)) * weight
        false += max(0, len(active_hyp) - len(active_ref)) * weight
        paired += min(len(active_ref), len(active_hyp)) * weight
    if total == 0:
        raise ValueError("DER is undefined without reference speech")
    best = max((sum(matrix[r][h] for r, h in enumerate(mapping))
                for mapping in itertools.permutations(range(n))), default=0.)
    confusion = max(0., paired - best)
    return {"der": round((missed + false + confusion) / total, 9),
            "missed_speaker_seconds": round(missed, 6),
            "false_alarm_speaker_seconds": round(false, 6),
            "confusion_speaker_seconds": round(confusion, 6),
            "reference_speaker_seconds": round(total, 6),
            "reference_speakers": len(ref_ids), "hypothesis_speakers": len(hyp_ids),
            "frame_seconds": .01, "collar_seconds": 0, "overlap_included": True,
            "scoring_duration_seconds": duration}


def read_rttm(path):
    entries = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        f = line.split()
        if len(f) != 10 or f[0] != "SPEAKER":
            raise ValueError("invalid RTTM row")
        entries.append({"start": float(f[3]), "end": float(f[3]) + float(f[4]), "speaker": f[7]})
    return entries


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("hypothesis", type=Path)
    p.add_argument("--duration", type=float, required=True)
    a = p.parse_args()
    print(json.dumps(der(read_rttm(a.reference), json.loads(a.hypothesis.read_text())["segments"], a.duration)))
