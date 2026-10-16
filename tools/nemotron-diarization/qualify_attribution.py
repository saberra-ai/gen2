"""Measure actual Pnola speaker-word output against licensed AMI manual words.

The conservative cpWER treats unknown/overlap words as an unattributed stream:
they cannot be relabeled as a known person to improve the score. It preserves
cleanup omissions and never measures only the successful attribution subset.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from qualify_alignment import normalize


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        row = [i]
        for j, other in enumerate(hypothesis, 1):
            row.append(min(previous[j] + 1, row[j-1] + 1, previous[j-1] + (word != other)))
        previous = row
    return previous[-1]


def score(reference, attribution):
    if attribution["status"] != "complete" or not attribution["complete"]:
        raise ValueError("requires a completed native timeline")
    words = attribution["words"]
    reference = [w for w in reference if normalize(w["text"])]
    if not reference:
        raise ValueError("requires reference words")
    refs = sorted({w["speaker"] for w in reference})
    hyp_ids = sorted({w["attribution"]["speaker"] for w in words if w["attribution"]["kind"] == "speaker"})
    n = max(len(refs), len(hyp_ids))
    if n > 8:
        raise ValueError("at most eight known speakers")
    ref_words = [[normalize(w["text"]) for w in reference if w["speaker"] == speaker] for speaker in refs]
    hyp_words = [[normalize(w["text"]) for w in words if w["attribution"]["kind"] == "speaker" and w["attribution"]["speaker"] == speaker and normalize(w["text"])] for speaker in hyp_ids]
    ref_words += [[] for _ in range(n - len(ref_words))]
    hyp_words += [[] for _ in range(n - len(hyp_words))]
    matrix = [[edit_distance(a, b) for b in hyp_words] for a in ref_words]
    assignment = min(itertools.permutations(range(n)), key=lambda p: sum(matrix[r][h] for r, h in enumerate(p)))
    known_errors = sum(matrix[r][h] for r, h in enumerate(assignment))
    unknown = sum(bool(normalize(w["text"])) for w in words if w["attribution"]["kind"] == "unknown")
    overlap = sum(bool(normalize(w["text"])) for w in words if w["attribution"]["kind"] == "overlap")
    predicted = [normalize(w["text"]) for w in words if normalize(w["text"])]
    lexical_errors = edit_distance([normalize(w["text"]) for w in reference], predicted)
    # A separate joint speaker-tagged edit score charges one substitution when
    # either word or speaker is wrong. Unknown/overlap can never match a person.
    tagged_reference=[(normalize(w["text"]),("speaker",w["speaker"])) for w in reference]
    tagged_best=None
    for permutation in itertools.permutations(range(n)):
        mapping={hyp_ids[h]:refs[r] for r,h in enumerate(permutation) if h<len(hyp_ids) and r<len(refs)}
        tagged_hypothesis=[(normalize(w["text"]),("speaker",mapping[a["speaker"]]) if a["kind"]=="speaker" and a["speaker"] in mapping else ("unattributed",a["kind"])) for w in words if normalize(w["text"]) for a in [w["attribution"]]]
        errors=edit_distance(tagged_reference,tagged_hypothesis)
        if tagged_best is None or errors<tagged_best[0]:tagged_best=(errors,mapping)
    return {"reference_words": len(reference), "predicted_words": len(predicted),
            "chronological_merged_wer": lexical_errors / len(reference),
            "known_speaker_words": sum(map(len, hyp_words)), "unknown_words": unknown, "overlap_words": overlap,
            "known_speaker_stream_edit_errors": known_errors,
            "conservative_cpwer": (known_errors + unknown + overlap) / len(reference),
            "speaker_tagged_word_error_rate":tagged_best[0]/len(reference),
            "speaker_tagged_word_edit_errors":tagged_best[0],
            "speaker_tagged_metric_definition":"minimum chronological Levenshtein on (word, speaker) pairs over global known-speaker permutations; word/speaker mismatch costs one substitution; unknown/overlap never match a reference person",
            "speaker_tagged_mapping":tagged_best[1],
            "metric_definition": "minimum concatenated per-speaker Levenshtein errors with global permutation of known IDs, plus unknown/overlap words as an unmapped insertion stream; reference omissions remain deletions",
            "speaker_mapping": {str(hyp_ids[h]): refs[r] for r, h in enumerate(assignment) if r < len(refs) and h < len(hyp_ids)},
            "short_reference_words_400ms_or_less": sum(w["end_seconds"] - w["start_seconds"] <= .4 for w in reference)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reference = json.loads(args.reference.read_text())
    result = json.loads(args.result.read_text())
    metrics = score(reference["reference_words"], result["attribution"])
    output = {"schema_version": 1, "release_qualified": False, "scope": "single-clip alignment/attribution investigation",
              "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "result_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "metrics": metrics}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
