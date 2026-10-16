"""Measure Bark token-derived word intervals against licensed AMI manual words.

This is an alignment investigation, not an attribution acceptance receipt.
Overlapping reference words are ordered chronologically for the diagnostic WER.
No timestamps are assigned by proportional text offsets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET
import zipfile

ARCHIVE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ARCHIVE_SHA256 = "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def token_words(timestamps):
    words, text, start, end = [], "", None, None
    for token in timestamps:
        for fragment in re.findall(r"\s+|\S+", token["text"]):
            if fragment.isspace():
                if normalize(text):
                    words.append((normalize(text), start / 1000, end / 1000))
                text, start, end = "", None, None
            else:
                text += fragment
                start = token["startms"] if start is None else min(start, token["startms"])
                end = token["endms"] if end is None else max(end, token["endms"])
    if normalize(text):
        words.append((normalize(text), start / 1000, end / 1000))
    return words


def evaluate(reference, predicted):
    n, m = len(reference), len(predicted)
    if not n:
        raise ValueError("empty reference")
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        costs[i][0] = i
    for j in range(m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i][j] = min(costs[i-1][j] + 1, costs[i][j-1] + 1,
                              costs[i-1][j-1] + (reference[i-1][0] != predicted[j-1][0]))
    i, j, matched = n, m, []
    deletions = insertions = substitutions = 0
    while i or j:
        if i and j and costs[i][j] == costs[i-1][j-1] + (reference[i-1][0] != predicted[j-1][0]):
            if reference[i-1][0] == predicted[j-1][0]:
                matched.append((reference[i-1], predicted[j-1]))
            else:
                substitutions += 1
            i, j = i-1, j-1
        elif i and costs[i][j] == costs[i-1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    result = {"reference_words": n, "predicted_words": m, "matched_words": len(matched),
              "chronological_merged_wer": costs[n][m] / n,
              "deletions": deletions, "insertions": insertions, "substitutions": substitutions}
    for name, index in [("start", 1), ("end", 2)]:
        errors = sorted(abs(a[index] - b[index]) for a, b in matched)
        result[f"matched_{name}_error_median_seconds"] = statistics.median(errors) if errors else None
        result[f"matched_{name}_error_p95_seconds"] = errors[int(.95 * (len(errors)-1))] if errors else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--transcription", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meeting", default="EN2002d")
    parser.add_argument("--start", type=float, default=2132.5)
    parser.add_argument("--seconds", type=float, default=60.)
    args = parser.parse_args()
    archive_hash = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    if archive_hash != ARCHIVE_SHA256:
        parser.error("manual-annotation archive differs from pinned 1.6.2")
    with zipfile.ZipFile(args.annotations) as archive:
        names = sorted(n for n in archive.namelist() if Path(n).name.startswith(args.meeting + ".") and n.endswith(".words.xml"))
        if not names:
            parser.error("meeting missing from annotation archive")
        words = []
        for name in names:
            for word in ET.fromstring(archive.read(name)):
                if word.tag == "w" and "starttime" in word.attrib and "endtime" in word.attrib:
                    start, end = float(word.attrib["starttime"]), float(word.attrib["endtime"])
                    if args.start <= start and end <= args.start + args.seconds and normalize(word.text or ""):
                        words.append({"speaker": Path(name).name.split(".")[1], "start_seconds": start-args.start, "end_seconds": end-args.start, "text": word.text})
    words.sort(key=lambda w: (w["start_seconds"], w["speaker"]))
    transcription = json.loads(args.transcription.read_text())
    predicted = token_words(transcription["timestamps"])
    report = evaluate([(normalize(w["text"]), w["start_seconds"], w["end_seconds"]) for w in words], predicted)
    result = {"schema_version": 1, "source": ARCHIVE_URL, "archive_sha256": archive_hash,
              "license": "CC-BY-4.0", "meeting": args.meeting, "start_seconds": args.start,
              "duration_seconds": args.seconds, "files": names, "reference_words": words,
              "transcription_sha256": hashlib.sha256(args.transcription.read_bytes()).hexdigest(),
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "word_intervals": "union of contributing whole token intervals; no proportional offsets",
              "metrics": report, "release_qualified": False}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
