"""Compare public-interface results and human-reference WER/CER per language."""
import argparse
import collections
import hashlib
import json
import pathlib
import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text).split())


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = row
    return previous[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--native", type=pathlib.Path, required=True)
    parser.add_argument("--forced", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    native = json.loads(args.native.read_text(encoding="utf-8"))
    forced = json.loads(args.forced.read_text(encoding="utf-8"))
    records = reference["records"]
    assert len(forced) == len(records)
    assert len(records) == len(native) and len(records) >= 140
    by_language = collections.defaultdict(lambda: collections.Counter())
    by_slice = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    critical = collections.defaultdict(collections.Counter)
    mismatches = []
    for record, actual, forced_actual in zip(records, native, forced):
        forced_transcript = forced_actual["transcription"]
        if forced_transcript["text"] != record["forced_text"] or forced_transcript["token_ids"] != record["forced_token_ids"] or forced_actual["language"] != record["language"] or forced_actual["detected_language"] != record["detected_language"] or (forced_actual["completion"] == "EndOfSpeech") != record["forced_complete"]:
            mismatches.append(record["id"] + "-forced")
        transcript = actual["transcription"]
        if transcript["text"] != record["text"] or transcript["token_ids"] != record["token_ids"] or actual["language"] != record["detected_language"] or (actual["completion"] == "EndOfSpeech") != record["complete"]:
            mismatches.append(record["id"])
        human, text = normalize(record["human_text"]), normalize(transcript["text"])
        score = by_language[record["language"]]
        metrics = dict(clips=1, words=len(human.split()), word_errors=distance(human.split(), text.split()),
            characters=len(human), character_errors=distance(human, text),
            language_correct=int(actual["language"] == record["language"]), incomplete=int(actual["completion"] != "EndOfSpeech"))
        if record["slice"] in ("parity", "held-out-quality"): score.update(metrics)
        by_slice[record["slice"]][record["language"]].update(metrics)
        if record["slice"] in ("parity", "held-out-quality"):
            for tag in record.get("critical_content", []): critical[tag].update(metrics)
    def rates(counts):
        return {**counts, "wer": counts["word_errors"] / max(1, counts["words"]), "cer": counts["character_errors"] / max(1, counts["characters"])}
    result = {"normalizer": "NFC-lower-alphanumeric-space-v1", "clips": len(records),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "forced_sha256": hashlib.sha256(args.forced.read_bytes()).hexdigest(),
        "native_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(),
        "raw_parity_mismatches": mismatches,
        "languages": {language: {**counts, "wer": counts["word_errors"] / counts["words"],
            "cer": counts["character_errors"] / counts["characters"]} for language, counts in by_language.items()},
        "slices": {name:{language:rates(counts) for language,counts in languages.items()} for name,languages in by_slice.items()},
        "critical_content": {tag:rates(counts) for tag,counts in critical.items()},
        "passed": not mismatches}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    assert result["passed"], f"raw parity failed: {mismatches}"
