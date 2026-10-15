"""Score human labels separately from implementation parity, retaining limitations."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import jiwer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--native", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
reference = json.loads(args.reference.read_text())
records = reference["records"]
native = json.loads(args.native.read_text())
facts_path = Path(__file__).with_name("critical-content.json")
facts = json.loads(facts_path.read_text())
assert reference["dataset_revision"] == facts["dataset_revision"]
assert len(records) == len(native) == 112
normalize = jiwer.Compose([
    jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(), jiwer.ReduceToListOfListOfWords(),
])

def contains(text, phrase):
    words, expected = normalize(text)[0], normalize(phrase)[0]
    return any(words[i:i + len(expected)] == expected for i in range(len(words)))

gates = []
slices = {}
for category in sorted({r["category"] for r in records}):
    indices = [i for i, r in enumerate(records) if r["category"] == category]
    labels = [records[i]["human"] for i in indices]
    slice_result = {"clips": len(indices)}
    for path, texts in [("reference", [r["text"] for r in records]),
                        ("native", [r["fulltext"] for r in native])]:
        score = jiwer.process_words(labels, [texts[i] for i in indices],
                                    reference_transform=normalize, hypothesis_transform=normalize)
        slice_result[path] = {"wer": score.wer, "substitutions": score.substitutions,
                              "deletions": score.deletions, "insertions": score.insertions}
    gates.append(slice_result["native"]["wer"] <= slice_result["reference"]["wer"] + 1e-12)
    slices[category] = slice_result

critical = []
for fact in facts["facts"]:
    for prefix in ["fluent-", "noisy-"]:
        wav = prefix + fact["id"] + ".wav"
        index = next(i for i, r in enumerate(records) if r["wav"] == wav)
        assert contains(records[index]["human"], fact["phrase"]), "fact must come from human label"
        result = dict(fact, wav=wav, reference_pass=contains(records[index]["text"], fact["phrase"]),
                      native_pass=contains(native[index]["fulltext"], fact["phrase"]))
        critical.append(result)
        if fact["release_gate"]:
            gates.append(result["reference_pass"] and result["native_pass"])

for r in records:
    assert hashlib.sha256((args.reference.parent / r["wav"]).read_bytes()).hexdigest() == r["sha256"]
matches = sum(a["text"] == b["fulltext"] for a, b in zip(records, native))
differences = [{"wav": a["wav"], "rate": a["rate"], "reference": a["text"], "native": b["fulltext"],
                "normalized_words_match": normalize(a["text"]) == normalize(b["fulltext"])}
               for a, b in zip(records, native) if a["text"] != b["fulltext"]]
gates.append(all(d["rate"] != 16000 and d["normalized_words_match"] for d in differences))
receipt = {
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "dataset_revision": reference["dataset_revision"], "clips": len(records),
    "normalizer": "lowercase-remove-punctuation-collapse-space-v1",
    "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
    "native_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(),
    "critical_facts_sha256": hashlib.sha256(facts_path.read_bytes()).hexdigest(),
    "raw_text_matches": matches, "slices": slices, "critical_content": critical,
    "experimental_resampling": True, "raw_text_differences": differences,
    "limitations": [r for r in critical if not r["reference_pass"] or not r["native_pass"]],
    "passed": all(gates),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k not in ["critical_content", "limitations"]}, indent=2))
assert receipt["passed"], "independent audio qualification failed; see receipt"
