"""Evaluate labeled Laya holdouts; never use holdout labels to fit a baseline.

Input predictions are complete laya_benchmark JSONL reports produced from a
prepare_evaluation.py plan. Reports contain private input text: store them in
an appropriate local data directory, not in source control.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def question_key(question):
    q = dict(question)
    if not isinstance(q["instructions"], str):
        q["instructions"] = json.dumps(q["instructions"], ensure_ascii=False, allow_nan=False)
    q.pop("option_order", None)  # presentation does not change the label space
    if q["type"] == "choice":
        fields = [q["type"], q["instructions"], [[o["label"], o.get("description")] for o in q["options"]]]
    elif q["type"] == "score":
        fields = [q["type"], q["instructions"], q["levels"]]
    elif q["type"] == "yes_no":
        fields = [q["type"], q["instructions"], q["false_label"], q["true_label"], q.get("false_description"), q.get("true_description")]
    else:
        raise ValueError("unknown question type")
    return canonical(fields)


def width(question):
    return len(question["options"]) if question["type"] == "choice" else len(question["levels"]) if question["type"] == "score" else 2


def validate_dataset(dataset):
    if dataset.get("schema") != "gen2-laya-quality/v1" or not dataset.get("dataset_id"):
        raise ValueError("dataset schema or identity missing")
    ids, states, templates = set(), defaultdict(set), {}
    records = dataset["records"]
    if not records:
        raise ValueError("empty dataset")
    for record in records:
        identifier = record["id"]
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("missing or duplicate sample ID")
        ids.add(identifier)
        split = record["split"]
        if split not in ("calibration", "holdout") or not isinstance(record.get("group"), str) or not record["group"]:
            raise ValueError("each record needs a split and stable question group")
        # Detect exact shared evidence even if sample IDs differ. Near-duplicate
        # and entity/time leakage still require the dataset owner's review.
        states[split].add(canonical(record["request"]["state"]["value"]))
        questions = record["request"]["questions"]
        if not questions or set(record["labels"]) != {qid for qid, _ in questions} or len({qid for qid, _ in questions}) != len(questions):
            raise ValueError("labels must cover each distinct question exactly once")
        for qid, question in questions:
            k, label = width(question), record["labels"][qid]
            if type(label) is not int or not 0 <= label < k:
                raise ValueError("labels must be canonical integer option/level indices")
            group = (record["group"], qid)
            key = question_key(question)
            if group in templates and templates[group] != key:
                raise ValueError("question meaning or label order changed within a group")
            templates[group] = key
            heuristic = record.get("heuristic", {}).get(qid)
            if heuristic is not None and (type(heuristic) is not int or not 0 <= heuristic < k):
                raise ValueError("heuristic labels must be canonical indices")
    if not states["calibration"] or not states["holdout"]:
        raise ValueError("separate calibration and holdout records are required")
    if states["calibration"] & states["holdout"]:
        raise ValueError("calibration/holdout evidence overlap")
    return records


def metrics(rows):
    if not rows:
        return None
    count = len(rows)
    bins = defaultdict(list)
    for row in rows:
        bins[min(int(row["confidence"]*10), 9)].append(row)
    ece = sum(len(group)/count * abs(sum(r["correct"] for r in group)/len(group) - sum(r["confidence"] for r in group)/len(group)) for group in bins.values())
    ordered = sorted(rows, key=lambda r: -r["confidence"])
    thresholds = sorted({ordered[min(count-1, math.ceil(count*i/10)-1)]["confidence"] for i in range(1, 11)}, reverse=True)
    risk = []
    for threshold in thresholds:
        accepted = [row for row in rows if row["confidence"] >= threshold]
        risk.append(dict(threshold=threshold, coverage=len(accepted)/count, error_rate=1-sum(r["correct"] for r in accepted)/len(accepted), count=len(accepted)))
    scores = [r["absolute_error"] for r in rows if r["absolute_error"] is not None]
    return dict(count=count, accuracy=sum(r["correct"] for r in rows)/count,
                score_mae=sum(scores)/len(scores) if scores else None,
                multiclass_brier=sum(r["brier"] for r in rows)/count, ece_10_bins=ece, risk_coverage=risk)


def measurement(probabilities, label, kind, confidence=None):
    if not probabilities or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities) or abs(sum(probabilities)-1) > 1e-5:
        raise ValueError("invalid probability distribution")
    winner = max(range(len(probabilities)), key=probabilities.__getitem__)
    confidence = max(probabilities) if confidence is None else confidence
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("invalid calibrated confidence")
    return dict(correct=int(winner == label), confidence=confidence,
                absolute_error=abs(sum(i*p for i, p in enumerate(probabilities))-label) if kind == "score" else None,
                brier=sum((p-int(i == label))**2 for i, p in enumerate(probabilities)))


def evaluate(dataset, predictions):
    records = validate_dataset(dataset)
    counts = defaultdict(Counter)
    for record in records:
        if record["split"] == "calibration":
            for qid, label in record["labels"].items():
                counts[(record["group"], qid)][label] += 1
    if predictions[0].get("type") != "host" or predictions[-1].get("type") != "complete":
        raise ValueError("incomplete native prediction report")
    cases = {}
    for item in predictions:
        if item.get("type") == "case":
            if item["name"] in cases:
                raise ValueError("duplicate prediction sample")
            cases[item["name"]] = item
    if set(cases) != {r["id"] for r in records}:
        raise ValueError("predictions must cover the exact dataset")
    collected = defaultdict(lambda: defaultdict(list))
    heuristic_missing = 0
    for record in records:
        case = cases[record["id"]]
        original, actual = record["request"], case["request"]
        if canonical([original["state"]["kind"], original["state"]["value"]]) != canonical([actual["state"]["kind"], actual["state"]["value"]]):
            raise ValueError("prediction state differs from the labeled state")
        expected_questions, actual_questions = original["questions"], actual["questions"]
        if len(expected_questions) != len(actual_questions) or any(a != b or question_key(q) != question_key(r) or q.get("option_order") != r.get("option_order") for (a,q),(b,r) in zip(expected_questions, actual_questions)):
            raise ValueError("prediction question differs from the labeled question")
        result = case["result"]
        if result["manifest_sha256"] != predictions[0]["manifest_sha256"]:
            raise ValueError("mixed model manifests in prediction report")
        if [qid for qid, _ in result["answers"]] != [qid for qid, _ in expected_questions]:
            raise ValueError("missing or reordered prediction answers")
        if record["split"] != "holdout":
            continue
        for (qid, question), (_, answer) in zip(expected_questions, result["answers"]):
            k, kind, label = width(question), question["type"], record["labels"][qid]
            if len(answer["probabilities"]) != k or answer["value"]["type"] != kind:
                raise ValueError("prediction answer shape/type mismatch")
            probabilities = answer["probabilities"]
            measured = measurement(probabilities, label, kind, answer["answer_confidence"])
            winner = max(range(k), key=probabilities.__getitem__)
            value = answer["value"]
            if kind == "choice" and (value.get("index") != winner or canonical(value.get("label")) != canonical(question["options"][winner]["label"])):
                raise ValueError("choice value disagrees with its distribution")
            scalar = value.get("expectation") if kind == "score" else value.get("probability_true") if kind == "yes_no" else None
            expected_scalar = sum(i*p for i,p in enumerate(probabilities)) if kind == "score" else probabilities[1] if kind == "yes_no" else None
            if kind != "choice" and (type(scalar) not in (int,float) or not math.isfinite(scalar) or abs(scalar-expected_scalar) > 1e-5):
                raise ValueError("numeric answer disagrees with its distribution")
            training = counts[(record["group"], qid)]
            if not training:
                raise ValueError("holdout group has no calibration-only baseline data")
            majority = max(range(k), key=lambda i: training[i])
            bucket = "1-2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
            rows = {
                "model": measured,
                "majority": measurement([float(i == majority) for i in range(k)], label, kind),
                "calibration_prior": measurement([training[i]/sum(training.values()) for i in range(k)], label, kind),
            }
            heuristic = record.get("heuristic", {}).get(qid)
            if heuristic is not None:
                rows["heuristic"] = measurement([float(i == heuristic) for i in range(k)], label, kind)
            else:
                heuristic_missing += 1
            for name, row in rows.items():
                for scope in ("overall", "type/"+kind, "options/"+bucket, "group/"+canonical([record["group"],qid])):
                    collected[scope][name].append(row)
    return dict(dataset_id=dataset["dataset_id"], dataset_purpose=dataset.get("purpose", "unspecified"), manifest_sha256=predictions[0]["manifest_sha256"],
                metrics={scope: {name: metrics(rows) for name, rows in methods.items()} for scope, methods in collected.items()},
                heuristic_missing_decisions=heuristic_missing,
                definitions=dict(accuracy="argmax over canonical options; first index wins ties",
                                 score_mae="absolute error of expected ordinal index",
                                 brier="sum over all classes of (probability - one_hot_label)^2; range 0..2",
                                 ece="10 equal-width bins of answer_confidence versus argmax correctness",
                                 risk_coverage="accept confidence >= threshold; ties remain together",
                                 baselines="majority and class prior fit only on calibration split; heuristic supplied by dataset owner"),
                qualification="Metrics only; application-specific acceptance thresholds and dataset representativeness require owner review.",
                leakage_check="Exact evidence overlap rejected; semantic/entity/time leakage is not automatically proven absent.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "predictions", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    dataset_bytes, prediction_bytes = a.dataset.read_bytes(), a.predictions.read_bytes()
    result = evaluate(json.loads(dataset_bytes), [json.loads(line) for line in prediction_bytes.splitlines() if line.strip()])
    result.update(dataset_sha256=hashlib.sha256(dataset_bytes).hexdigest(), predictions_sha256=hashlib.sha256(prediction_bytes).hexdigest())
    with a.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
