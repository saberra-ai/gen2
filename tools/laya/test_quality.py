import copy
import unittest
from evaluate_quality import evaluate, measurement, metrics, validate_dataset


class QualityTests(unittest.TestCase):
    def test_known_binary_metrics_and_confidence_ties(self):
        rows = [measurement(p, y, "yes_no") for p, y in [([.9,.1],0),([.2,.8],1),([.6,.4],0),([.6,.4],1)]]
        result = metrics(rows)
        self.assertEqual(result["accuracy"], .75)
        self.assertAlmostEqual(result["multiclass_brier"], .285)
        self.assertAlmostEqual(result["ece_10_bins"], .125)
        self.assertEqual([r["count"] for r in result["risk_coverage"]], [1,2,4])
        self.assertIsNone(result["score_mae"])

    def test_score_uses_expectation_not_winning_class(self):
        row = measurement([.2,.3,.5], 2, "score")
        self.assertEqual(row["correct"], 1)
        self.assertAlmostEqual(row["absolute_error"], .7)

    def test_reject_nonfinite_and_incomplete_distributions(self):
        for p in ([float("nan"),0],[-.1,1.1],[.2,.3],[]):
            with self.assertRaises(ValueError):
                measurement(p, 0, "choice")

    def fixture(self):
        question = dict(type="yes_no",instructions="Holds?",false_label="false",true_label="true")
        records = [dict(id=name, split=split, group="check", labels={"q":label}, heuristic={"q":0},
                        request=dict(state=dict(kind="text",value=name),questions=[["q",question]]))
                   for name,split,label in [("a","calibration",0),("b","holdout",1),("c","holdout",1)]]
        dataset = dict(schema="gen2-laya-quality/v1",dataset_id="synthetic-test",records=records)
        predictions = [dict(type="host",manifest_sha256="fixture")]
        for record in records:
            predictions.append(dict(type="case",name=record["id"],request=record["request"], result=dict(manifest_sha256="fixture",
                answers=[["q",dict(value=dict(type="yes_no",probability_true=.9), probabilities=[.1,.9],answer_confidence=.9)]])))
        predictions.append(dict(type="complete"))
        return dataset,predictions

    def test_baseline_never_learns_holdout_majority(self):
        dataset,predictions = self.fixture()
        report = evaluate(dataset,predictions)["metrics"]["overall"]
        self.assertEqual(report["model"]["accuracy"],1)
        self.assertEqual(report["majority"]["accuracy"],0)
        self.assertEqual(report["heuristic"]["accuracy"],0)

    def test_changed_prediction_input_and_split_overlap_fail(self):
        dataset,predictions = self.fixture()
        changed=copy.deepcopy(predictions)
        changed[2]["request"]["state"]["value"]="different"
        with self.assertRaises(ValueError):evaluate(dataset,changed)
        dataset["records"][1]["request"]["state"]["value"]="a"
        with self.assertRaises(ValueError):validate_dataset(dataset)

    def test_inconsistent_typed_value_fails(self):
        dataset,predictions=self.fixture()
        predictions[2]["result"]["answers"][0][1]["value"]["probability_true"]=.1
        with self.assertRaises(ValueError):evaluate(dataset,predictions)


if __name__ == "__main__":
    unittest.main()
