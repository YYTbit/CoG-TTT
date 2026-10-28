import copy

import numpy as np
import pytest

from cog_ttt.evaluation import summarize


def mlgym_records():
    records = []
    for seed in range(5):
        for task in range(13):
            for method in ("No-TTT", "CoG-TTT"):
                value = 1.0
                if method == "CoG-TTT":
                    value = 2.0 if task < 10 else 1.00004 if task == 10 else 0.5
                records.append({"stage": "final_evaluation", "backbone": "software-check", "benchmark": "mlgym",
                                "revision": "metric-unit-check", "data_kind": "benchmark", "method": method,
                                "seed": seed, "task_id": str(task),
                                "metrics": {"native": {"value": value, "unit": "score", "direction": "higher",
                                                       "reported_precision": 3}}})
    return records


def test_win13_includes_half_credit_and_percentage_scale():
    result = summarize(mlgym_records())[0]
    assert result["mean"] == pytest.approx(100 * 10.5 / 13)
    assert result["sample_std"] == 0
    assert result["seed_credits"] == dict.fromkeys(range(5), 10.5)
    assert result["unit"] == "percent"


def test_win13_uses_raw_rate_precision_and_metric_direction():
    records = mlgym_records()
    for record in records:
        if record["task_id"] == "0":
            record["metrics"]["native"].update(value=0.50001 if record["method"] == "CoG-TTT" else 0.5,
                                                unit="rate", reported_precision=4)
        if record["task_id"] == "1":
            record["metrics"]["native"]["direction"] = "lower"
    result = summarize(records)[0]
    assert result["mean"] == pytest.approx(100 * 9 / 13)


def test_win13_reference_is_independent_of_pairwise_baseline():
    result = summarize(mlgym_records(), baseline="SDPO")[0]
    assert result["reference_method"] == "No-TTT"
    assert result["mean"] == pytest.approx(100 * 10.5 / 13)


def test_win13_standard_deviation_uses_seed_percentages():
    records = mlgym_records()
    for record in records:
        if record["method"] == "CoG-TTT" and record["task_id"] == "0" and record["seed"] < 2:
            record["metrics"]["native"]["value"] = 0.5
    scores = [100 * credit / 13 for credit in [9.5, 9.5, 10.5, 10.5, 10.5]]
    result = summarize(records)[0]
    assert result["mean"] == pytest.approx(np.mean(scores))
    assert result["sample_std"] == pytest.approx(np.std(scores, ddof=1))


def test_win13_rejects_missing_or_inconsistent_reported_precision():
    records = mlgym_records()
    missing = copy.deepcopy(records)
    missing[0]["metrics"]["native"].pop("reported_precision")
    with pytest.raises(ValueError, match="reported_precision"):
        summarize(missing)
    records[0]["metrics"]["native"]["reported_precision"] = 4
    with pytest.raises(ValueError, match="same reported precision"):
        summarize(records)


def test_win13_rejects_different_native_metric_names():
    records = mlgym_records()
    record = next(row for row in records if row["method"] == "CoG-TTT")
    record["metrics"] = {"different_metric": record["metrics"]["native"]}
    with pytest.raises(ValueError, match="same native metric"):
        summarize(records)
