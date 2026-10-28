import math
from collections import defaultdict
from decimal import Decimal, localcontext

import numpy as np


def native_value(metric):
    value, unit = metric["value"], metric["unit"]
    if isinstance(value, bool) or not math.isfinite(value) or unit not in {"rate", "score", "count"}:
        raise ValueError("Native evaluation metrics declare finite values and units")
    if unit == "rate" and not 0 <= value <= 1:
        raise ValueError("Native rates lie in [0,1] before percentage conversion")
    if unit == "count" and (value < 0 or int(value) != value):
        raise ValueError("Native counts are nonnegative integers")
    return float(value) * (100 if unit == "rate" else 1)


def index_records(records):
    indexed = {}
    for record in records:
        if record.get("data_kind") == "software_fixture":
            raise ValueError("Software-fixture outputs are excluded from benchmark summaries")
        key = (record["backbone"], record["benchmark"], record["revision"], record["method"],
               record["seed"], record["task_id"])
        if key in indexed:
            raise ValueError("Final evaluation records duplicate a method-seed-task identity")
        if record.get("stage") != "final_evaluation":
            raise ValueError("Benchmark summaries use independent final evaluation records")
        for metric in record["metrics"].values():
            native_value(metric)
        indexed[key] = record
    return indexed


def paired_interval(first, second, draws=10000, seed=0):
    if set(first) != set(second) or not first:
        raise ValueError("Paired comparisons use exactly matching seed-task identities")
    if isinstance(draws, bool) or int(draws) != draws or draws < 1:
        raise ValueError("The paired bootstrap uses a positive integer draw count")
    groups = defaultdict(list)
    for (replicate, task), value in sorted(first.items()):
        groups[replicate].append(value - second[(replicate, task)])
    generator = np.random.default_rng(seed)
    samples = np.zeros(int(draws))
    for differences in groups.values():
        differences = np.asarray(differences)
        selected = generator.integers(0, len(differences), size=(int(draws), len(differences)))
        samples += differences[selected].mean(axis=1) / len(groups)
    return {"difference": float(np.mean([np.mean(values) for values in groups.values()])),
            "lower": float(np.quantile(samples, 0.025)), "upper": float(np.quantile(samples, 0.975)),
            "draws": int(draws), "stratified_by": "seed"}


def summarize(records, baseline=None, draws=10000, bootstrap_seed=0, require_five_seeds=True,
              mlgym_reference="No-TTT"):
    indexed = index_records(records)
    grouped, units = defaultdict(dict), {}
    for key, record in indexed.items():
        backbone, benchmark, revision, method, seed, task = key
        if benchmark == "mlgym":
            continue
        for metric, specification in record["metrics"].items():
            group = (backbone, benchmark, revision, method, metric)
            if group in units and units[group] != specification["unit"]:
                raise ValueError("One metric retains consistent native units")
            units[group] = specification["unit"]
            grouped[group][(seed, task)] = native_value(specification)
    output = []
    for group, values in sorted(grouped.items()):
        backbone, benchmark, revision, method, metric = group
        seeds = sorted({seed for seed, task in values})
        if require_five_seeds and seeds != list(range(5)):
            raise ValueError("Paper summaries use seeds 0 through 4")
        task_sets = [{task for seed, task in values if seed == selected} for selected in seeds]
        if any(tasks != task_sets[0] for tasks in task_sets):
            raise ValueError("Each seed evaluates the same task set")
        means = [np.mean([value for (seed, task), value in values.items() if seed == selected]) for selected in seeds]
        row = {"backbone": backbone, "benchmark": benchmark, "revision": revision, "method": method, "metric": metric,
               "unit": "percentage_points" if units[group] == "rate" else units[group],
               "mean": float(np.mean(means)), "sample_std": float(np.std(means, ddof=1)) if len(means) > 1 else None,
               "seed_means": dict(zip(seeds, map(float, means))), "tasks_per_seed": len(task_sets[0])}
        if baseline is not None and method != baseline:
            reference = (backbone, benchmark, revision, baseline, metric)
            if reference not in grouped or units[reference] != units[group]:
                raise ValueError("The paired baseline is available in the same native units")
            row["paired_interval"] = paired_interval(values, grouped[reference], draws, bootstrap_seed)
        output.append(row)
    if any(key[1] == "mlgym" for key in indexed):
        output.extend(mlgym_wins(indexed, mlgym_reference, require_five_seeds))
    return output


def reported_metric_value(metric):
    native_value(metric)
    precision = metric.get("reported_precision")
    if isinstance(precision, bool) or not isinstance(precision, int) or not 0 <= precision <= 15:
        raise ValueError("MLGym comparisons declare reported_precision in the metric's original units")
    value = Decimal(str(metric["value"]))
    with localcontext() as context:
        context.prec = max(32, len(value.as_tuple().digits) + max(0, value.adjusted()) + precision + 2)
        return value.quantize(Decimal(1).scaleb(-precision))


def mlgym_wins(indexed, baseline, require_five_seeds=True):
    grouped = defaultdict(dict)
    metric_names = {}
    for key, record in indexed.items():
        backbone, benchmark, revision, method, seed, task = key
        if benchmark != "mlgym":
            continue
        if len(record["metrics"]) != 1:
            raise ValueError("An MLGym task selects its native comparison metric")
        metric_name, metric = next(iter(record["metrics"].items()))
        identity = (backbone, revision, seed, task)
        if identity in metric_names and metric_names[identity] != metric_name:
            raise ValueError("MLGym methods use the same native metric")
        metric_names[identity] = metric_name
        if metric.get("direction") not in {"higher", "lower"}:
            raise ValueError("An MLGym task declares its native metric direction")
        reported_metric_value(metric)
        grouped[(backbone, revision, method)][(seed, task)] = metric
    output = []
    for (backbone, revision, method), values in sorted(grouped.items()):
        if method == baseline:
            continue
        reference = grouped.get((backbone, revision, baseline), {})
        if set(values) != set(reference):
            raise ValueError("MLGym wins compare matching tasks and seeds")
        seeds = sorted({seed for seed, task in values})
        if require_five_seeds and seeds != list(range(5)):
            raise ValueError("MLGym summaries use five seeds")
        credits, scores = [], []
        task_sets = [{task for selected, task in values if selected == seed} for seed in seeds]
        if any(tasks != task_sets[0] for tasks in task_sets):
            raise ValueError("Every MLGym seed evaluates the same thirteen tasks")
        for seed in seeds:
            tasks = [task for selected, task in values if selected == seed]
            if len(tasks) != 13:
                raise ValueError("Win13 compares all thirteen MLGym tasks")
            credit = 0.0
            for task in tasks:
                first, second = values[(seed, task)], reference[(seed, task)]
                if first["unit"] != second["unit"] or first["direction"] != second["direction"]:
                    raise ValueError("MLGym methods use the same native metric")
                if first["reported_precision"] != second["reported_precision"]:
                    raise ValueError("MLGym methods use the same reported precision")
                difference = reported_metric_value(first) - reported_metric_value(second)
                if first["direction"] == "lower":
                    difference = -difference
                credit += 1.0 if difference > 0 else 0.5 if difference == 0 else 0.0
            credits.append(credit)
            scores.append(100 * credit / 13)
        output.append({"backbone": backbone, "benchmark": "mlgym", "revision": revision, "method": method,
                       "metric": "Win13", "unit": "percent", "mean": float(np.mean(scores)),
                       "sample_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
                       "seed_scores": dict(zip(seeds, scores)), "seed_credits": dict(zip(seeds, credits)),
                       "tasks_per_seed": 13, "reference_method": baseline})
    return output


def supported_citation_pairs(records):
    return len({(row["statement_id"], row["source_id"]) for row in records if row["supported"] is True})
