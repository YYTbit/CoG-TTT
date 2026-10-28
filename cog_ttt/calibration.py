import math

import numpy as np

from .confidence import Estimate, anytime_radius
from .graph import nonnegative_integer
from .measurements import legal_difference
from .storage import fingerprint
from .surrogate import SignedModel


def calibration_plan(graph, specification, features, budget):
    if not isinstance(specification, dict) or not {"graphs", "repetitions", "residual_bound"}.issubset(specification):
        raise ValueError("Development calibration declares a graph set, repetitions, and a residual bound")
    repetitions = nonnegative_integer(specification["repetitions"], "Calibration repetitions")
    residual = specification["residual_bound"]
    if repetitions < 1 or isinstance(residual, bool) or not math.isfinite(residual) or residual < 0:
        raise ValueError("Calibration uses positive repetitions and a finite nonnegative structural residual bound")
    graphs = [graph.decision(active) for active in specification["graphs"]]
    if not graphs or len({decision.identifier for decision in graphs}) != len(graphs):
        raise ValueError("The locked development graph set is nonempty and contains distinct legal configurations")
    for decision in graphs:
        graph.validate(decision, budget)
    model = SignedModel(graph, features=features)
    anchor = model.vector(graphs[0])
    design = np.array([model.vector(decision) - anchor for decision in graphs[1:]], dtype=float)
    if model.features and (len(graphs) < 2 or np.linalg.matrix_rank(design) != len(model.features)):
        raise ValueError("Development contrasts must identify the declared features; choose an independent feature set")
    record = {"graphs": [decision.bits for decision in graphs],
              "features": [list(feature.support) for feature in model.features],
              "repetitions": repetitions, "residual_bound": float(residual)}
    return {**record, "decisions": graphs, "identity": fingerprint(record)}


def fit_calibration(graph, plan, rows, delta, scope):
    if len(rows) != len(plan["decisions"]) or any(first != second[0]
            for first, second in zip(plan["decisions"], rows)):
        raise ValueError("Calibration observations follow the locked development graph order")
    observations = []
    for decision, estimate in rows:
        graph.validate(decision)
        if len(estimate.scores) < plan["repetitions"]:
            raise ValueError("Every development configuration has the declared number of observations")
        selected = Estimate(estimate.identifier)
        for score in estimate.scores[:plan["repetitions"]]:
            selected.add(score)
        observations.append(selected)
    model = SignedModel(graph, features=plan["features"])
    anchor = model.vector(rows[0][0])
    design = np.array([model.vector(decision) - anchor for decision, _ in rows[1:]], dtype=float)
    contrasts = [legal_difference(graph, decision, rows[0][0], estimate, observations[0],
                                 max(graph.cost(decision), graph.cost(rows[0][0])), delta, plan["residual_bound"])
                 for (decision, _), estimate in zip(rows[1:], observations[1:])]
    if model.features:
        if np.linalg.matrix_rank(design) != len(model.features):
            raise ValueError("Development contrasts do not identify the declared signed coefficients")
        inverse = np.linalg.pinv(design)
        coefficients = inverse @ np.array([row["difference"] for row in contrasts])
        # Combine shared-anchor coefficients before bounding mean errors and residuals.
        weights = np.column_stack((-inverse.sum(axis=1), inverse))
        uncertainty = np.array([anytime_radius(delta, estimate.identifier, len(estimate.scores))
                                + plan["residual_bound"] for estimate in observations])
        radii = np.abs(weights) @ uncertainty
    else:
        coefficients, radii = np.zeros(0), np.zeros(0)
    bias = float(observations[0].mean - anchor @ coefficients)
    bias_radius = float(anytime_radius(delta, observations[0].identifier, plan["repetitions"])
                        + plan["residual_bound"] + anchor @ radii)
    if not np.isfinite(np.r_[bias, bias_radius, coefficients, radii]).all():
        raise ValueError("Calibration estimates and interval radii are finite")
    return {"schema": 1, "plan_identity": plan["identity"], "scope": dict(scope), "delta": delta,
            "graph_identity": fingerprint(graph.record()), "features": plan["features"],
            "residual_bound": plan["residual_bound"], "bias": bias, "bias_radius": bias_radius,
            "coefficients": coefficients.tolist(), "radii": radii.tolist(),
            "coefficient_upper": (coefficients + radii).tolist(),
            "observations": [{"graph": list(decision.bits), "identifier": estimate.identifier,
                              "count": len(estimate.scores), "mean": estimate.mean}
                             for (decision, _), estimate in zip(rows, observations)]}


def validate_calibration(record, graph, plan, delta, scope):
    if (record.get("schema") != 1 or record.get("plan_identity") != plan["identity"]
            or record.get("scope") != scope or record.get("delta") != delta
            or record.get("graph_identity") != fingerprint(graph.record())
            or record.get("features") != plan["features"]
            or record.get("residual_bound") != plan["residual_bound"]):
        raise ValueError("Calibration belongs to the declared development plan, graph, score source, and parameter version")
    size = len(plan["features"])
    arrays = [np.asarray(record[name], dtype=float) for name in ("coefficients", "radii", "coefficient_upper")]
    if (any(value.shape != (size,) or not np.isfinite(value).all() for value in arrays)
            or np.any(arrays[1] < 0) or not np.array_equal(arrays[0] + arrays[1], arrays[2])
            or not np.isfinite([record["bias"], record["bias_radius"]]).all() or record["bias_radius"] < 0):
        raise ValueError("Calibration stores matching finite coefficient estimates and nonnegative uncertainty radii")


def apply_calibration(model, record):
    if record["features"] != [list(feature.support) for feature in model.features]:
        raise ValueError("Calibrated and fitted signed models use the same declared feature set")
    # Recenter the development interval around the current fit without shrinking its coverage.
    model.radii = np.asarray(record["radii"]) + np.abs(model.coefficients - np.asarray(record["coefficients"]))
    model.calibration = record
    return model
