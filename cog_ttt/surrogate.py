import itertools
from dataclasses import dataclass

import numpy as np

from .confidence import anytime_radius


@dataclass(frozen=True)
class Feature:
    support: tuple

    def __post_init__(self):
        if not 1 <= len(self.support) <= 2 or len(set(self.support)) != len(self.support):
            raise ValueError("Signed graph features are singleton or pairwise activations")

    def value(self, decision, graph):
        return int(all(decision.bits[graph.index[name]] for name in self.support))


class SignedModel:
    def __init__(self, graph, features=None, bias=0.0, coefficients=None, radii=None):
        self.graph = graph
        mandatory = {node.name for node in graph.nodes if node.required} | set(graph.terminals)
        optional = [name for name in graph.names if name not in mandatory]
        supports = ([(name,) for name in optional] + list(itertools.combinations(optional, 2))
                    if features is None else features)
        self.features = tuple(Feature(tuple(support)) for support in supports)
        if any(not set(feature.support).issubset(graph.index) for feature in self.features):
            raise ValueError("A signed feature references an unknown graph bit")
        self.bias = float(bias)
        self.coefficients = np.zeros(len(self.features)) if coefficients is None else np.asarray(coefficients, dtype=float)
        self.radii = np.full(len(self.features), np.inf) if radii is None else np.asarray(radii, dtype=float)
        self.calibration = None
        if self.coefficients.shape != (len(self.features),) or self.radii.shape != self.coefficients.shape:
            raise ValueError("Signed coefficient dimensions differ from their feature set")
        if not np.isfinite(self.bias) or not np.isfinite(self.coefficients).all() or np.any(np.isnan(self.radii)) or np.any(self.radii < 0):
            raise ValueError("Signed estimates are finite and uncertainty radii are nonnegative")

    def vector(self, decision):
        return np.array([feature.value(decision, self.graph) for feature in self.features], dtype=float)

    def predict(self, decision):
        return float(self.bias + self.vector(decision) @ self.coefficients)

    def fit(self, rows, delta, residual_bound=None):
        if not rows:
            raise ValueError("Surrogate fitting requires legally measured graph records")
        matrix = np.array([np.r_[1.0, self.vector(decision)] for decision, estimate in rows])
        values = np.array([estimate.mean for _, estimate in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Surrogate fitting uses observed continuation scores")
        parameters = np.linalg.lstsq(matrix, values, rcond=None)[0]
        self.bias, self.coefficients = float(parameters[0]), parameters[1:]
        self.radii = np.full(len(self.features), np.inf)
        if residual_bound is not None:
            if not np.isfinite(residual_bound) or residual_bound < 0:
                raise ValueError("The coefficient residual bound is finite and nonnegative")
            if np.linalg.matrix_rank(matrix) == matrix.shape[1]:
                noise = np.array([anytime_radius(delta, estimate.identifier, len(estimate.scores)) + residual_bound
                                  for _, estimate in rows])
                self.radii = (np.abs(np.linalg.pinv(matrix)) @ noise)[1:]
        return self

    @property
    def lipschitz(self):
        return max((sum(abs(value) for feature, value in zip(self.features, self.coefficients)
                        if name in feature.support) for name in self.graph.names), default=0.0)

    def witness_width(self, first, second):
        difference = np.abs(self.vector(first) - self.vector(second))
        active = difference > 0
        return float(self.radii[active] @ difference[active]) if np.any(active) else 0.0

    def affected_features(self, owner, global_state=False):
        affected = self.graph.affected(owner, global_state)
        return [index for index, feature in enumerate(self.features) if set(feature.support) & affected]


def residual_certificate(model, validation, delta, feasible=None):
    if not validation:
        return None
    if feasible is None:
        covering = len(model.graph.names)
        covering_method = "graph_bit_count_upper_bound"
    else:
        covering = max((min(sum(a != b for a, b in zip(decision.bits, observed.bits))
                            for observed, _ in validation) for decision in feasible), default=0)
        covering_method = "complete_feasible_graph_enumeration"
    errors = [abs(estimate.mean - model.predict(decision)) for decision, estimate in validation]
    radii = [anytime_radius(delta, estimate.identifier, len(estimate.scores)) for _, estimate in validation]
    allowance = max(error + radius for error, radius in zip(errors, radii)) + (1 + model.lipschitz) * covering
    return {"allowance": float(allowance), "covering_radius": covering, "covering_method": covering_method,
            "score_lipschitz": 1.0, "surrogate_lipschitz": model.lipschitz,
            "residual_percentile_95": float(np.quantile(errors, 0.95)), "validation_graphs": len(validation)}


def score_interval(model, decision, estimate, delta, certificate=None):
    lower, upper = estimate.interval(delta)
    if certificate is not None:
        value, radius = model.predict(decision), certificate["allowance"]
        lower, upper = max(lower, 0, value - radius), min(upper, 1, value + radius)
    if lower > upper + 1e-12:
        raise ValueError("Direct and model bounds are inconsistent at this version")
    return float(lower), float(upper)
