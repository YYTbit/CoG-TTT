import itertools
from dataclasses import dataclass

import numpy as np

from .graph import Decision, nonnegative_integer


@dataclass
class LocalProfile:
    names: tuple
    values: np.ndarray
    assignments: list
    method: str


def allocate_profiles(profiles, budget):
    budget = nonnegative_integer(budget, "Partition budget")
    arrays = [np.asarray(profile, dtype=float) for profile in profiles]
    if not arrays or any(array.shape != (budget + 1,) or np.any(np.isnan(array))
                         or np.any(np.isposinf(array)) for array in arrays):
        raise ValueError("Local profiles provide one finite or infeasible value per integer budget")
    values = np.zeros(budget + 1)
    choices = []
    for profile in arrays:
        updated = np.full(budget + 1, -np.inf)
        selected = np.zeros(budget + 1, dtype=int)
        for available in range(budget + 1):
            alternatives = profile[:available + 1] + values[available::-1]
            chosen = int(np.argmax(alternatives))
            updated[available], selected[available] = alternatives[chosen], chosen
        choices.append(selected)
        values = updated
    available, allocation = budget, []
    for selected in reversed(choices):
        allocated = int(selected[available])
        allocation.append(allocated)
        available -= allocated
    return {"value": float(values[budget]), "allocation": list(reversed(allocation)), "unused_budget": available}


def partition_profiles(graph, model, partitions, budget, execution_price=1.0, cross_split=0.5, max_local_bits=18):
    budget = nonnegative_integer(budget, "Graph budget")
    if not np.isfinite(execution_price) or execution_price < 0 or not 0 <= cross_split <= 1:
        raise ValueError("Execution prices are nonnegative and cross-partition splits lie in [0,1]")
    partitions = [tuple(names) for names in partitions]
    flattened = [name for names in partitions for name in names]
    if len(set(flattened)) != len(flattened) or set(flattened) != set(graph.names):
        raise ValueError("Partitions cover each node and routing bit exactly once")
    membership = {name: index for index, names in enumerate(partitions) for name in names}
    linear = [dict.fromkeys(names, 0.0) for names in partitions]
    internal = [[] for _ in partitions]
    for index, (feature, coefficient) in enumerate(zip(model.features, model.coefficients)):
        first = feature.support[0]
        group = membership[first]
        if len(feature.support) == 1:
            linear[group][first] += coefficient
        elif membership[feature.support[1]] == group:
            internal[group].append((feature.support, coefficient))
        else:
            if model.calibration is not None:
                coefficient += model.radii[index]
                if not np.isfinite(coefficient):
                    raise ValueError("Cross-partition calibration uses finite coefficient upper bounds")
            if coefficient > 0:
                second = feature.support[1]
                linear[group][first] += coefficient * cross_split
                linear[membership[second]][second] += coefficient * (1 - cross_split)
    profiles = []
    mandatory = {node.name for node in graph.nodes if node.required} | set(graph.terminals)
    for group, names in enumerate(partitions):
        costs = {name: graph.costs[graph.index[name]] for name in names}
        local_clauses = [(coefficients, lower, upper) for coefficients, lower, upper in graph.clauses()
                         if set(coefficients).issubset(names)]
        values = np.full(budget + 1, -np.inf)
        assignments = [None] * (budget + 1)
        if len(names) > max_local_bits:
            minimum = sum(costs[name] for name in names if name in mandatory)
            bound = (sum(max(value, 0.0) for value in linear[group].values())
                     + sum(max(value, 0.0) for _, value in internal[group]) - execution_price * minimum)
            values[minimum:] = bound
            profiles.append(LocalProfile(names, values, assignments, "range_relaxation"))
            continue
        for bits in itertools.product((0, 1), repeat=len(names)):
            active = dict(zip(names, bits))
            cost = sum(costs[name] * active[name] for name in names)
            if cost > budget:
                continue
            legal = True
            for coefficients, lower, upper in local_clauses:
                total = sum(active[name] * value for name, value in coefficients.items())
                if (lower is not None and total < lower - 1e-12) or (upper is not None and total > upper + 1e-12):
                    legal = False
                    break
            if not legal:
                continue
            value = sum(active[name] * coefficient for name, coefficient in linear[group].items())
            value += sum(coefficient * active[first] * active[second] for (first, second), coefficient in internal[group])
            value -= execution_price * cost
            if value > values[cost]:
                values[cost], assignments[cost] = value, bits
        for available in range(1, budget + 1):
            if values[available - 1] > values[available]:
                values[available], assignments[available] = values[available - 1], assignments[available - 1]
        profiles.append(LocalProfile(names, values, assignments, "local_enumeration"))
    return profiles


def graph_upper_bound(graph, model, certificate, budget, training_cost=0.0, training_price=1.0,
                       execution_price=1.0, partitions=None, cross_split=0.5, max_local_bits=18):
    if not np.isfinite([training_cost, training_price, execution_price]).all() or min(training_cost, training_price, execution_price) < 0:
        raise ValueError("Candidate costs and objective prices are finite and nonnegative")
    required_cost = sum(node.cost for node in graph.nodes if node.required or node.name in graph.terminals)
    fallback = 1 - training_price * training_cost - execution_price * required_cost
    result = {"upper": float(fallback), "method": "bounded_score_range", "dp_value": None,
              "relaxed_witness": None, "original_feasible": False}
    if certificate is None:
        return result
    profiles = partition_profiles(graph, model, partitions or graph.partitions(), budget,
                                    execution_price, cross_split, max_local_bits)
    allocation = allocate_profiles([profile.values for profile in profiles], budget)
    upper = model.bias + certificate["allowance"] - training_price * training_cost + allocation["value"]
    if not np.isfinite(upper):
        raise ValueError("The relaxed family is infeasible at the supplied integer budget")
    result.update(upper=float(min(fallback, upper)), method="certified_partition_dp",
                  dp_value=float(upper), allocation=allocation["allocation"],
                  cross_partition_coefficients=("calibrated_upper_bounds" if model.calibration else "fitted_values"),
                  profiles=[{"names": profile.names, "method": profile.method} for profile in profiles])
    active = {}
    for profile, available in zip(profiles, allocation["allocation"]):
        assignment = profile.assignments[available]
        if assignment is None:
            return result
        active.update(zip(profile.names, assignment))
    candidate = Decision(tuple(active[name] for name in graph.names))
    result["relaxed_witness"] = candidate.bits
    try:
        graph.validate(candidate, budget)
        result["original_feasible"] = True
    except ValueError:
        pass
    return result
