import copy
import math
from pathlib import Path

import yaml

from .graph import Graph, nonnegative_integer
from .storage import read_json


def read_configuration(path):
    path = Path(path)
    value = read_json(path) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Configurations are mappings")
    return value


def merge(first, second):
    result = copy.deepcopy(first)
    for key, value in second.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def validate_case(case, configuration):
    graph = Graph.from_record(case["graph"])
    owners = {node.owner for node in graph.nodes if node.kind == "model"}
    if owners != set(case.get("owners", {})):
        raise ValueError("The model specifications match the graph's parameter owners")
    objective = merge(configuration["objective"], case["objective"])
    for name in ("total_budget", "execution_budget"):
        nonnegative_integer(objective[name], name)
    for name in ("cost_quantum_gpu_seconds", "utility_per_cost_unit"):
        if not math.isfinite(objective[name]) or objective[name] <= 0:
            raise ValueError("Cost discretization and utility conversion are explicit positive inputs")
    for name in ("lambda_training", "lambda_execution"):
        if not math.isfinite(objective[name]) or objective[name] < 0:
            raise ValueError("Objective cost prices are finite and nonnegative")
    budget = merge(configuration["budget"], case["budget"])
    for name in ("total_gpu_seconds", "reserve_gpu_seconds", "pilot_gpu_seconds", "verification_gpu_seconds"):
        if not math.isfinite(budget[name]) or budget[name] < 0:
            raise ValueError("Operational bounds are finite and nonnegative")
    if budget["reserve_gpu_seconds"] > budget["total_gpu_seconds"]:
        raise ValueError("The execution reserve fits the operational budget")
    initial = graph.decision(case["initial_graph"])
    graph.validate(initial, min(objective["execution_budget"], objective["total_budget"]))
    supplied = [initial] + [graph.decision(active) for active in case.get("witnesses", [])]
    for decision in supplied:
        graph.validate(decision, min(objective["execution_budget"], objective["total_budget"]))
        if graph.cost(decision) * objective["cost_quantum_gpu_seconds"] > budget["reserve_gpu_seconds"]:
            raise ValueError("The declared execution reserve covers every supplied continuation witness")
    identifiers = set()
    for program in case.get("programs", []):
        if not program["id"] or program["id"] == "null" or program["id"] in identifiers or program["owner"] not in owners:
            raise ValueError("Programs have unique identifiers and declared owners")
        identifiers.add(program["id"])
        decision = graph.decision(program["graph"])
        graph.validate(decision, objective["execution_budget"])
        if not any(node.owner == program["owner"] and decision.bits[graph.index[node.name]] for node in graph.nodes):
            raise ValueError("The training graph activates its parameter owner's operation")
        if nonnegative_integer(program["steps"], "Gradient steps") < 1 or not math.isfinite(program["learning_rate"]) or program["learning_rate"] <= 0:
            raise ValueError("Programs explicitly configure positive step counts and learning rates")
    settings = merge(merge(configuration["search"], case.get("search", {})), objective)
    for name in ("lambda_training", "lambda_execution"):
        settings[name] *= objective["utility_per_cost_unit"]
    for name in ("max_probes", "max_pilots", "max_graph_bits", "max_local_bits", "split_seed"):
        nonnegative_integer(settings[name], name)
    for name in ("epsilon", "acceptance_tolerance", "screen_margin", "refinement_tolerance"):
        if not math.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError("Search tolerances are finite and nonnegative")
    if not 0 < configuration["confidence"]["delta"] < 1 or not 0 <= settings["cross_partition_split"] <= 1:
        raise ValueError("Confidence and partition-split settings are within their declared domains")
    return graph, initial, supplied, settings, budget
