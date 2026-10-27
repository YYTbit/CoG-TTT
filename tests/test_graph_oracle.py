import itertools

import numpy as np
import pytest

from cog_ttt.graph import Decision, Graph
from cog_ttt.oracle import allocate_profiles, graph_upper_bound
from cog_ttt.surrogate import SignedModel


def test_required_ports_and_exclusive_routes(routing_case):
    graph = Graph.from_record(routing_case["graph"])
    graph.validate(graph.decision(routing_case["initial_graph"]), 3)
    for active in (["source", "terminal"], ["source", "terminal", "revised"],
                   ["source", "alternative", "terminal", "direct", "revised"]):
        with pytest.raises(ValueError):
            graph.validate(graph.decision(active), 3)


def test_cycles_and_shared_owner_descendants():
    graph = Graph(nodes=[{"name": name, "cost": 1, "operation": "act", "owner": owner}
                         for name, owner in [("a", "first"), ("b", "second"), ("c", "first")]],
                  routes=[{"name": "route", "source": "b", "target": "c"}], dependencies=[("a", "b")])
    assert graph.affected("first") == {"a", "b", "c", "route"}
    assert graph.affected("second") == {"b", "c", "route"}
    assert graph.affected("second", True) == set(graph.names)
    cyclic = Graph(nodes=[{"name": name, "cost": 0, "operation": "act", "kind": "tool"} for name in ["a", "b"]],
                   dependencies=[("a", "b"), ("b", "a")])
    with pytest.raises(ValueError, match="cycle"):
        cyclic.validate(cyclic.decision(["a", "b"]), 0)


def test_allocation_matches_exhaustive_integer_search():
    profiles = [np.array([0.0, 0.3, 0.3, 0.4, 0.4]), np.array([0, 0.2, 0.8, 0.8, 0.8]),
                np.array([0.0, 0.15, 0.4, 0.45, 0.5])]
    result = allocate_profiles(profiles, 4)
    expected = max(sum(profile[value] for profile, value in zip(profiles, allocation))
                   for allocation in itertools.product(range(5), repeat=3) if sum(allocation) <= 4)
    assert result["value"] == pytest.approx(expected)
    assert sum(result["allocation"]) <= 4


@pytest.mark.parametrize("seed", range(8))
def test_signed_partition_bound_dominates_every_feasible_graph(seed, routing_case):
    graph = Graph.from_record(routing_case["graph"])
    model = SignedModel(graph)
    generator = np.random.default_rng(seed)
    model.bias = generator.normal()
    model.coefficients = generator.normal(size=len(model.features))
    feasible = graph.enumerate(3)
    certificate = {"allowance": 20.0}
    result = graph_upper_bound(graph, model, certificate, 3, training_cost=0.1,
                               training_price=0.2, execution_price=0.03)
    exact_surrogate = max(model.predict(decision) + 20 - 0.02 - 0.03 * graph.cost(decision) for decision in feasible)
    assert result["dp_value"] >= exact_surrogate - 1e-9
    exact_score_upper = max(min(1.0, model.predict(decision) + 20) - 0.02 - 0.03 * graph.cost(decision) for decision in feasible)
    assert result["upper"] >= exact_score_upper - 1e-9


def test_relaxed_assignment_is_not_an_executable_witness():
    graph = Graph(nodes=[{"name": name, "cost": 1, "operation": "act", "kind": "tool"} for name in ["a", "b"]],
                  dependencies=[("a", "b")])
    model = SignedModel(graph, features=[("a",), ("b",)], coefficients=[-2, 1])
    result = graph_upper_bound(graph, model, {"allowance": 0.0}, 1, execution_price=0,
                               partitions=[("a",), ("b",)])
    assert result["relaxed_witness"] == (0, 1)
    assert not result["original_feasible"]
    with pytest.raises(ValueError):
        graph.validate(Decision(result["relaxed_witness"]), 1)


def test_new_witness_split_is_independent_of_discovery_order(routing_case, settings):
    from cog_ttt.confidence import ConfidenceRegistry
    from cog_ttt.configuration import validate_case
    from cog_ttt.search import JointSearch
    graph, initial, supplied, configured, budget = validate_case(routing_case, settings)
    first = JointSearch(graph, ConfidenceRegistry(), "state", 0, configured, [initial])
    second = JointSearch(graph, ConfidenceRegistry(), "state", 0, configured, [initial])
    additional = [decision for decision in graph.enumerate(3) if decision != initial]
    for decision in additional:
        first.register_witness(decision)
    for decision in reversed(additional):
        second.register_witness(decision)
    assert first.fit_roles == second.fit_roles
