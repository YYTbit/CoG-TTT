import math

import numpy as np

from .confidence import Estimate
from .graph import Decision
from .oracle import graph_upper_bound
from .storage import fingerprint
from .surrogate import SignedModel, residual_certificate, score_interval


class JointSearch:
    def __init__(self, graph, registry, task_state, version, settings, witnesses, complete_graphs=None):
        self.graph, self.registry, self.task_state, self.version = graph, registry, task_state, version
        self.settings = settings
        self.witnesses = {decision.identifier: decision for decision in witnesses}
        self.complete_graphs = complete_graphs
        self.partitions = graph.partitions()
        self.refinements = 0
        self.operational_execution_budget = None
        identifiers = sorted(self.witnesses)
        generator = np.random.default_rng(self.settings["split_seed"])
        generator.shuffle(identifiers)
        cutoff = max(1, min(len(identifiers) - 1, int(0.7 * len(identifiers)))) if len(identifiers) > 1 else len(identifiers)
        self.fit_roles = {identifier: index < cutoff for index, identifier in enumerate(identifiers)}

    def estimate(self, program, decision):
        return self.registry.lookup(self.task_state, self.version, program, decision.identifier) or Estimate(1)

    def graph_budget(self, training_cost):
        values = [self.settings["execution_budget"], self.settings["total_budget"] - training_cost]
        if self.operational_execution_budget is not None:
            values.append(self.operational_execution_budget)
        return math.floor(min(values) + 1e-12)

    def register_witness(self, decision):
        if decision.identifier not in self.fit_roles:
            self.fit_roles[decision.identifier] = self.additional_fit_role(decision.identifier)
        self.witnesses[decision.identifier] = decision

    def additional_fit_role(self, identifier):
        return int(fingerprint([self.settings["split_seed"], identifier]), 16) % 10 < 7

    def model(self, program, training_cost):
        observed = [(decision, self.estimate(program, decision)) for decision in self.witnesses.values()
                    if self.graph.cost(decision) <= self.graph_budget(training_cost)
                    and self.estimate(program, decision).scores]
        fit, validation = [], []
        for decision, estimate in observed:
            if decision.identifier not in self.fit_roles:
                self.fit_roles[decision.identifier] = self.additional_fit_role(decision.identifier)
            (fit if self.fit_roles[decision.identifier] else validation).append((decision, estimate))
        model = SignedModel(self.graph, features=self.settings.get("features"))
        if fit:
            model.fit(fit, self.registry.delta, self.settings.get("coefficient_residual_bound"))
        feasible = None if self.complete_graphs is None else [decision for decision in self.complete_graphs
                    if self.graph.cost(decision) <= self.graph_budget(training_cost)]
        certificate = residual_certificate(model, validation, self.registry.delta, feasible) if fit else None
        return model, certificate

    def bounds(self, costs):
        pairs, programs = [], {}
        for program, training_cost in costs.items():
            budget = self.graph_budget(training_cost)
            feasible = [decision for decision in self.witnesses.values() if self.graph.cost(decision) <= budget]
            if not feasible:
                continue
            model, certificate = self.model(program, training_cost)
            oracle = graph_upper_bound(self.graph, model, certificate, budget, training_cost,
                                       self.settings["lambda_training"], self.settings["lambda_execution"],
                                       self.partitions, self.settings["cross_partition_split"],
                                       self.settings["max_local_bits"])
            if oracle["original_feasible"]:
                decision = Decision(tuple(oracle["relaxed_witness"]))
                self.register_witness(decision)
                if decision not in feasible:
                    feasible.append(decision)
            current = []
            for decision in feasible:
                estimate = self.estimate(program, decision)
                lower, upper = score_interval(model, decision, estimate, self.registry.delta, certificate)
                penalty = self.settings["lambda_training"] * training_cost + self.settings["lambda_execution"] * self.graph.cost(decision)
                current.append({"program": program, "graph": decision, "lower": lower - penalty,
                                "upper": upper - penalty, "score_lower": lower, "score_upper": upper,
                                "direct_upper": estimate.interval(self.registry.delta)[1] - penalty,
                                "observations": len(estimate.scores), "training_cost": training_cost})
            if self.complete_graphs is not None:
                # Every feasible graph is present, so its largest interval is also a valid oracle.
                oracle["upper"] = min(oracle["upper"], max(pair["upper"] for pair in current))
                oracle["enumerated_intersection"] = True
            if oracle["upper"] + 1e-9 < max(pair["upper"] for pair in current):
                # A graph oracle may be tighter than a direct interval; intersect that interval too.
                for pair in current:
                    pair["upper"] = min(pair["upper"], oracle["upper"])
                    if pair["upper"] + 1e-9 < pair["lower"]:
                        raise ValueError("The graph oracle contradicts a feasible lower bound")
            optimistic = max(current, key=lambda pair: (pair["upper"], -self.graph.cost(pair["graph"])))
            relaxation = None
            if oracle["dp_value"] is not None:
                def optimistic_surrogate(pair):
                    return (model.predict(pair["graph"]) + certificate["allowance"]
                            - self.settings["lambda_training"] * training_cost
                            - self.settings["lambda_execution"] * self.graph.cost(pair["graph"]))
                witness = max(current, key=optimistic_surrogate)
                witness_value = optimistic_surrogate(witness)
                if oracle["dp_value"] + 1e-9 < witness_value:
                    raise ValueError("A partition bound lies below its feasible optimistic witness")
                relaxation = {"upper": max(0.0, oracle["dp_value"] - witness_value),
                              "witness": witness["graph"].bits, "witness_value": witness_value,
                              "objective": "signed_surrogate_plus_certified_residual_minus_cost"}
            programs[program] = {"oracle": oracle, "certificate": certificate, "model": model,
                                 "optimistic": optimistic,
                                 "relaxation_gap": relaxation,
                                 "slack": max(0.0, oracle["upper"] - optimistic["upper"])}
            pairs.extend(current)
        if not pairs:
            raise ValueError("Materialized programs have no feasible executable witness")
        incumbent = max(pairs, key=lambda pair: (pair["lower"], pair["program"] == "null",
                                                -self.graph.cost(pair["graph"])))
        gap = max(0.0, max(item["oracle"]["upper"] for item in programs.values()) - incumbent["lower"])
        evaluated = [pair for pair in pairs if pair["observations"]]
        frontier = max(pair["direct_upper"] for pair in evaluated) - incumbent["lower"] if evaluated else None
        screened = [program for program, item in programs.items()
                    if item["oracle"]["upper"] <= incumbent["lower"] + self.settings["screen_margin"]]
        return {"incumbent": incumbent, "gap": gap, "frontier_gap": None if frontier is None else max(0.0, frontier), "pairs": pairs,
                "programs": programs, "screened": screened}

    def refine(self, result):
        if len(self.partitions) < 2 or max(item["slack"] for item in result["programs"].values()) <= self.settings["refinement_tolerance"]:
            return False
        model = max(result["programs"].values(), key=lambda item: item["slack"])["model"]
        membership = {name: index for index, group in enumerate(self.partitions) for name in group}
        gains = {}
        for feature, value in zip(model.features, model.coefficients):
            if len(feature.support) == 2:
                first, second = sorted(membership[name] for name in feature.support)
                if first != second:
                    gains[(first, second)] = gains.get((first, second), 0.0) + abs(value)
        if not gains:
            return False
        first, second = max(gains, key=gains.get)
        if len(self.partitions[first]) + len(self.partitions[second]) > self.settings["max_local_bits"]:
            return False
        self.partitions[first] = self.partitions[first] + self.partitions[second]
        self.partitions.pop(second)
        self.refinements += 1
        return True

    def next_probe(self, result):
        incumbent = result["incumbent"]
        eligible = [pair for pair in result["pairs"] if pair["program"] not in result["screened"]]
        if not eligible:
            return None
        def priority(pair):
            model = result["programs"][pair["program"]]["model"]
            width = model.witness_width(pair["graph"], incumbent["graph"])
            return (pair["observations"] == 0, pair["upper"], width, -pair["observations"])
        return max(eligible, key=priority)

    def report(self, result):
        incumbent = result["incumbent"]
        return {"parameter_version": self.version, "program": incumbent["program"],
                "graph": incumbent["graph"].bits, "objective_lower": incumbent["lower"],
                "objective_upper": incumbent["upper"], "decision_gap": result["gap"],
                "evaluated_frontier_gap": result["frontier_gap"], "screened_programs": result["screened"],
                "complete_graph_enumeration": self.complete_graphs is not None,
                "remaining_execution_budget_units": self.operational_execution_budget,
                "oracle_refinements": self.refinements,
                "programs": {name: {"oracle": item["oracle"], "certificate": item["certificate"],
                                    "oracle_slack": item["slack"], "relaxation_gap": item["relaxation_gap"]}
                             for name, item in result["programs"].items()}}
