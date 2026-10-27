import itertools
from dataclasses import dataclass, field

import numpy as np

from .storage import fingerprint


def nonnegative_integer(value, name):
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(name + " requires a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class Node:
    name: str
    cost: int
    operation: str
    owner: str | None = None
    kind: str = "model"
    required: bool = False
    partition: str | None = None
    settings: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Route:
    name: str
    source: str
    target: str
    cost: int = 0
    port: str = "input"


@dataclass(frozen=True)
class Decision:
    bits: tuple

    def __post_init__(self):
        if any(value not in (0, 1) for value in self.bits):
            raise ValueError("Graph decisions are binary")
        object.__setattr__(self, "bits", tuple(int(value) for value in self.bits))

    @property
    def identifier(self):
        return fingerprint(self.bits)


class Graph:
    def __init__(self, nodes, routes=(), dependencies=(), ports=(), exclusive=(), terminals=(), constraints=()):
        self.nodes = tuple(Node(**row) if isinstance(row, dict) else row for row in nodes)
        self.routes = tuple(Route(**row) if isinstance(row, dict) else row for row in routes)
        self.names = tuple(node.name for node in self.nodes) + tuple(route.name for route in self.routes)
        if (not self.nodes or any(not isinstance(name, str) or not name for name in self.names)
                or len(set(self.names)) != len(self.names)):
            raise ValueError("Operations and routes require distinct identifiers")
        self.index = {name: index for index, name in enumerate(self.names)}
        self.node_map = {node.name: node for node in self.nodes}
        self.dependencies = tuple(tuple(pair) for pair in dependencies)
        self.ports = tuple(ports)
        self.exclusive = tuple(tuple(group) for group in exclusive)
        self.terminals = tuple(terminals)
        self.constraints = tuple(constraints)
        self.costs = tuple(nonnegative_integer(item.cost, "Graph execution cost") for item in self.nodes + self.routes)
        for node in self.nodes:
            if node.kind not in {"model", "tool"} or (node.kind == "model" and not node.owner):
                raise ValueError("Model operations declare their parameter owner")
            if not isinstance(node.operation, str) or not node.operation or not isinstance(node.settings, dict) or not isinstance(node.required, bool):
                raise ValueError("Operations declare their name, settings mapping, and mandatory flag")
        for route in self.routes:
            if route.source not in self.node_map or route.target not in self.node_map:
                raise ValueError("A route references an unknown operation")
        if any(len(pair) != 2 or any(name not in self.node_map for name in pair) for pair in self.dependencies):
            raise ValueError("Dependencies reference two known operations")
        if any(name not in self.node_map for name in self.terminals):
            raise ValueError("Terminal operations are declared vertices")
        routes_by_name = {route.name: route for route in self.routes}
        for port in self.ports:
            if port["node"] not in self.node_map or not port["routes"]:
                raise ValueError("Required input ports declare their operation and producers")
            for name in port["routes"]:
                route = routes_by_name.get(name)
                if route is None or route.target != port["node"] or route.port != port["port"]:
                    raise ValueError("Input-port producers use authorized incoming routes")
        if any(name not in routes_by_name for group in self.exclusive for name in group):
            raise ValueError("Exclusive groups contain declared routes")
        for constraint in self.constraints:
            if not set(constraint["coefficients"]).issubset(self.index):
                raise ValueError("A linear constraint references an unknown graph bit")
            if not np.isfinite(list(constraint["coefficients"].values())).all():
                raise ValueError("Graph constraint coefficients are finite")
            bounds = [constraint.get("lower"), constraint.get("upper")]
            if any(value is not None and not np.isfinite(value) for value in bounds):
                raise ValueError("Declared linear-constraint bounds are finite")
            if all(value is not None for value in bounds) and bounds[0] > bounds[1]:
                raise ValueError("A linear constraint has an ordered lower and upper bound")

    @classmethod
    def from_record(cls, value):
        return cls(**value)

    def record(self):
        return {"nodes": [node.__dict__ for node in self.nodes], "routes": [route.__dict__ for route in self.routes],
                "dependencies": self.dependencies, "ports": self.ports, "exclusive": self.exclusive,
                "terminals": self.terminals, "constraints": self.constraints}

    def decision(self, active):
        if not set(active).issubset(self.index):
            raise ValueError("A graph decision references unknown bits")
        return Decision(tuple(int(name in active) for name in self.names))

    def cost(self, decision):
        if len(decision.bits) != len(self.names):
            raise ValueError("Graph decision dimensions differ from its specification")
        return sum(cost * bit for cost, bit in zip(self.costs, decision.bits))

    def clauses(self):
        result = []
        for node in self.nodes:
            if node.required or node.name in self.terminals:
                result.append(({node.name: 1}, 1, 1))
        for source, target in self.dependencies:
            result.append(({target: 1, source: -1}, None, 0))
        for route in self.routes:
            result.append(({route.name: 1, route.source: -1}, None, 0))
            result.append(({route.name: 1, route.target: -1}, None, 0))
        for port in self.ports:
            result.append(({**dict.fromkeys(port["routes"], 1), port["node"]: -1}, 0, None))
        for group in self.exclusive:
            result.append((dict.fromkeys(group, 1), None, 1))
        for value in self.constraints:
            result.append((value["coefficients"], value.get("lower"), value.get("upper")))
        return result

    def validate(self, decision, budget=None):
        cost = self.cost(decision)
        if budget is not None and (not np.isfinite(budget) or budget < 0):
            raise ValueError("The graph execution budget is finite and nonnegative")
        if budget is not None and cost > budget:
            raise ValueError("Graph execution exceeds its available budget")
        active = dict(zip(self.names, decision.bits))
        for coefficients, lower, upper in self.clauses():
            value = sum(active[name] * coefficient for name, coefficient in coefficients.items())
            if (lower is not None and value < lower - 1e-12) or (upper is not None and value > upper + 1e-12):
                raise ValueError("Graph violates a prerequisite, input-port, route, or endpoint constraint")
        self.execution_order(decision)
        return True

    def execution_order(self, decision):
        active = {name for name, bit in zip(self.names, decision.bits) if bit}
        edges = set(self.dependencies)
        edges.update((route.source, route.target) for route in self.routes if route.name in active)
        pending = {name for name in self.node_map if name in active}
        result = []
        while pending:
            ready = sorted(name for name in pending if not any(target == name and source in pending for source, target in edges))
            if not ready:
                raise ValueError("Selected execution dependencies contain a cycle")
            result.extend(ready)
            pending.difference_update(ready)
        return result

    def enumerate(self, budget, max_bits=22):
        if len(self.names) > max_bits:
            raise ValueError("Exact graph enumeration exceeds max_bits; supply feasible witnesses and partition profiles")
        result = []
        for bits in itertools.product((0, 1), repeat=len(self.names)):
            decision = Decision(bits)
            try:
                self.validate(decision, budget)
            except ValueError:
                continue
            result.append(decision)
        return result

    def affected(self, owner, global_state=False):
        if global_state:
            return set(self.names)
        affected = {node.name for node in self.nodes if node.owner == owner}
        edges = set(self.dependencies) | {(route.source, route.target) for route in self.routes}
        while True:
            expanded = affected | {target for source, target in edges if source in affected}
            if expanded == affected:
                break
            affected = expanded
        return affected | {route.name for route in self.routes if route.source in affected or route.target in affected}

    def partitions(self):
        labels = {node.name: ("owner:" + node.owner if node.owner else "endpoint:" + (node.partition or node.name))
                  for node in self.nodes}
        labels.update({route.name: labels[route.source] for route in self.routes})
        return [tuple(name for name in self.names if labels[name] == label) for label in sorted(set(labels.values()))]
