import math
from dataclasses import dataclass, field

from .storage import fingerprint


def failure_allocation(delta, identifier, count):
    if (not 0 < delta < 1 or isinstance(identifier, bool) or isinstance(count, bool)
            or int(identifier) != identifier or int(count) != count or identifier < 1 or count < 1):
        raise ValueError("Anytime confidence uses a failure level in (0,1) and positive indices")
    return 36 * delta / (math.pi ** 4 * identifier ** 2 * count ** 2)


def anytime_radius(delta, identifier, count):
    level = failure_allocation(delta, identifier, count)
    return math.sqrt(2 / count * math.log(2 / level))


@dataclass
class Estimate:
    identifier: int
    scores: list = field(default_factory=list)

    def add(self, score):
        if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Continuation scores are finite numbers in [0,1]")
        self.scores.append(float(score))

    def interval(self, delta):
        if not self.scores:
            return 0.0, 1.0
        mean = sum(self.scores) / len(self.scores)
        radius = anytime_radius(delta, self.identifier, len(self.scores))
        return max(0.0, mean - radius), min(1.0, mean + radius)

    @property
    def mean(self):
        return sum(self.scores) / len(self.scores) if self.scores else None


class ConfidenceRegistry:
    def __init__(self, delta=0.05):
        if not 0 < delta < 1:
            raise ValueError("The total confidence failure budget lies in (0,1)")
        self.delta, self.entries, self.keys = float(delta), {}, {}
        self.next_identifier = 1

    def key(self, task_state, version, program, graph, injection=None):
        fields = {"task_state": task_state, "version": version, "program": program,
                  "graph": graph, "injection": injection}
        key = fingerprint(fields)
        if key not in self.entries:
            self.entries[key] = Estimate(self.next_identifier)
            self.keys[key] = fields
            self.next_identifier += 1
        return key

    def record(self):
        return {"delta": self.delta, "next_identifier": self.next_identifier, "keys": self.keys,
                "entries": {key: {"identifier": value.identifier, "scores": value.scores}
                            for key, value in self.entries.items()}}

    def lookup(self, task_state, version, program, graph, injection=None):
        return self.entries.get(fingerprint({"task_state": task_state, "version": version, "program": program,
                                             "graph": graph, "injection": injection}))

    @classmethod
    def restore(cls, value):
        registry = cls(value["delta"])
        if (isinstance(value["next_identifier"], bool) or int(value["next_identifier"]) != value["next_identifier"]
                or value["next_identifier"] < 1):
            raise ValueError("The next confidence identifier is a positive integer")
        registry.next_identifier = int(value["next_identifier"])
        registry.keys = dict(value["keys"])
        if set(registry.keys) != set(value["entries"]):
            raise ValueError("Persistent confidence keys match their estimates")
        registry.entries = {}
        for key, row in value["entries"].items():
            if key != fingerprint(registry.keys[key]):
                raise ValueError("Confidence state has mismatched persistent identities")
            if isinstance(row["identifier"], bool) or int(row["identifier"]) != row["identifier"]:
                raise ValueError("Confidence identifiers are integers")
            estimate = Estimate(int(row["identifier"]))
            for score in row["scores"]:
                estimate.add(score)
            registry.entries[key] = estimate
        ids = [entry.identifier for entry in registry.entries.values()]
        if len(set(ids)) != len(ids) or min(ids, default=1) < 1 or max(ids, default=0) >= registry.next_identifier:
            raise ValueError("Confidence state reuses an identifier")
        return registry
