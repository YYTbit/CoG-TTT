import math
from dataclasses import dataclass, field


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class BudgetLedger:
    limit: float
    reserve: float
    entries: list = field(default_factory=list)

    def __post_init__(self):
        if not math.isfinite(self.limit) or not math.isfinite(self.reserve) or not 0 <= self.reserve <= self.limit:
            raise ValueError("The execution reserve lies within the total operational budget")

    @property
    def spent(self):
        return sum(entry["cost"] for entry in self.entries)

    @property
    def remaining(self):
        return max(0.0, self.limit - self.spent)

    def permits(self, upper_cost, final_execution=False):
        if not math.isfinite(upper_cost) or upper_cost < 0:
            raise ValueError("An operation declares a finite nonnegative cost bound")
        protected = 0 if final_execution else self.reserve
        return self.spent + upper_cost + protected <= self.limit + 1e-12

    def charge(self, stage, cost, **metadata):
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("The operational ledger records every finite incurred cost")
        self.entries.append({"stage": stage, "cost": float(cost), **metadata})
        if self.spent > self.limit + 1e-12:
            raise BudgetExhausted("Observed cost exceeded the declared operational cap")

    def reserve_action(self, stage, upper_cost, final_execution=False, **metadata):
        if not self.permits(upper_cost, final_execution):
            raise BudgetExhausted("The action would consume the protected execution reserve")
        index = len(self.entries)
        self.charge(stage, upper_cost, status="reserved", upper_cost=float(upper_cost), **metadata)
        return index

    def settle(self, index, observed, **metadata):
        entry = self.entries[index]
        if entry.get("status") != "reserved":
            raise ValueError("An action reservation is settled once")
        if observed is not None and (not math.isfinite(observed) or observed < 0):
            raise ValueError("Observed action cost is finite and nonnegative")
        entry.update(cost=entry["upper_cost"] if observed is None else float(observed), status="completed",
                     accounting="declared_upper_bound" if observed is None else "measured_gpu_seconds", **metadata)
        if entry["cost"] > entry["upper_cost"] + 1e-9 or self.spent > self.limit + 1e-9:
            entry["status"] = "cost_bound_exceeded"
            raise BudgetExhausted("Measured cost exceeded its declared action bound")

    def record(self):
        return {"limit": self.limit, "reserve": self.reserve, "spent": self.spent,
                "remaining": self.remaining, "entries": self.entries}

    @classmethod
    def restore(cls, record):
        result = cls(record["limit"], record["reserve"])
        for entry in record["entries"]:
            values = dict(entry)
            result.charge(values.pop("stage"), values.pop("cost"), **values)
        if abs(result.spent - record["spent"]) > 1e-9:
            raise ValueError("Resumed cost accounting differs from its stored total")
        return result
