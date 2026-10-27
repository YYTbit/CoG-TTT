import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cog_ttt.configuration import read_configuration
from examples.fixture import FixtureEnvironment


ROOT = Path(__file__).resolve().parents[1]


class BinaryOwner:
    def __init__(self, name="shared", logit=0.0):
        self.name = name
        self.logit = torch.nn.Parameter(torch.tensor(float(logit), dtype=torch.float64))

    def parameters(self):
        return [self.logit]

    def snapshot(self):
        return {"logit": self.logit.detach().clone()}

    def restore(self, state):
        with torch.no_grad():
            self.logit.copy_(state["logit"])

    def log_probability(self, history, target, requirement=None):
        token = target["text"] if isinstance(target, dict) else target
        if token not in {"yes", "no"}:
            raise ValueError("Unknown binary fixture token")
        return torch.nn.functional.logsigmoid(self.logit if token == "yes" else -self.logit)

    def sample(self, history, seed=0, **settings):
        selected = np.random.default_rng(seed).random() < float(self.logit.detach().sigmoid())
        return {"text": "yes" if selected else "no", "token_ids": [int(selected)]}

    def save_state(self, path, state):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load_state(self, path):
        return torch.load(path, weights_only=True)


class LocalWorker:
    def __init__(self):
        self.environment = FixtureEnvironment()
        self.description = self.environment.describe()
        self.online_sources = set(self.description["online_sources"])

    def call(self, method, **arguments):
        return getattr(self.environment, method)(**arguments)

    def checkpoint(self, task):
        return self.call("checkpoint", task=task)

    def restore(self, checkpoint):
        result = self.call("restore", snapshot=checkpoint["snapshot"], task_state=checkpoint["task_state"])
        if not result["restored"]:
            raise ValueError("Fixture checkpoint changed")


@pytest.fixture
def settings():
    return read_configuration(ROOT / "configs/paper.yaml")


@pytest.fixture
def routing_case():
    return json.loads((ROOT / "examples/protocol_check.json").read_text())


@pytest.fixture
def model_case():
    return {
        "benchmark": {"name": "protocol-fixture", "revision": "1", "data_kind": "software_fixture"},
        "environment": {"command": ["python", "-m", "cog_ttt.worker", "--factory", "examples.fixture:create"]},
        "task": {"id": "binary-training-check", "request": "Return a permitted fixture token."}, "seed": 0,
        "owners": {"shared": {"checkpoint": "binary-software-fixture"}},
        "graph": {"nodes": [{"name": "response", "cost": 1, "operation": "respond", "owner": "shared"}],
                  "terminals": ["response"]},
        "initial_graph": ["response"],
        "programs": [{"id": "owner_update", "owner": "shared", "graph": ["response"], "steps": 3, "learning_rate": 0.5}],
        "objective": {"total_budget": 100, "execution_budget": 1, "cost_quantum_gpu_seconds": 1.0, "utility_per_cost_unit": 0.01},
        "budget": {"total_gpu_seconds": 12.0, "reserve_gpu_seconds": 1.0, "pilot_gpu_seconds": 2.0, "verification_gpu_seconds": 1.0}
    }
