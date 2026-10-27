import json


class FixtureEnvironment:
    """Deterministic software fixture for the environment protocol."""

    def describe(self):
        return {"protocol": 1, "name": "protocol-fixture", "revision": "1", "data_kind": "software_fixture",
                "final_labels_online": False, "online_sources": ["execution_check"]}

    def checkpoint(self, task):
        return {"task_state": task["id"], "snapshot": {"task": task}, "public_task": task}

    def restore(self, snapshot, task_state):
        return {"restored": snapshot["task"]["id"] == task_state, "task_state": task_state}

    def prepare_node(self, task_state, public_task, node, operation, dependencies, routed_inputs, settings):
        return {"history": [{"role": "user", "content": json.dumps({"request": public_task["request"],
                            "operation": operation, "dependencies": dependencies, "inputs": routed_inputs}, sort_keys=True)}]}

    def tool(self, task_state, public_task, node, operation, dependencies, routed_inputs, settings, seed):
        if operation == "constant":
            output = {"value": settings["value"]}
        elif operation == "forward":
            output = routed_inputs[0]["output"]
        else:
            raise ValueError("Unknown fixture operation")
        return {"output": output, "external_gpu_seconds": 0.0}

    def continuation_score(self, task_state, outputs, terminals, seed):
        output = next(iter(terminals.values()))
        score = min(1.0, output["value"] / 2) if "value" in output else float(output["text"] == "yes")
        return {"task_state": task_state, "source": "execution_check", "score": score, "external_gpu_seconds": 0.0}

    def verified_targets(self, task_state, graph, trace):
        targets = [{"node": row["node"], "requirement": "Return the permitted fixture token.", "target": "yes",
                    "verification": {"passed": True, "tool_result_id": "fixture-token-check",
                                     "task_state": task_state, "source": "execution_check"}}
                   for row in trace if row["kind"] == "model"]
        return {"targets": targets, "external_gpu_seconds": 0.0}

    def final_evaluate(self, task, execution):
        output = next(iter(execution["terminals"].values()))
        value = float(output.get("value") == 2 or output.get("text") == "yes")
        return {"metrics": {"fixture_check": {"value": value, "unit": "rate"}}, "evaluator": "software_fixture"}


def create(configuration):
    return FixtureEnvironment()
