import math
import time

from .storage import fingerprint


class DeviceMeter:
    def __init__(self, owners):
        self.devices = set()
        for owner in owners.values():
            for parameter in owner.parameters():
                if parameter.device.type == "cuda":
                    self.devices.add(parameter.device.index)

    def synchronize(self):
        if self.devices:
            import torch
            for device in self.devices:
                torch.cuda.synchronize(device)

    def __enter__(self):
        self.synchronize()
        self.started = time.perf_counter()
        return self

    def __exit__(self, *arguments):
        self.synchronize()
        self.elapsed = time.perf_counter() - self.started
        self.gpu_seconds = self.elapsed * len(self.devices)


class GraphExecutor:
    def __init__(self, graph, owners, worker):
        self.graph, self.owners, self.worker = graph, owners, worker
        self.score_source = getattr(worker, "score_source", None)
        if self.score_source is None and len(worker.online_sources) == 1:
            self.score_source = next(iter(worker.online_sources))
        if self.score_source not in worker.online_sources:
            raise ValueError("A checkpoint binds one predeclared continuation-score source")

    def run(self, decision, checkpoint, seed, execution_budget, injection=None, online=True):
        self.graph.validate(decision, execution_budget)
        self.worker.restore(checkpoint)
        selected = dict(zip(self.graph.names, decision.bits))
        nodes = {node.name: node for node in self.graph.nodes}
        outputs, trace, external_usage = {}, [], []
        injected_node = injection.get("node") if injection else None
        if injection and (injected_node not in nodes or not selected[injected_node]
                          or nodes[injected_node].kind != "model"):
            raise ValueError("A continuation contrast injects an active trainable node")
        reached_injection = False
        for index, name in enumerate(self.graph.execution_order(decision)):
            node = nodes[name]
            dependencies = {source: outputs[source] for source, target in self.graph.dependencies if target == name}
            routed = [{"route": route.name, "port": route.port, "source": route.source,
                       "output": outputs[route.source]} for route in self.graph.routes
                      if route.target == name and selected[route.name]]
            arguments = {"task_state": checkpoint["task_state"], "public_task": checkpoint["public_task"],
                         "node": name, "operation": node.operation, "dependencies": dependencies,
                         "routed_inputs": routed, "settings": node.settings}
            node_seed = (injection["prefix_seed"] if injection and not reached_injection else seed) + index
            if node.kind == "tool":
                result = self.worker.call("tool", seed=node_seed, **arguments)
                output = result["output"]
                usage = result.get("external_gpu_seconds")
                if usage is not None and (not math.isfinite(usage) or usage < 0):
                    raise ValueError("Tool GPU accounting is finite and nonnegative")
                external_usage.append(usage)
                row = {"node": name, "kind": "tool", "output": output, "native": result}
            else:
                context = self.worker.call("prepare_node", **arguments)
                history = context["history"]
                if name == injected_node:
                    if fingerprint(history) != injection["history_fingerprint"]:
                        raise ValueError("The counterfactual node input changed at the frozen checkpoint")
                    output = injection["output"]
                    reached_injection = True
                else:
                    output = self.owners[node.owner].sample(history, seed=node_seed,
                                                           max_tokens=node.settings.get("max_output_tokens"),
                                                           temperature=node.settings.get("temperature", 1.0))
                row = {"node": name, "kind": "model", "owner": node.owner, "history": history,
                       "output": output, "injected": name == injected_node}
            outputs[name] = output
            trace.append(row)
        result = {"graph": decision.bits, "task_state": checkpoint["task_state"], "trace": trace,
                  "outputs": outputs, "terminals": {name: outputs[name] for name in self.graph.terminals},
                  "external_gpu_seconds": (sum(external_usage) if all(value is not None for value in external_usage)
                                           else None)}
        if online:
            feedback = self.worker.call("continuation_score", task_state=checkpoint["task_state"],
                                        outputs=outputs, terminals=result["terminals"], seed=seed)
            score = feedback.get("score")
            if (feedback.get("task_state") != checkpoint["task_state"]
                    or feedback.get("source") != self.score_source
                    or isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("Online feedback uses a declared source and a bounded score at this checkpoint")
            result["feedback"] = feedback
            external_usage.append(feedback.get("external_gpu_seconds"))
            if any(value is not None and (not math.isfinite(value) or value < 0) for value in external_usage):
                raise ValueError("Native GPU accounting is finite and nonnegative")
            result["external_gpu_seconds"] = (sum(external_usage) if all(value is not None for value in external_usage)
                                               else None)
        return result
