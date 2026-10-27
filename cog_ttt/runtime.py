import math
from dataclasses import asdict
from pathlib import Path

from .budget import BudgetExhausted, BudgetLedger
from .confidence import ConfidenceRegistry
from .configuration import validate_case
from .execution import DeviceMeter, GraphExecutor
from .graph import Decision
from .models import VersionStore
from .search import JointSearch
from .storage import fingerprint, read_json, write_json
from .training import Preference, VerifiedTarget, contrast_preference, train_program


class Session:
    def __init__(self, case, configuration, worker, owners, directory, resume=False):
        self.case, self.configuration, self.worker, self.owners = case, configuration, worker, owners
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.graph, self.initial, supplied, self.settings, self.budget = validate_case(case, configuration)
        self.executor = GraphExecutor(self.graph, owners, worker)
        self.versions = VersionStore(owners)
        model_identities = {name: getattr(owner, "identity", None) for name, owner in owners.items()}
        if case["benchmark"]["data_kind"] != "software_fixture" and any(value is None for value in model_identities.values()):
            raise ValueError("Benchmark parameter owners retain immutable model and tokenizer identities")
        self.identity = fingerprint({"case": case, "configuration": configuration, "models": model_identities,
                                     "score_source": self.executor.score_source})
        if resume:
            self.state = read_json(self.directory / "state.json")
            if self.state["identity"] != self.identity:
                raise ValueError("Resume requires the same task, graph, models, and configuration")
            self.registry = ConfidenceRegistry.restore(self.state["confidence"])
            self.ledger = BudgetLedger.restore(self.state["ledger"])
            self.versions.restore(self.directory / "parameters", self.state["parameters"])
            for entry in self.ledger.entries:
                if entry.get("status") == "reserved":
                    entry["status"] = "interrupted_charged_upper_bound"
                    if entry.get("program") in self.state["raw_costs"]:
                        self.state["raw_costs"][entry["program"]] += entry["cost"]
            for status in self.state["program_status"].values():
                if status.get("status") == "running":
                    status["status"] = "interrupted"
            self.state["raw_costs"] = {name: cost for name, cost in self.state["raw_costs"].items()
                                       if name == "null" or name in self.versions.candidates}
        else:
            if (self.directory / "state.json").exists():
                raise ValueError("An existing run requires --resume or a new output directory")
            self.registry = ConfidenceRegistry(configuration["confidence"]["delta"])
            self.ledger = BudgetLedger(self.budget["total_gpu_seconds"], self.budget["reserve_gpu_seconds"])
            self.state = {"identity": self.identity, "checkpoint": worker.checkpoint(case["task"]),
                          "models": model_identities, "score_source": self.executor.score_source,
                          "probe_count": 0, "pilot_count": 0, "raw_costs": {"null": 0.0},
                          "program_status": {}, "credit": {}, "verified": {}, "data_version": None,
                          "accepted": [], "witnesses": [decision.bits for decision in supplied]}
        self.checkpoint = self.state["checkpoint"]
        witnesses = [Decision(tuple(bits)) for bits in self.state["witnesses"]]
        complete = None
        if len(self.graph.names) <= self.settings["max_graph_bits"]:
            complete = self.graph.enumerate(min(self.settings["execution_budget"], self.settings["total_budget"]),
                                            self.settings["max_graph_bits"])
            witnesses += complete
        self.search = JointSearch(self.graph, self.registry, self.checkpoint["task_state"], self.versions.version,
                                  self.settings, witnesses, complete)
        if "partitions" in self.state:
            self.search.partitions = [tuple(group) for group in self.state["partitions"]]
            self.search.refinements = self.state.get("oracle_refinements", 0)
        if "fit_roles" in self.state:
            self.search.fit_roles = dict(self.state["fit_roles"])
        self.refresh_budget()
        self.recover_pending_probe()
        self.save()

    def save(self):
        self.state["confidence"] = self.registry.record()
        self.state["ledger"] = self.ledger.record()
        self.state["parameters"] = self.versions.save(self.directory / "parameters")
        if hasattr(self, "search"):
            self.state["witnesses"] = [decision.bits for decision in self.search.witnesses.values()]
            self.state["partitions"] = self.search.partitions
            self.state["oracle_refinements"] = self.search.refinements
            self.state["fit_roles"] = self.search.fit_roles
        write_json(self.directory / "state.json", self.state)

    def action(self, stage, bound, function, program=None, final=False, result_key=None, external_required=False):
        index = self.ledger.reserve_action(stage, bound, final_execution=final, program=program,
                                           version=self.versions.version)
        self.save()
        try:
            with DeviceMeter(self.owners) as meter:
                result = function()
            external = result.get("external_gpu_seconds", None if external_required else 0.0)
            observed = None if external is None else meter.gpu_seconds + external
            self.ledger.settle(index, observed, elapsed_seconds=meter.elapsed)
        except BaseException as error:
            if self.ledger.entries[index]["status"] == "reserved":
                self.ledger.entries[index].update(status="failed_charged_upper_bound", error=type(error).__name__)
            if program in self.state["raw_costs"]:
                self.state["raw_costs"][program] += self.ledger.entries[index]["cost"]
            self.save()
            raise
        if program in self.state["raw_costs"]:
            self.state["raw_costs"][program] += self.ledger.entries[index]["cost"]
        if result_key is not None:
            self.state[result_key] = result
        self.save()
        return result

    def costs(self):
        self.refresh_budget()
        quantum = self.settings["cost_quantum_gpu_seconds"]
        return {program: cost / quantum for program, cost in self.state["raw_costs"].items()}

    def refresh_budget(self):
        if hasattr(self, "search"):
            self.search.operational_execution_budget = self.ledger.remaining / self.settings["cost_quantum_gpu_seconds"]

    def probe(self, program, decision, injection=None, seed=None, stage="probe"):
        self.recover_pending_probe()
        if self.state["probe_count"] >= self.settings["max_probes"]:
            raise BudgetExhausted("The checkpoint's declared probe limit has been reached")
        cost = self.graph.cost(decision)
        bound = cost * self.settings["cost_quantum_gpu_seconds"]
        budget = self.search.graph_budget(self.costs().get(program, 0.0))
        self.graph.validate(decision, budget)
        required_witness_cost = max(cost, self.graph.cost(self.initial)) if program == "null" else cost
        if required_witness_cost + self.costs().get(program, 0.0) + bound / self.settings["cost_quantum_gpu_seconds"] > self.settings["total_budget"]:
            raise BudgetExhausted("The probe's attributed validation cost would empty its feasible graph family")
        if not self.ledger.permits(bound):
            raise BudgetExhausted("A legal probe would consume the protected execution reserve")
        key = self.registry.key(self.checkpoint["task_state"], self.versions.version, program,
                                decision.identifier, injection)
        self.search.register_witness(decision)
        self.state["probe_count"] += 1
        seed = (int(self.case.get("seed", 0)) + 104729 * len(self.ledger.entries)) if seed is None else seed
        self.state["pending_probe"] = {"key": key, "action_id": len(self.ledger.entries), "program": program,
                                       "version": self.versions.version, "seed": seed, "graph": decision.bits,
                                       "before_count": len(self.registry.entries[key].scores)}
        self.versions.activate(program)
        try:
            result = self.action(stage, bound, lambda: self.executor.run(decision, self.checkpoint, seed, budget,
                                                                        injection=injection), program,
                                 result_key="pending_probe_result")
        finally:
            self.versions.activate()
        return self.commit_probe()

    def recover_pending_probe(self):
        if "pending_probe_result" in self.state:
            self.commit_probe()
        else:
            self.state.pop("pending_probe", None)

    def commit_probe(self):
        pending = self.state["pending_probe"]
        result = self.state["pending_probe_result"]
        estimate = self.registry.entries[pending["key"]]
        if len(estimate.scores) != pending["before_count"] or pending["version"] != self.versions.version:
            raise ValueError("A durable probe result is committed once at its original parameter version")
        record = {**pending, "identifier": estimate.identifier, "result": result}
        path = self.directory / "observations" / f"{pending['action_id']:08d}.json"
        if path.exists():
            if fingerprint(read_json(path)) != fingerprint(record):
                raise ValueError("A durable observation identity has different recorded content")
        else:
            write_json(path, record)
        estimate.add(result["feedback"]["score"])
        self.search.register_witness(Decision(tuple(pending["graph"])))
        self.state.pop("pending_probe")
        self.state.pop("pending_probe_result")
        self.save()
        return result, estimate

    def collect_credit(self, identifier, request):
        node = self.graph.node_map[request["node"]]
        graph = self.graph.decision(request["graph"])
        if node.kind != "model":
            raise ValueError("Credit records belong to a trainable model operation")
        record_key = f"{self.versions.version}:{identifier}"
        if record_key not in self.state["credit"]:
            prefix_seed = int(self.case.get("seed", 0)) + 1009 * (len(self.state["credit"]) + 1)
            anchor, _ = self.probe("null", graph, seed=prefix_seed, stage="credit_anchor")
            row = next(row for row in anchor["trace"] if row["node"] == node.name)
            output_bound = node.cost * self.settings["cost_quantum_gpu_seconds"]
            alternative = self.action("credit_alternative", output_bound,
                          lambda: self.owners[node.owner].sample(row["history"], seed=prefix_seed + 1))
            self.state["credit"][record_key] = {"history": row["history"], "first": row["output"],
                                               "second": alternative, "prefix_seed": prefix_seed}
            self.save()
        record = self.state["credit"][record_key]
        graphs = [graph]
        if request.get("alternative_graph") is not None:
            graphs.append(self.graph.decision(request["alternative_graph"]))
        estimates = []
        repetitions = int(request["repetitions"])
        if repetitions < 1:
            raise ValueError("Continuation comparisons declare positive repetition counts")
        for context in graphs:
            self.graph.validate(context, self.settings["execution_budget"])
            for output in (record["first"], record["second"]):
                injection = {"node": node.name, "output": {key: output[key] for key in ("text", "token_ids")},
                             "prefix_seed": record["prefix_seed"], "history_fingerprint": fingerprint(record["history"])}
                key = self.registry.key(self.checkpoint["task_state"], self.versions.version, "null",
                                        context.identifier, injection)
                estimate = self.registry.entries[key]
                while len(estimate.scores) < repetitions:
                    self.probe("null", context, injection, record["prefix_seed"] + 10000 + len(estimate.scores),
                               stage="credit_contrast")
                estimates.append(estimate)
        return contrast_preference(node.owner, graph.identifier, self.checkpoint["task_state"], record["history"],
                                   record["first"], record["second"], estimates, self.registry.delta)

    def collect_data(self):
        if self.state["data_version"] == self.versions.version:
            return
        preferences = [self.collect_credit(str(index), request) for index, request in enumerate(self.case.get("credit_requests", []))]
        verified = []
        seen = set()
        for program in self.case.get("programs", []):
            graph = self.graph.decision(program["graph"])
            if graph.identifier in seen:
                continue
            seen.add(graph.identifier)
            key = f"{self.versions.version}:{graph.identifier}"
            if key not in self.state["verified"]:
                anchor, _ = self.probe("null", graph, stage="verified_anchor")
                response = self.action("tool_verification", self.budget["verification_gpu_seconds"],
                    lambda: self.worker.call("verified_targets", task_state=self.checkpoint["task_state"],
                                             graph=graph.bits, trace=anchor["trace"]), external_required=True)
                rows = []
                by_node = {row["node"]: row for row in anchor["trace"] if row["kind"] == "model"}
                for target in response.get("targets", []):
                    row = by_node[target["node"]]
                    rows.append(asdict(VerifiedTarget(row["owner"], graph.identifier, self.checkpoint["task_state"],
                                  row["history"], target["requirement"], target["target"], target["verification"])))
                self.state["verified"][key] = rows
                self.save()
            verified.extend(VerifiedTarget(**row) for row in self.state["verified"][key])
        self.state["preferences"] = [asdict(row) for row in preferences]
        self.state["verified_records"] = [asdict(row) for row in verified]
        self.state["data_version"] = self.versions.version
        self.save()

    def materialize_next(self):
        for specification in self.case.get("programs", []):
            identifier = specification["id"]
            key = f"{self.versions.version}:{identifier}"
            previous = self.state["program_status"].get(key, {}).get("status")
            if previous is not None and previous not in {"interrupted", "failed"}:
                continue
            if self.state["pilot_count"] >= self.settings["max_pilots"]:
                return False
            bound = self.budget["pilot_gpu_seconds"]
            if not self.ledger.permits(bound):
                return False
            self.state["pilot_count"] += 1
            self.state["program_status"][key] = {"status": "running"}
            self.state["raw_costs"][identifier] = 0.0
            graph = self.graph.decision(specification["graph"])
            self.versions.activate()
            try:
                result = self.action("training_pilot", bound,
                    lambda: train_program(self.owners[specification["owner"]], graph.identifier, self.checkpoint["task_state"],
                                          [Preference(**row) for row in self.state.get("preferences", [])],
                                          [VerifiedTarget(**row) for row in self.state.get("verified_records", [])],
                                          specification["steps"], specification["learning_rate"], self.configuration["training"],
                                          int(self.case.get("seed", 0)) + self.state["pilot_count"]), identifier)
            except BaseException:
                self.state["program_status"][key] = {"status": "failed"}
                self.state["raw_costs"].pop(identifier, None)
                self.save()
                raise
            status = {key: value for key, value in result.items() if key != "state"}
            if result["status"] == "trained":
                self.versions.add(identifier, specification["owner"], result["state"], self.versions.version)
                if not any(self.graph.cost(decision) <= self.search.graph_budget(self.costs()[identifier])
                           for decision in self.search.witnesses.values()):
                    status["status"] = "trained_outside_joint_budget"
                    self.versions.candidates.pop(identifier)
                    self.state["raw_costs"].pop(identifier)
            else:
                self.state["raw_costs"].pop(identifier)
            self.state["program_status"][key] = status
            self.save()
            return True
        return False

    def family_complete(self):
        allowed = {"trained", "skipped_no_training_signal", "trained_outside_joint_budget"}
        return all(self.state["program_status"].get(f"{self.versions.version}:{specification['id']}", {}).get("status") in allowed
                   for specification in self.case.get("programs", []))

    def accept(self, result):
        chosen = result["incumbent"]
        if chosen["program"] == "null":
            return False
        reference = next((pair for pair in result["pairs"] if pair["program"] == "null" and pair["graph"] == self.initial), None)
        if reference is None:
            return False
        if chosen["lower"] < reference["upper"] - self.settings["acceptance_tolerance"]:
            return False
        owner = self.versions.candidates[chosen["program"]]["owner"]
        affected = sorted(self.graph.affected(owner, self.case.get("global_hidden_state", False)))
        report = self.search.report(result)
        report.update(owner=owner, invalidated_graph_bits=affected,
                      invalidated_features=self.search.model(chosen["program"], chosen["training_cost"])[0].affected_features(
                          owner, self.case.get("global_hidden_state", False)),
                      reference_upper=reference["upper"], acceptance_tolerance=self.settings["acceptance_tolerance"])
        self.state["accepted"].append(report)
        self.versions.accept(chosen["program"])
        self.state["raw_costs"] = {"null": 0.0}
        self.search.version = self.versions.version
        self.state["data_version"] = None
        self.save()
        return True

    def run(self):
        if self.state.get("completed"):
            return read_json(self.directory / "result.json")
        if self.state.get("final_execution") is not None:
            report = self.state["final_report"]
            report.update(ledger=self.ledger.record(), execution=self.state["final_execution"])
            write_json(self.directory / "result.json", report)
            self.state["completed"] = True
            self.save()
            return report
        reason = "probe_limit"
        try:
            while self.state["probe_count"] < self.settings["max_probes"]:
                if self.state["pilot_count"] < self.settings["max_pilots"]:
                    self.collect_data()
                if self.materialize_next():
                    continue
                result = self.search.bounds(self.costs())
                if self.search.refine(result):
                    self.save()
                    continue
                if self.accept(result):
                    continue
                if result["gap"] <= self.settings["epsilon"] and self.family_complete():
                    reason = "certified_tolerance"
                    break
                pair = self.search.next_probe(result)
                if pair is None:
                    reason = "certified_screening"
                    break
                self.probe(pair["program"], pair["graph"])
        except BudgetExhausted:
            if any(entry.get("status") == "cost_bound_exceeded" for entry in self.ledger.entries):
                raise
            reason = "budget_exhausted"
        result = self.search.bounds(self.costs())
        accepted_pairs = [pair for pair in result["pairs"] if pair["program"] == "null"
                          and self.graph.cost(pair["graph"]) * self.settings["cost_quantum_gpu_seconds"] <= self.ledger.remaining]
        chosen = max(accepted_pairs, key=lambda pair: (pair["lower"], -self.graph.cost(pair["graph"])))
        result["incumbent"] = chosen
        result["gap"] = max(0.0, max(item["oracle"]["upper"] for item in result["programs"].values()) - chosen["lower"])
        evaluated = [pair for pair in result["pairs"] if pair["observations"]]
        result["frontier_gap"] = (max(0.0, max(pair["direct_upper"] for pair in evaluated) - chosen["lower"])
                                  if evaluated else None)
        self.versions.activate()
        report = self.search.report(result)
        report.update(benchmark=self.case["benchmark"], task=self.case["task"], seed=int(self.case.get("seed", 0)),
                      stopping_reason=reason, declared_program_family_complete=self.family_complete(),
                      accepted_versions=self.state["accepted"], probe_count=self.state["probe_count"],
                      pilot_count=self.state["pilot_count"],
                      case_identity=self.identity)
        self.state["final_report"] = report
        self.save()
        execution_budget = self.search.graph_budget(self.costs()["null"])
        execution = self.action("final_execution", self.graph.cost(chosen["graph"]) * self.settings["cost_quantum_gpu_seconds"],
                     lambda: self.executor.run(chosen["graph"], self.checkpoint, int(self.case.get("seed", 0)),
                                               execution_budget, online=False), final=True,
                     result_key="final_execution")
        report.update(ledger=self.ledger.record(), execution=execution)
        write_json(self.directory / "result.json", report)
        self.state["completed"] = True
        self.save()
        return report
