import copy
import subprocess
import sys

import pytest

from cog_ttt.budget import BudgetExhausted, BudgetLedger
from cog_ttt.evaluation import native_value, paired_interval, summarize, supported_citation_pairs
from cog_ttt.native import NativeWorker
from cog_ttt.runtime import Session
from cog_ttt.storage import read_json
from conftest import BinaryOwner, LocalWorker, ROOT


def test_failed_actions_and_protected_reserve_are_charged():
    ledger = BudgetLedger(10, 3)
    index = ledger.reserve_action("pilot", 6)
    assert not ledger.permits(2)
    ledger.settle(index, 4)
    restored = BudgetLedger.restore(ledger.record())
    assert restored.spent == 4
    with pytest.raises(BudgetExhausted):
        restored.reserve_action("probe", 4)
    index = restored.reserve_action("final_execution", 6, final_execution=True)
    restored.settle(index, None)
    assert restored.remaining == 0


def test_incorrect_action_bound_stops_further_execution(tmp_path, routing_case, settings):
    worker = LocalWorker()
    original = worker.call
    def excessive_usage(method, **arguments):
        result = original(method, **arguments)
        if method == "tool":
            result["external_gpu_seconds"] = 20.0
        return result
    worker.call = excessive_usage
    session = Session(routing_case, settings, worker, {}, tmp_path)
    with pytest.raises(BudgetExhausted, match="declared action bound"):
        session.run()
    assert session.ledger.entries[-1]["status"] == "cost_bound_exceeded"
    assert all(entry["stage"] != "final_execution" for entry in session.ledger.entries)


def test_native_worker_protocol_and_cli(tmp_path, routing_case):
    with NativeWorker([sys.executable, "-m", "cog_ttt.worker", "--factory", "examples.fixture:create"],
                      routing_case["benchmark"], tmp_path / "worker.log", timeout=5) as worker:
        checkpoint = worker.checkpoint(routing_case["task"])
        worker.restore(checkpoint)
    result = subprocess.run([sys.executable, "-m", "cog_ttt", "check", "--case", "examples/protocol_check.json"],
                            cwd=ROOT, text=True, capture_output=True, check=True)
    assert '"valid": true' in result.stdout


def test_complete_session_resume_and_final_label_separation(tmp_path, routing_case, settings):
    settings["search"]["max_probes"] = 4
    worker = LocalWorker()
    final_calls = []
    original = worker.call
    def call(method, **arguments):
        if method == "final_evaluate":
            final_calls.append(method)
        return original(method, **arguments)
    worker.call = call
    session = Session(routing_case, settings, worker, {}, tmp_path)
    report = session.run()
    assert not final_calls
    assert report["probe_count"] == 4
    assert report["decision_gap"] > settings["search"]["epsilon"]
    assert report["stopping_reason"] == "probe_limit"
    assert report["ledger"]["entries"][-1]["stage"] == "final_execution"
    resumed = Session(routing_case, settings, worker, {}, tmp_path, resume=True)
    assert resumed.run() == read_json(tmp_path / "result.json")


def test_training_pilot_uses_native_verified_feedback(tmp_path, model_case, settings):
    settings["search"]["max_probes"] = 6
    settings["search"]["max_pilots"] = 1
    owner = BinaryOwner()
    session = Session(model_case, settings, LocalWorker(), {"shared": owner}, tmp_path)
    report = session.run()
    status = session.state["program_status"]["0:owner_update"]
    assert status["status"] == "trained"
    assert status["verified_targets"] == 1
    assert report["pilot_count"] == 1
    assert float(owner.logit.detach()) == 0
    assert session.versions.candidates["owner_update"]["state"]["logit"] > 0


def test_failed_probe_survives_resume_without_losing_its_charge(tmp_path, routing_case, settings):
    worker = LocalWorker()
    session = Session(routing_case, settings, worker, {}, tmp_path)
    original = worker.call
    def fail(method, **arguments):
        if method == "continuation_score":
            raise RuntimeError("Injected software failure")
        return original(method, **arguments)
    worker.call = fail
    with pytest.raises(RuntimeError, match="Injected"):
        session.probe("null", session.initial)
    charge = session.ledger.spent
    assert charge > 0
    resumed = Session(routing_case, settings, LocalWorker(), {}, tmp_path, resume=True)
    assert resumed.ledger.spent == charge
    assert resumed.registry.next_identifier == session.registry.next_identifier
    assert resumed.state["probe_count"] == 1


def test_durable_probe_outcome_is_committed_once_after_interruption(tmp_path, routing_case, settings, monkeypatch):
    session = Session(routing_case, settings, LocalWorker(), {}, tmp_path)
    def interrupt_commit():
        raise RuntimeError("Interrupted after durable probe execution")
    monkeypatch.setattr(session, "commit_probe", interrupt_commit)
    with pytest.raises(RuntimeError, match="durable probe"):
        session.probe("null", session.initial)
    assert session.state["pending_probe_result"]["feedback"]["score"] == 0.5
    resumed = Session(routing_case, settings, LocalWorker(), {}, tmp_path, resume=True)
    estimate = resumed.registry.lookup(resumed.checkpoint["task_state"], 0, "null", resumed.initial.identifier)
    assert estimate.scores == [0.5]
    assert len(resumed.ledger.entries) == 1
    assert len(list((tmp_path / "observations").glob("*.json"))) == 1
    again = Session(routing_case, settings, LocalWorker(), {}, tmp_path, resume=True)
    assert again.registry.lookup(again.checkpoint["task_state"], 0, "null", again.initial.identifier).scores == [0.5]


def test_interrupted_pilot_can_retry_without_reusing_orphan_parameters(tmp_path, model_case, settings):
    settings["search"]["max_pilots"] = 2
    owner = BinaryOwner()
    session = Session(model_case, settings, LocalWorker(), {"shared": owner}, tmp_path)
    session.state["pilot_count"] = 1
    session.state["program_status"]["0:owner_update"] = {"status": "running"}
    session.state["raw_costs"]["owner_update"] = 0.0
    session.ledger.reserve_action("training_pilot", 2.0, program="owner_update", version=0)
    session.save()
    resumed = Session(model_case, settings, LocalWorker(), {"shared": BinaryOwner()}, tmp_path, resume=True)
    assert resumed.ledger.spent == 2.0
    assert "owner_update" not in resumed.costs()
    resumed.collect_data()
    assert resumed.materialize_next()
    assert resumed.state["pilot_count"] == 2
    assert resumed.state["program_status"]["0:owner_update"]["status"] == "trained"
    assert "owner_update" in resumed.versions.candidates
    assert resumed.ledger.entries[0]["status"] == "interrupted_charged_upper_bound"


def test_score_source_changes_cannot_enter_a_confidence_sequence(tmp_path, routing_case, settings):
    worker = LocalWorker()
    worker.online_sources.add("second_source")
    with pytest.raises(ValueError, match="score source"):
        Session(routing_case, settings, worker, {}, tmp_path / "ambiguous")
    worker.score_source = "execution_check"
    session = Session(routing_case, settings, worker, {}, tmp_path / "bound")
    original = worker.call
    def changed(method, **arguments):
        result = original(method, **arguments)
        if method == "continuation_score":
            result["source"] = "second_source"
        return result
    worker.call = changed
    with pytest.raises(ValueError, match="Online feedback"):
        session.probe("null", session.initial)
    assert all(not estimate.scores for estimate in session.registry.entries.values())


def test_prior_shared_spend_shrinks_the_certified_graph_family(tmp_path, routing_case, settings):
    routing_case["witnesses"] = []
    routing_case["budget"]["reserve_gpu_seconds"] = 2.0
    session = Session(routing_case, settings, LocalWorker(), {}, tmp_path)
    assert any(session.graph.cost(pair["graph"]) == 3 for pair in session.search.bounds(session.costs())["pairs"])
    session.ledger.charge("discarded_prior_work", 10.0)
    result = session.search.bounds(session.costs())
    assert session.search.operational_execution_budget == 2.0
    assert all(session.graph.cost(pair["graph"]) <= 2 for pair in result["pairs"])
    assert result["frontier_gap"] is None


def test_empty_evaluated_frontier_is_unavailable(tmp_path, routing_case, settings):
    routing_case["search"]["max_probes"] = 0
    report = Session(routing_case, settings, LocalWorker(), {}, tmp_path).run()
    assert report["evaluated_frontier_gap"] is None
    assert report["decision_gap"] > 0


def test_native_metrics_and_paired_seed_bootstrap():
    assert native_value({"value": 0.75, "unit": "rate"}) == 75
    assert native_value({"value": 7, "unit": "count"}) == 7
    with pytest.raises(ValueError):
        native_value({"value": 7.5, "unit": "count"})
    first = {(seed, task): float(task + 2) for seed in range(5) for task in range(3)}
    second = {identity: value - 2 for identity, value in first.items()}
    result = paired_interval(first, second, draws=200)
    assert result["lower"] == pytest.approx(2)
    assert result["upper"] == pytest.approx(2)
    with pytest.raises(ValueError):
        paired_interval(first, dict(list(second.items())[1:]), draws=200)


def test_summary_uses_sample_standard_deviation_and_excludes_fixtures():
    records = [{"stage": "final_evaluation", "backbone": "binary", "benchmark": "toolhop", "revision": "test",
                "method": "method", "seed": seed, "task_id": "task", "data_kind": "benchmark",
                "metrics": {"accuracy": {"value": seed / 4, "unit": "rate"}}} for seed in range(5)]
    result = summarize(records)[0]
    assert result["mean"] == pytest.approx(50)
    assert result["sample_std"] == pytest.approx(39.52847075)
    fixture = copy.deepcopy(records)
    fixture[0]["data_kind"] = "software_fixture"
    with pytest.raises(ValueError):
        summarize(fixture)
    assert supported_citation_pairs([{"statement_id": 1, "source_id": "source", "supported": True},
                                     {"statement_id": 2, "source_id": "source", "supported": True}]) == 2
