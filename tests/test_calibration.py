import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from cog_ttt.calibration import apply_calibration, calibration_plan, fit_calibration, validate_calibration
from cog_ttt.confidence import Estimate, anytime_radius
from cog_ttt.configuration import validate_case
from cog_ttt.graph import Graph
from cog_ttt.oracle import graph_upper_bound
from cog_ttt.runtime import Session
from cog_ttt.storage import read_json, write_json
from cog_ttt.surrogate import SignedModel
from conftest import BinaryOwner, LocalWorker, ROOT


def calibration_case():
    return read_json(ROOT / "examples/calibration_check.json")


def interaction_calibration():
    graph = Graph(nodes=[{"name": name, "cost": 1, "operation": "constant", "kind": "tool"}
                         for name in ("a", "b")])
    plan = calibration_plan(graph, {"graphs": [[], ["a"], ["b"], ["a", "b"]],
                                   "repetitions": 40, "residual_bound": 0.02}, None, 2)
    rows = [(decision, Estimate(index + 1, [0.2 + 0.1 * decision.bits[0] + 0.2 * decision.bits[1]
                                          - 0.05 * decision.bits[0] * decision.bits[1]] * 40))
            for index, decision in enumerate(plan["decisions"])]
    scope = {"task_state": "state", "parameter_version": 0, "program": "null",
             "score_source": "execution_check", "parameter_identity": "fixture"}
    return graph, plan, rows, scope


def test_legal_contrast_calibration_recovers_signed_effects_and_shared_anchor_radius():
    graph, plan, rows, scope = interaction_calibration()
    record = fit_calibration(graph, plan, rows, 0.05, scope)
    assert record["bias"] == pytest.approx(0.2)
    assert record["coefficients"] == pytest.approx([0.1, 0.2, -0.05])
    mean_radii = [anytime_radius(0.05, estimate.identifier, 40) for _, estimate in rows]
    assert record["radii"][0] == pytest.approx(sum(mean_radii[:2]) + 2 * 0.02)
    assert record["radii"][2] == pytest.approx(sum(mean_radii) + 4 * 0.02)
    assert record["coefficient_upper"] == pytest.approx(np.array(record["coefficients"]) + record["radii"])
    validate_calibration(record, graph, plan, 0.05, scope)


def test_development_design_rejects_aliases_and_illegal_graphs():
    graph, plan, rows, scope = interaction_calibration()
    with pytest.raises(ValueError, match="independent feature set"):
        calibration_plan(graph, {"graphs": [[], ["a"], ["b"]],
                                "repetitions": 1, "residual_bound": 0.0}, None, 2)
    with pytest.raises(ValueError, match="distinct"):
        calibration_plan(graph, {"graphs": [[], []], "repetitions": 1, "residual_bound": 0.0}, [], 2)
    with pytest.raises(ValueError, match="execution exceeds"):
        calibration_plan(graph, {"graphs": [["a", "b"]], "repetitions": 1, "residual_bound": 0.0}, [], 1)
    with pytest.raises(ValueError, match="structural residual bound"):
        calibration_plan(graph, {"graphs": [[]], "repetitions": 1, "residual_bound": float("inf")}, [], 2)


def test_calibration_can_use_a_more_expensive_anchor_and_a_constant_model():
    graph, plan, rows, scope = interaction_calibration()
    plan = calibration_plan(graph, {"graphs": [["a", "b"], ["a"], ["b"], []],
                                   "repetitions": 40, "residual_bound": 0.02}, None, 2)
    observed = {decision.identifier: (decision, estimate) for decision, estimate in rows}
    record = fit_calibration(graph, plan, [observed[decision.identifier] for decision in plan["decisions"]], 0.05, scope)
    assert record["coefficients"] == pytest.approx([0.1, 0.2, -0.05])
    constant = calibration_plan(graph, {"graphs": [[]], "repetitions": 1, "residual_bound": 0.0}, [], 2)
    record = fit_calibration(graph, constant, [(graph.decision([]), Estimate(1, [0.3]))], 0.05, scope)
    assert record["bias"] == pytest.approx(0.3)
    assert record["coefficients"] == record["radii"] == []


def test_calibration_freezes_its_declared_observation_prefix_and_checks_scope():
    graph, plan, rows, scope = interaction_calibration()
    first = fit_calibration(graph, plan, rows, 0.05, scope)
    for _, estimate in rows:
        estimate.add(1.0)
    assert fit_calibration(graph, plan, rows, 0.05, scope) == first
    for field, value in [("task_state", "other"), ("parameter_version", 1), ("program", "candidate"),
                         ("score_source", "other_source"), ("parameter_identity", "other_weights")]:
        with pytest.raises(ValueError, match="parameter version"):
            validate_calibration(first, graph, plan, 0.05, {**scope, field: value})


def test_recentered_intervals_cover_the_locked_development_interval():
    graph, plan, rows, scope = interaction_calibration()
    record = fit_calibration(graph, plan, rows, 0.05, scope)
    model = SignedModel(graph, bias=0.3, coefficients=[0.3, -0.1, 0.2])
    apply_calibration(model, record)
    original = np.array(record["coefficients"])
    radii = np.array(record["radii"])
    assert np.all(model.coefficients - model.radii <= original - radii + 1e-12)
    assert np.all(model.coefficients + model.radii >= original + radii - 1e-12)
    assert np.isfinite(model.witness_width(plan["decisions"][0], plan["decisions"][-1]))


@pytest.mark.parametrize("seed", range(4))
def test_calibrated_partition_upper_bounds_dominate_the_feasible_model(seed):
    graph, plan, rows, scope = interaction_calibration()
    record = fit_calibration(graph, plan, rows, 0.05, scope)
    generator = np.random.default_rng(seed)
    model = apply_calibration(SignedModel(graph, coefficients=generator.normal(size=3)), record)
    original = model.coefficients.copy()
    result = graph_upper_bound(graph, model, {"allowance": 5.0}, 2, execution_price=0.03)
    expected = max(model.predict(decision) + 5.0 - 0.03 * graph.cost(decision) for decision in graph.enumerate(2))
    assert result["dp_value"] >= expected - 1e-9
    assert result["cross_partition_coefficients"] == "calibrated_upper_bounds"
    assert np.array_equal(model.coefficients, original)


def test_development_probes_have_separate_identifiers_and_remain_in_the_budget(tmp_path, settings):
    case, worker = calibration_case(), LocalWorker()
    original = worker.call
    def costed_call(method, **arguments):
        result = original(method, **arguments)
        if method in {"tool", "continuation_score"}:
            result["external_gpu_seconds"] = 0.1
        return result
    worker.call = costed_call
    session = Session(case, settings, worker, {}, tmp_path)
    session.ensure_calibration()
    assert session.state["probe_count"] == 4
    assert session.ledger.spent == pytest.approx(1.4)
    assert all(entry["stage"] == "calibration_probe" for entry in session.ledger.entries)
    for decision in session.calibration_plan["decisions"]:
        assert session.registry.lookup(session.checkpoint["task_state"], 0, "null", decision.identifier) is None
    record = session.search.calibrations["null"]
    assert record["coefficients"] == pytest.approx([0.5])
    assert np.isfinite(record["radii"]).all()
    model, certificate = session.search.model("null", 0.0)
    assert certificate is None
    assert np.isfinite(model.radii).all()
    count, spent = session.state["probe_count"], session.ledger.spent
    resumed = Session(case, settings, worker, {}, tmp_path, resume=True)
    resumed.ensure_calibration()
    assert resumed.state["probe_count"] == count
    assert resumed.ledger.spent == spent
    assert resumed.search.calibrations["null"] == record
    resumed.probe("null", resumed.initial)
    assert resumed.registry.lookup(resumed.checkpoint["task_state"], 0, "null", resumed.initial.identifier).identifier > 2


def test_candidate_calibration_is_bound_to_its_parameter_snapshot(tmp_path, model_case, settings):
    case = copy.deepcopy(model_case)
    case["calibration"] = {"graphs": [["response"]], "repetitions": 1, "residual_bound": 0.0}
    settings["search"].update(max_probes=10, max_pilots=1)
    owner = BinaryOwner()
    session = Session(case, settings, LocalWorker(), {"shared": owner}, tmp_path)
    session.ensure_calibration()
    session.collect_data()
    assert session.materialize_next()
    session.ensure_calibration()
    assert set(session.search.calibrations) == {"null", "owner_update"}
    null_scope = session.search.calibrations["null"]["scope"]
    candidate_scope = session.search.calibrations["owner_update"]["scope"]
    assert null_scope["parameter_identity"] != candidate_scope["parameter_identity"]
    assert float(owner.logit.detach()) == 0
    old_ids = [row["identifier"] for row in session.search.calibrations["owner_update"]["observations"]]
    session.versions.accept("owner_update")
    session.state["raw_costs"] = {"null": 0.0}
    session.search.version = session.versions.version
    session.ensure_calibration()
    assert session.search.calibrations["null"]["scope"]["parameter_version"] == 1
    assert min(row["identifier"] for row in session.search.calibrations["null"]["observations"]) > max(old_ids)


def test_resume_rejects_a_stale_calibration_record(tmp_path, settings):
    case = calibration_case()
    session = Session(case, settings, LocalWorker(), {}, tmp_path)
    session.ensure_calibration()
    state = read_json(tmp_path / "state.json")
    state["calibrations"]["0:null"]["scope"]["parameter_version"] = 1
    write_json(tmp_path / "state.json", state)
    with pytest.raises(ValueError, match="parameter version"):
        Session(case, settings, LocalWorker(), {}, tmp_path, resume=True)


def test_interrupted_calibration_reuses_committed_observations(tmp_path, settings, monkeypatch):
    case = calibration_case()
    session = Session(case, settings, LocalWorker(), {}, tmp_path)
    def interrupt_commit():
        raise RuntimeError("Interrupted development observation commit")
    monkeypatch.setattr(session, "commit_probe", interrupt_commit)
    with pytest.raises(RuntimeError, match="development observation"):
        session.ensure_calibration()
    resumed = Session(case, settings, LocalWorker(), {}, tmp_path, resume=True)
    resumed.ensure_calibration()
    assert resumed.state["probe_count"] == 4
    assert len(resumed.ledger.entries) == 4
    assert [row["count"] for row in resumed.search.calibrations["null"]["observations"]] == [2, 2]


def test_automatic_calibration_honors_the_probe_limit(tmp_path, settings):
    case = calibration_case()
    case["search"]["max_probes"] = 4
    report = Session(case, settings, LocalWorker(), {}, tmp_path).run()
    assert report["probe_count"] == 4
    assert report["stopping_reason"] == "probe_limit"
    assert report["ledger"]["entries"][-1]["stage"] == "final_execution"


def test_calibration_limit_and_endpoint_route_partitions(settings):
    case = calibration_case()
    case["search"]["max_probes"] = 3
    with pytest.raises(ValueError, match="probe limit"):
        validate_case(case, settings)
    graph = Graph(nodes=[
        {"name": name, "cost": 0, "operation": "tool", "kind": "tool", "endpoint": endpoint, "route_rule": rule}
        for name, endpoint, rule in [("a", "report", "citation"), ("b", "report", "citation"),
                                     ("c", "report", "navigation"), ("d", "search", "citation")]])
    assert {frozenset(group) for group in graph.partitions()} == {frozenset(["a", "b"]), frozenset(["c"]), frozenset(["d"])}


def test_calibration_cli_then_joint_search_resume(tmp_path):
    arguments = [sys.executable, "-m", "cog_ttt"]
    case = "examples/calibration_check.json"
    calibration = subprocess.run(arguments + ["calibrate", "--case", case, "--output", str(tmp_path)],
                                 text=True, capture_output=True, check=True, cwd=ROOT)
    assert json.loads(calibration.stdout)["calibrated_programs"] == 1
    run = subprocess.run(arguments + ["run", "--case", case, "--output", str(tmp_path), "--resume"],
                         text=True, capture_output=True, check=True, cwd=ROOT)
    report = read_json(tmp_path / "result.json")
    assert report["calibration"]["null"]["features"] == 1
    assert report["benchmark"]["data_kind"] == "software_fixture"
    assert report["probe_count"] <= 8
    assert json.loads(run.stdout)["parameter_version"] == 0
