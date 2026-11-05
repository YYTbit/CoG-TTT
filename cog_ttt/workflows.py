from pathlib import Path

from .configuration import resolve_case, validate_case
from .native import NativeWorker
from .storage import append_json, fingerprint, json_value, read_json, write_json


def run_case(case, configuration, directory, *, backbone=None, backbones=None,
             seed=None, resume=False, calibration_only=False):
    """Run calibration and joint optimization with one immutable input record."""
    from .models import CausalOwner
    from .runtime import Session

    case, label = resolve_case(case, backbones or {}, backbone, seed)
    validate_case(case, configuration)
    if calibration_only and "calibration" not in case:
        raise ValueError("Calibration requires a development plan in the case file")
    directory = Path(directory)
    manifest_path = directory / "run.json"
    inputs = {"case": case, "configuration": configuration, "backbone": label}
    identity = fingerprint(inputs)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("schema") != 1 or manifest.get("input_identity") != identity:
            raise ValueError("The output directory belongs to a different task, backbone, or configuration")
        if fingerprint(manifest["inputs"]) != identity:
            raise ValueError("The saved run inputs changed")
    else:
        manifest = {"schema": 1, "input_identity": identity, "inputs": inputs}
        write_json(manifest_path, manifest)
    with NativeWorker(case["environment"]["command"], case["benchmark"], directory / "environment.log",
                      case["environment"].get("timeout", 3600)) as worker:
        owners = {}
        for name, specification in sorted(case.get("owners", {}).items()):
            loading = dict(specification)
            saved = manifest.get("models", {}).get(name) if resume else None
            if saved is not None:
                if saved["checkpoint"] != specification["checkpoint"]:
                    raise ValueError("The resumed parameter owner uses its recorded checkpoint")
                if saved.get("local_content_sha256") is None and saved.get("revision"):
                    loading["revision"] = saved["revision"]
            owners[name] = CausalOwner.from_pretrained(name, loading, seed=case.get("seed", 0))
        session = Session(case, configuration, worker, owners, directory, resume)
        model_identities = session.state["models"]
        if "models" in manifest and manifest["models"] != model_identities:
            raise ValueError("The run's resolved model identities changed")
        manifest["models"] = model_identities
        write_json(manifest_path, manifest)
        if calibration_only:
            session.ensure_calibration()
            return {"parameter_version": session.versions.version,
                    "calibrated_programs": len(session.search.calibrations), "output": str(directory)}
        return json_value(session.run())


def evaluate_run(directory, output, *, case=None, configuration=None, backbone=None,
                 backbones=None, method="CoG-TTT"):
    """Evaluate the completed execution using its saved task and model identities."""
    directory = Path(directory)
    report = read_json(directory / "result.json")
    state = read_json(directory / "state.json")
    if state.get("completed") is not True:
        raise ValueError("Final evaluation requires a completed run")
    manifest_path = directory / "run.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        inputs = manifest["inputs"]
        if manifest.get("schema") != 1 or fingerprint(inputs) != manifest.get("input_identity"):
            raise ValueError("The saved run inputs changed")
        if case is not None and (case["task"] != inputs["case"]["task"]
                                  or case["benchmark"] != inputs["case"]["benchmark"]):
            raise ValueError("Final evaluation uses the completed run's task and benchmark")
        case, configuration, label = inputs["case"], inputs["configuration"], inputs["backbone"]
        if manifest.get("models") != state["models"]:
            raise ValueError("Final evaluation uses the completed run's resolved model identities")
    else:
        # Earlier runs retain their original inputs and accepted parameter files.
        if case is None or configuration is None:
            raise ValueError("An earlier run requires its original --case and --config")
        case, label = resolve_case(case, backbones or {})
    if backbone is not None and backbone != label:
        raise ValueError("The evaluation backbone differs from the completed run")
    validate_case(case, configuration)
    identity = fingerprint({"case": case, "configuration": configuration,
                            "models": state["models"], "score_source": state["score_source"]})
    if identity != state["identity"] or identity != report["case_identity"]:
        raise ValueError("The task, configuration, and model identities do not match the completed run")
    if report["benchmark"] != case["benchmark"] or report["task"] != case["task"]:
        raise ValueError("Final evaluation matches the completed task and benchmark release")
    if fingerprint(report["execution"]) != fingerprint(state["final_execution"]):
        raise ValueError("Final evaluation uses the recorded completed execution")
    output = Path(output)
    with NativeWorker(case["environment"]["command"], case["benchmark"], output.with_suffix(".log"),
                      case["environment"].get("timeout", 3600)) as worker:
        evaluation = worker.call("final_evaluate", task=case["task"], execution=report["execution"])
        record = {"stage": "final_evaluation", "data_kind": case["benchmark"]["data_kind"],
                  "benchmark": case["benchmark"]["name"], "revision": case["benchmark"]["revision"],
                  "task_id": case["task"]["id"], "seed": report["seed"], "method": method,
                  "backbone": label, "models": state["models"], "metrics": evaluation["metrics"],
                  "evaluator": evaluation.get("evaluator", worker.description)}
    append_json(output, record)
    return record
