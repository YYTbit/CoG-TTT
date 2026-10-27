import argparse
import json
from pathlib import Path

from .configuration import read_configuration, validate_case
from .evaluation import summarize
from .native import NativeWorker
from .storage import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="CoG-TTT: Joint Test-Time Training for Evolving Agentic Graphs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--case", required=True)
        command.add_argument("--config", default="configs/paper.yaml")
        if name != "check":
            command.add_argument("--output", required=True)
        if name == "run":
            command.add_argument("--resume", action="store_true")
        if name == "evaluate":
            command.add_argument("--run", required=True)
            command.add_argument("--method", default="CoG-TTT")
            command.add_argument("--backbone", required=True)
    command = subparsers.add_parser("summarize")
    command.add_argument("--records", nargs="+", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--baseline")
    command.add_argument("--mlgym-reference", default="No-TTT")
    command.add_argument("--draws", type=int, default=10000)
    command.add_argument("--bootstrap-seed", type=int, default=0)
    command.add_argument("--allow-partial-seeds", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "summarize":
        records = []
        for filename in arguments.records:
            with open(filename) as stream:
                records.extend(json.loads(line) for line in stream if line.strip())
        result = summarize(records, arguments.baseline, arguments.draws, arguments.bootstrap_seed,
                           not arguments.allow_partial_seeds, arguments.mlgym_reference)
        write_json(arguments.output, result)
        print(json.dumps({"metrics": len(result), "output": arguments.output}))
        return
    case, configuration = read_configuration(arguments.case), read_configuration(arguments.config)
    graph, initial, witnesses, settings, budget = validate_case(case, configuration)
    if arguments.command == "check":
        print(json.dumps({"valid": True, "operations": len(graph.nodes), "routes": len(graph.routes),
                          "initial_execution_cost": graph.cost(initial), "programs": len(case.get("programs", [])),
                          "data_kind": case["benchmark"]["data_kind"]}))
        return
    output = Path(arguments.output)
    log = output / "environment.log" if arguments.command == "run" else output.with_suffix(".log")
    with NativeWorker(case["environment"]["command"], case["benchmark"], log,
                      case["environment"].get("timeout", 3600)) as worker:
        if arguments.command == "run":
            from .models import CausalOwner
            from .runtime import Session
            owners = {name: CausalOwner.from_pretrained(name, specification)
                      for name, specification in case.get("owners", {}).items()}
            report = Session(case, configuration, worker, owners, output, arguments.resume).run()
            print(json.dumps({"parameter_version": report["parameter_version"], "decision_gap": report["decision_gap"],
                              "stopping_reason": report["stopping_reason"], "output": str(output)}))
        else:
            run = read_json(Path(arguments.run) / "result.json")
            if run["benchmark"] != case["benchmark"] or run["task"] != case["task"]:
                raise ValueError("Final evaluation matches the completed task and benchmark release")
            evaluation = worker.call("final_evaluate", task=case["task"], execution=run["execution"])
            record = {"stage": "final_evaluation", "data_kind": case["benchmark"]["data_kind"],
                      "benchmark": case["benchmark"]["name"], "revision": case["benchmark"]["revision"],
                      "task_id": case["task"]["id"], "seed": run["seed"], "method": arguments.method,
                      "backbone": arguments.backbone, "metrics": evaluation["metrics"],
                      "evaluator": evaluation.get("evaluator", worker.description)}
            from .storage import append_json
            append_json(output, record)
            print(json.dumps({"stage": "final_evaluation", "output": str(output)}))


if __name__ == "__main__":
    main()
