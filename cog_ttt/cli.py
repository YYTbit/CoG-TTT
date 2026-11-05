import argparse
import json
from pathlib import Path

from .configuration import read_configuration
from .evaluation import summarize
from .storage import write_json


def main():
    parser = argparse.ArgumentParser(description="CoG-TTT: Joint Test-Time Training for Evolving Agentic Graphs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "calibrate", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--case", required=name != "evaluate")
        command.add_argument("--config", default="configs/paper.yaml")
        command.add_argument("--backbone", choices=("qwen", "llama", "gemma"))
        command.add_argument("--backbones", default="configs/backbones.yaml")
        command.add_argument("--output", required=True)
        if name in ("run", "calibrate"):
            command.add_argument("--seed", type=int)
            command.add_argument("--resume", action="store_true")
        else:
            command.add_argument("--run", required=True)
            command.add_argument("--method", default="CoG-TTT")
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
    from .workflows import evaluate_run, run_case

    case = read_configuration(arguments.case) if arguments.case else None
    if arguments.command == "evaluate":
        manifest_exists = (Path(arguments.run) / "run.json").is_file()
        configuration = None if manifest_exists else read_configuration(arguments.config)
        backbones = {} if manifest_exists else read_configuration(arguments.backbones)
        evaluate_run(arguments.run, arguments.output, case=case, configuration=configuration,
                     backbone=arguments.backbone, backbones=backbones, method=arguments.method)
        print(json.dumps({"stage": "final_evaluation", "output": arguments.output}))
        return
    report = run_case(case, read_configuration(arguments.config), arguments.output,
                      backbone=arguments.backbone, backbones=read_configuration(arguments.backbones),
                      seed=arguments.seed, resume=arguments.resume,
                      calibration_only=arguments.command == "calibrate")
    fields = ("parameter_version", "calibrated_programs") if arguments.command == "calibrate" else (
        "parameter_version", "decision_gap", "stopping_reason")
    print(json.dumps({**{key: report[key] for key in fields}, "output": arguments.output}))


if __name__ == "__main__":
    main()
