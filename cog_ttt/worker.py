import argparse
import importlib
import json
import sys


METHODS = {"describe", "checkpoint", "restore", "prepare_node", "tool", "continuation_score",
           "verified_targets", "final_evaluate"}


def serve(environment):
    for line in sys.stdin:
        identifier = None
        try:
            request = json.loads(line)
            identifier = request["id"]
            if request["method"] not in METHODS:
                raise ValueError("Unknown native environment operation")
            result = getattr(environment, request["method"])(**request["arguments"])
            response = {"id": identifier, "result": result}
        except Exception as error:
            response = {"id": identifier, "error": {"type": type(error).__name__, "message": str(error)}}
        sys.stdout.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Serve a native benchmark environment over JSON lines")
    parser.add_argument("--factory", required=True, help="Installed module:function returning a native environment")
    parser.add_argument("--config", help="JSON configuration passed to the factory")
    arguments = parser.parse_args()
    module, function = arguments.factory.split(":", 1)
    configuration = {}
    if arguments.config:
        from .storage import read_json
        configuration = read_json(arguments.config)
    environment = getattr(importlib.import_module(module), function)(configuration)
    serve(environment)


if __name__ == "__main__":
    main()
