import json
import selectors
import subprocess
from pathlib import Path


BENCHMARKS = {
    "marble", "mlgym", "deepresearch", "toolhop", "webshop", "webwalkerqa", "xbench-deepsearch"
}


class NativeWorker:
    """One native environment process, using request-correlated JSON lines."""

    def __init__(self, command, specification, log_path, timeout=3600):
        if not isinstance(command, list) or not command or any(not isinstance(value, str) for value in command):
            raise ValueError("An environment command is an argument list")
        if timeout <= 0:
            raise ValueError("The native environment timeout is positive")
        self.specification = dict(specification)
        if specification.get("data_kind") != "software_fixture" and specification.get("name") not in BENCHMARKS:
            raise ValueError("Select a declared benchmark or an explicitly identified software fixture")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, text=True, bufsize=1)
        except BaseException:
            self.log.close()
            raise
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.timeout, self.sequence = timeout, 0
        try:
            description = self.call("describe")
            if description.get("protocol") != 1 or description.get("final_labels_online") is not False:
                raise ValueError("Native workers use protocol 1 and separate final labels from online feedback")
            for field in ("name", "revision", "data_kind"):
                if description.get(field) != specification.get(field):
                    raise ValueError(f"The native environment has a different {field}")
            self.description = description
            self.online_sources = set(description["online_sources"])
            if not self.online_sources:
                raise ValueError("A worker declares its authorized continuation-score sources")
            self.score_source = specification.get("score_source")
            if self.score_source is None and len(self.online_sources) == 1:
                self.score_source = next(iter(self.online_sources))
            if self.score_source not in self.online_sources:
                raise ValueError("Select one predeclared continuation-score source for the run")
        except BaseException:
            self.close()
            raise

    def call(self, method, **arguments):
        self.sequence += 1
        request = {"id": self.sequence, "method": method, "arguments": arguments}
        self.process.stdin.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(self.timeout):
            raise TimeoutError(f"Native environment timed out during {method}")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Native environment exited during {method}")
        response = json.loads(line)
        if response.get("id") != self.sequence:
            raise ValueError("Native environment response identifiers differ")
        if response.get("error"):
            raise RuntimeError(f"Native environment rejected {method}: {response['error']}")
        if not isinstance(response.get("result"), dict):
            raise ValueError("Native environment results are JSON objects")
        return response["result"]

    def checkpoint(self, task):
        result = self.call("checkpoint", task=task)
        if not result.get("task_state") or "snapshot" not in result or "public_task" not in result:
            raise ValueError("A checkpoint includes an identity, restorable snapshot, and public task input")
        return result

    def restore(self, checkpoint):
        result = self.call("restore", snapshot=checkpoint["snapshot"], task_state=checkpoint["task_state"])
        if result.get("task_state") != checkpoint["task_state"] or result.get("restored") is not True:
            raise ValueError("A continuation requires the same frozen environment checkpoint")

    def close(self):
        if getattr(self, "process", None) is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            self.process.stdin.close()
            self.process.stdout.close()
        if getattr(self, "selector", None) is not None:
            self.selector.close()
        if getattr(self, "log", None) is not None:
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        self.close()
