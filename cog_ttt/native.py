import json
import math
import os
import selectors
import subprocess
import time
from pathlib import Path


BENCHMARKS = {
    "marble", "mlgym", "deepresearch", "toolhop", "webshop", "webwalkerqa", "xbench-deepsearch"
}


class NativeWorker:
    """One native environment process, using request-correlated JSON lines."""

    def __init__(self, command, specification, log_path, timeout=3600):
        if not isinstance(command, list) or not command or any(not isinstance(value, str) for value in command):
            raise ValueError("An environment command is an argument list")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("The native environment timeout is positive")
        self.specification = dict(specification)
        if specification.get("data_kind") != "software_fixture" and specification.get("name") not in BENCHMARKS:
            raise ValueError("Select a declared benchmark or an explicitly identified software fixture")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, bufsize=0)
        except BaseException:
            self.log.close()
            raise
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self._pending = bytearray()
        self._failed = False
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
        if self._failed:
            raise RuntimeError("The native environment connection is unavailable")
        self.sequence += 1
        request = {"id": self.sequence, "method": method, "arguments": arguments}
        outgoing = memoryview((json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        deadline = time.monotonic() + self.timeout
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                selector.register(self.process.stdout, selectors.EVENT_READ)
                # Readiness covers bytes, so one deadline spans the request and the complete response.
                while outgoing or b"\n" not in self._pending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Native environment timed out during {method}")
                    events = selector.select(remaining)
                    if not events:
                        raise TimeoutError(f"Native environment timed out during {method}")
                    for key, _ in events:
                        try:
                            if key.fileobj is self.process.stdin:
                                sent = os.write(key.fd, outgoing[:65536])
                                outgoing = outgoing[sent:]
                                if not outgoing:
                                    selector.unregister(self.process.stdin)
                            else:
                                incoming = os.read(key.fd, 65536)
                                if not incoming:
                                    raise RuntimeError(f"Native environment exited during {method}")
                                self._pending.extend(incoming)
                        except BlockingIOError:
                            continue
            line, _, remainder = self._pending.partition(b"\n")
            self._pending = bytearray(remainder)
            response = json.loads(line)
            if not isinstance(response, dict):
                raise ValueError("Native environment responses are JSON objects")
            if type(response.get("id")) is not int or response["id"] != self.sequence:
                raise ValueError("Native environment response identifiers differ")
            if ("error" in response) == ("result" in response):
                raise ValueError("Native environment responses contain either a result or an error")
            if "result" in response and not isinstance(response["result"], dict):
                raise ValueError("Native environment results are JSON objects")
        except TimeoutError:
            self._failed = True
            raise
        except OSError as error:
            self._failed = True
            raise RuntimeError("The native environment connection failed") from error
        except BaseException:
            self._failed = True
            raise
        if "error" in response:
            raise RuntimeError(f"Native environment rejected {method}: {response['error']}")
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
        self._failed = True
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
