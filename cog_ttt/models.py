import os
import tempfile
import uuid
from pathlib import Path

import numpy as np


class CausalOwner:
    def __init__(self, name, model, tokenizer, settings=None):
        self.name, self.model, self.tokenizer = name, model, tokenizer
        self.settings = dict(settings or {})
        self.model.eval()
        self.latest_usage = {}

    @classmethod
    def from_pretrained(cls, name, specification):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        checkpoint = specification["checkpoint"]
        revision = specification.get("revision")
        configuration = AutoConfig.from_pretrained(checkpoint, revision=revision)
        revision = getattr(configuration, "_commit_hash", None) or revision
        model_class = AutoModelForImageTextToText if configuration.model_type == "gemma3" else AutoModelForCausalLM
        model = model_class.from_pretrained(checkpoint, revision=revision, config=configuration,
                                            device_map=specification.get("device_map", "auto"),
                                            torch_dtype=getattr(torch, specification.get("dtype", "bfloat16")))
        adaptation = specification.get("adaptation", {"kind": "full"})
        if adaptation["kind"] == "lora":
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(r=int(adaptation["rank"]), lora_alpha=adaptation["scale"],
                                                     target_modules=adaptation["target_modules"],
                                                     lora_dropout=0.0, task_type="CAUSAL_LM"))
        elif adaptation["kind"] == "full":
            model.requires_grad_(True)
        else:
            raise ValueError("Parameter owners declare full-parameter or LoRA updates")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        from .identity import model_identity
        owner = cls(name, model, tokenizer, specification)
        owner.identity = model_identity(specification, configuration, tokenizer)
        return owner

    def parameters(self):
        return self.model.parameters()

    def snapshot(self):
        return {name: parameter.detach().cpu().clone() for name, parameter in self.model.named_parameters()
                if parameter.requires_grad}

    def restore(self, state):
        import torch
        parameters = {name: parameter for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        if set(state) != set(parameters):
            raise ValueError("Parameter snapshots differ from the declared owner scope")
        with torch.no_grad():
            for name, parameter in parameters.items():
                if tuple(state[name].shape) != tuple(parameter.shape):
                    raise ValueError("A parameter version has an incompatible shape")
                parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))

    def encode(self, history, requirement=None):
        messages = [{"role": "user", "content": history}] if isinstance(history, str) else [dict(row) for row in history]
        if not messages or any(row.get("role") not in {"system", "user", "assistant", "tool"}
                               or not isinstance(row.get("content"), str) for row in messages):
            raise ValueError("Node histories contain textual chat messages")
        if requirement:
            if messages[-1]["role"] != "user":
                raise ValueError("Verified input histories end with a user request")
            messages[-1]["content"] += "\n\nVerified output requirement\n" + requirement
        if getattr(self.model.config, "model_type", "") in {"gemma3", "gemma3_text"} and messages[0]["role"] == "system":
            system = messages.pop(0)["content"]
            if not messages or messages[0]["role"] != "user":
                raise ValueError("Gemma histories contain a user request after the system instruction")
            messages[0]["content"] = system + "\n\n" + messages[0]["content"]
        identifiers = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                          enable_thinking=self.settings.get("enable_thinking", False))
        if not identifiers or len(identifiers) > self.settings.get("context_tokens", 32768):
            raise ValueError("The complete node input exceeds the configured context window")
        return tuple(identifiers)

    def log_probability(self, history, target, requirement=None):
        import torch
        prompt = self.encode(history, requirement)
        generated = tuple(target["token_ids"]) if isinstance(target, dict) else tuple(self.tokenizer.encode(target, add_special_tokens=False))
        if not generated or len(prompt) + len(generated) > self.settings.get("context_tokens", 32768):
            raise ValueError("A complete training sequence exceeds the declared context window")
        device = self.model.get_input_embeddings().weight.device
        identifiers = torch.tensor([prompt + generated], dtype=torch.long, device=device)
        output = self.model(input_ids=identifiers, attention_mask=torch.ones_like(identifiers), use_cache=False)
        logits = output.logits[0, len(prompt) - 1:len(prompt) + len(generated) - 1].float()
        labels = torch.tensor(generated, dtype=torch.long, device=logits.device)
        return torch.log_softmax(logits, dim=-1).gather(1, labels[:, None]).sum()

    def sample(self, history, seed=0, max_tokens=None, temperature=1.0):
        import time
        import torch
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("Node generation uses a finite nonnegative temperature")
        prompt = self.encode(history)
        requested = self.settings.get("max_output_tokens", 256) if max_tokens is None else max_tokens
        if isinstance(requested, bool) or int(requested) != requested:
            raise ValueError("Generation limits are positive integer token counts")
        cap = int(requested)
        if cap < 1 or len(prompt) + cap > self.settings.get("context_tokens", 32768):
            raise ValueError("The input and output budget exceed the declared node context")
        device = self.model.get_input_embeddings().weight.device
        identifiers = torch.tensor([prompt], dtype=torch.long, device=device)
        devices = sorted({parameter.device.index for parameter in self.model.parameters() if parameter.device.type == "cuda"})
        for selected in devices:
            torch.cuda.synchronize(selected)
        started = time.perf_counter()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(int(seed))
            arguments = {"do_sample": temperature > 0, "max_new_tokens": cap, "use_cache": True,
                         "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id}
            if temperature > 0:
                arguments.update(temperature=temperature, top_p=1.0, top_k=0)
            output = self.model.generate(input_ids=identifiers, attention_mask=torch.ones_like(identifiers), **arguments)
        for selected in devices:
            torch.cuda.synchronize(selected)
        elapsed = time.perf_counter() - started
        generated = tuple(output[0, len(prompt):].tolist())
        if not generated:
            raise ValueError("Node generation returned an empty token sequence")
        self.latest_usage = {"input_tokens": len(prompt), "output_tokens": len(generated),
                             "elapsed_seconds": elapsed, "gpu_seconds": elapsed * len(devices) if devices else None,
                             "meter": "synchronized_elapsed_times_participating_cuda_devices" if devices else "cpu_wall_time"}
        return {"text": self.tokenizer.decode(generated, skip_special_tokens=True), "token_ids": generated,
                "usage": self.latest_usage}

    def save_state(self, path, state):
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".parameters-")
        os.close(handle)
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_state(self, path):
        import torch
        return torch.load(path, map_location="cpu", weights_only=True)


class VersionStore:
    def __init__(self, owners):
        self.owners = dict(owners)
        self.version = 0
        self.accepted = {name: owner.snapshot() for name, owner in self.owners.items()}
        self.accepted_identities = {name: uuid.uuid4().hex for name in self.owners}
        self.candidates = {}

    def add(self, identifier, owner, state, base_version):
        if owner not in self.owners or base_version != self.version or identifier == "null" or identifier in self.candidates:
            raise ValueError("Candidate parameters belong to a unique owner and the current accepted version")
        self.candidates[identifier] = {"owner": owner, "state": state, "base_version": base_version,
                                      "state_identity": uuid.uuid4().hex}

    def activate(self, identifier="null"):
        for name, owner in self.owners.items():
            state = self.accepted[name]
            if identifier != "null":
                if identifier not in self.candidates or self.candidates[identifier]["base_version"] != self.version:
                    raise ValueError("A candidate parameter version is unavailable at this checkpoint")
                if self.candidates[identifier]["owner"] == name:
                    state = self.candidates[identifier]["state"]
            owner.restore(state)

    def accept(self, identifier):
        if identifier == "null":
            return None
        candidate = self.candidates[identifier]
        if candidate["base_version"] != self.version:
            raise ValueError("A stale parameter candidate cannot be accepted")
        owner = candidate["owner"]
        self.accepted[owner] = candidate["state"]
        self.accepted_identities[owner] = candidate["state_identity"]
        self.version += 1
        self.candidates.clear()
        self.activate()
        return owner

    def save(self, directory):
        from .storage import write_json
        directory = Path(directory)
        record = {"version": self.version, "accepted": {}, "candidates": {}}
        for index, (name, owner) in enumerate(sorted(self.owners.items())):
            path = f"accepted_{index}_{self.accepted_identities[name]}.pt"
            if not (directory / path).exists():
                owner.save_state(directory / path, self.accepted[name])
            record["accepted"][name] = {"path": path, "identity": self.accepted_identities[name]}
        for index, (identifier, candidate) in enumerate(sorted(self.candidates.items())):
            path = f"candidate_{self.version}_{candidate['state_identity']}.pt"
            if not (directory / path).exists():
                self.owners[candidate["owner"]].save_state(directory / path, candidate["state"])
            record["candidates"][identifier] = {"owner": candidate["owner"], "state": path,
                                                "base_version": candidate["base_version"],
                                                "state_identity": candidate["state_identity"]}
        write_json(directory / "versions.json", record)
        return record

    def restore(self, directory, record=None):
        from .storage import fingerprint, read_json
        directory = Path(directory)
        record = read_json(directory / "versions.json") if record is None else record
        if set(record["accepted"]) != set(self.owners):
            raise ValueError("Checkpoint parameter owners differ from the active models")
        if isinstance(record["version"], bool) or int(record["version"]) != record["version"] or record["version"] < 0:
            raise ValueError("Accepted parameter versions are nonnegative integers")
        self.version = int(record["version"])
        self.accepted = {name: self.owners[name].load_state(directory / (row if isinstance(row, str) else row["path"]))
                         for name, row in record["accepted"].items()}
        self.accepted_identities = {name: (fingerprint(row) if isinstance(row, str) else row["identity"])
                                    for name, row in record["accepted"].items()}
        self.candidates = {identifier: {**value, "state": self.owners[value["owner"]].load_state(directory / value["state"])}
                           for identifier, value in record["candidates"].items()}
        for candidate in self.candidates.values():
            candidate.setdefault("state_identity", uuid.uuid4().hex)
        if any(value["base_version"] != self.version for value in self.candidates.values()):
            raise ValueError("Resumed candidates belong to the accepted parameter version")
        self.activate()
