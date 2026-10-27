import hashlib
import importlib.metadata
import re
from pathlib import Path

from .storage import fingerprint


def local_checkpoint_digest(directory):
    directory = Path(directory)
    selected = sorted(path for path in directory.rglob("*") if path.is_file()
                      and path.suffix.lower() in {".json", ".safetensors", ".bin", ".pt", ".model", ".txt", ".tiktoken", ".py"}
                      and not {".git", ".cache"}.intersection(path.relative_to(directory).parts))
    if not selected:
        raise ValueError("A local model directory contains no checkpoint or tokenizer files")
    digest = hashlib.sha256()
    for path in selected:
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def model_identity(specification, configuration, tokenizer):
    checkpoint = specification["checkpoint"]
    local = Path(checkpoint).is_dir()
    revision = getattr(configuration, "_commit_hash", None) or specification.get("revision")
    if not local and (revision is None or re.fullmatch(r"[a-fA-F0-9]{40}", revision) is None):
        raise ValueError("Remote model versions resolve to an immutable checkpoint commit")
    tokenizer_record = {"vocabulary": tokenizer.get_vocab(), "chat_template": tokenizer.chat_template,
                        "special_tokens": {name: str(value) for name, value in tokenizer.special_tokens_map.items()}}
    libraries = {}
    for name in ("torch", "transformers", "peft", "tokenizers", "sentencepiece", "accelerate", "safetensors"):
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"checkpoint": checkpoint, "revision": revision,
            "local_content_sha256": local_checkpoint_digest(checkpoint) if local else None,
            "tokenizer_sha256": fingerprint(tokenizer_record), "libraries": libraries}
