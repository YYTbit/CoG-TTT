# CoG-TTT

Code for **CoG-TTT: Joint Test-Time Training for Evolving Agentic Graphs**.

**CoG-TTT** jointly optimizes node-level training and evolving agentic graphs under a shared compute budget. The value of a parameter update depends on how downstream nodes use its outputs, while improved node capabilities can reshape the optimal collaboration graph.

The method compares feasible execution continuations to construct confidence-aware training signals, performs graph-conditioned parameter updates, and coordinates training and graph selection through witness-guided optimization. Conditional bounds on joint decision quality and an exact dynamic programming solution to the graph-bound allocation subproblem guide resource allocation.

The paper evaluates CoG-TTT on **seven public benchmarks** with three model families, reporting an average improvement of **4.87** over the strongest baseline for each of eight native metrics across all three backbones and a **52%** reduction in legal-probe costs at matched decision quality.

![CoG-TTT overview](assets/overview.png)

## Installation

Requires Python 3.10 or later. Run the following commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[models,test]"
export HF_ENDPOINT=https://hf-mirror.com
```

The `models` extra installs PyTorch, Transformers, Accelerate, and PEFT. Use a PyTorch installation compatible with your CUDA environment for GPU execution.

## Quick Start

Validate and run the included CPU protocol example.

```bash
python -m cog_ttt check --case examples/protocol_check.json
python -m cog_ttt run \
    --case examples/protocol_check.json \
    --output results/protocol-check
```

This software fixture exercises legal graph execution, continuation comparisons, and persistence. The completed execution is recorded in `result.json`.

## Task Preparation

### Backbones

The model specifications in [configs/backbones.yaml](configs/backbones.yaml) use the following checkpoints.

| Backbone | Checkpoint |
| --- | --- |
| Qwen3-8B | `Qwen/Qwen3-8B` |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Gemma-3-12B-IT | `google/gemma-3-12b-it` |

Place the selected specification under `owners` in your task case. For example, an owner entry for Qwen is

```json
{
  "owners": {
    "agent": {
      "checkpoint": "Qwen/Qwen3-8B",
      "dtype": "bfloat16",
      "device_map": "auto"
    }
  }
}
```

Complete the provider's access requirements before loading the checkpoint. Full-parameter updates are the default; LoRA settings and revision fields are described in the [environment interface](docs/environment.md).

### Native Environments

Install the selected benchmark using its upstream resources in [configs/benchmarks.yaml](configs/benchmarks.yaml). Connect it through a Python factory implementing the [native environment interface](docs/environment.md), including task checkpointing, prompts, tools, continuation scores, and final evaluation.

For a user-provided factory `task_adapter:create` and configuration `environment.json`, the worker entry point is

```bash
python -m cog_ttt.worker \
    --factory task_adapter:create \
    --config environment.json
```

Set `environment.command` in the task case to this command as an argument list. The runner starts and manages the worker process.

```json
{
  "environment": {
    "command": ["python", "-m", "cog_ttt.worker", "--factory", "task_adapter:create", "--config", "environment.json"]
  }
}
```

### Task Cases

Prepare `task.json` following the [case schema](docs/environment.md). Specify the benchmark identifier and release, task input and seed, parameter owners, legal graph, candidate training programs, and compute budgets.

Graph costs use integer units with an explicit GPU-second conversion. The execution reserve covers the supplied witnesses. The [protocol example](examples/protocol_check.json) illustrates the case structure.

## Run CoG-TTT

Validate the configured task's graph and budgets, then start joint optimization.

```bash
python -m cog_ttt check \
    --case task.json \
    --config configs/paper.yaml

python -m cog_ttt run \
    --case task.json \
    --config configs/paper.yaml \
    --output results/task
```

Resume an interrupted run with the same case, configuration, model checkpoints, and output directory.

```bash
python -m cog_ttt run \
    --case task.json \
    --config configs/paper.yaml \
    --output results/task \
    --resume
```

Run state retains parameter versions, confidence observations, and budget accounting. The final report includes the selected execution, decision gap, and stopping reason.

## Evaluation

Evaluate a completed execution through the native final evaluator.

```bash
python -m cog_ttt evaluate \
    --case task.json \
    --run results/task \
    --backbone qwen \
    --output results/final.jsonl
```

Each call appends one task record. Collect records over seeds 0–4, then aggregate means and sample standard deviations. MLGym summaries also require matching No-TTT reference records.

```bash
python -m cog_ttt summarize \
    --records results/final.jsonl \
    --output results/summary.json
```

For paired comparisons, provide baseline records with matching task identities, seeds, and metric units. The following command uses 10,000 bootstrap draws.

```bash
python -m cog_ttt summarize \
    --records results/final.jsonl results/baseline.jsonl \
    --baseline No-TTT \
    --draws 10000 \
    --output results/paired-summary.json
```

## Configuration

| File | Contents |
| --- | --- |
| [configs/paper.yaml](configs/paper.yaml) | Loss coefficients, confidence settings, search tolerances, and budget defaults |
| [configs/backbones.yaml](configs/backbones.yaml) | Backbone checkpoint identifiers, precision, and device placement |
| [configs/benchmarks.yaml](configs/benchmarks.yaml) | Benchmark resources, evaluation protocols, and seeds |

Pass method settings through `--config`. Task-specific objectives, budgets, and search overrides belong in the case file.

## Project Structure

```text
cog_ttt/        Core method, runtime, model updates, and evaluation
configs/        Method, backbone, and benchmark settings
docs/           Native environment and case reference
examples/       Executable protocol fixture
tests/          Software checks
```

## Tests

```bash
python -m pytest -q
```

## License

[MIT License](LICENSE).
