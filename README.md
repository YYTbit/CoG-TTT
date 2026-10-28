# CoG-TTT: Joint Test-Time Training for Evolving Agentic Graphs

Long-horizon Web research agents coordinate search, evidence verification, and
tool use through evolving collaboration graphs. Their executions provide
feedback for test-time training (TTT), yet unreliable training signals and
uncertain downstream gains make node-level adaptation challenging. In
particular, the value of a parameter update depends on how downstream nodes use
its outputs, while improved node capabilities can reshape the optimal
collaboration graph. Existing TTT methods focus on model adaptation, yet provide
limited guidance for these coupled decisions. To address this challenge, we
propose CoG-TTT, a graph-coupled method that jointly optimizes node-level
training and evolving agentic graphs. Specifically, we compare feasible
execution continuations to construct confidence-aware training signals, perform
graph-conditioned parameter updates, and coordinate training and graph selection
through witness-guided optimization. We further derive conditional bounds on
joint decision quality and solve a graph-bound allocation subproblem exactly
using dynamic programming, providing theoretical guidance for resource
allocation. We evaluate CoG-TTT on seven public benchmarks using Qwen3-8B,
Llama-3.1-8B-Instruct, and Gemma-3-12B-it. Across all three backbones, CoG-TTT
improves over the strongest baseline for each of the eight native metrics by
4.87 on average. It also improves upon matched alternating optimization across
the three backbones and reduces legal-probe costs by 52% at matched decision
quality. Code is available at <https://anonymous.4open.science/r/CoG-TTT-1F28/>.

![CoG-TTT overview](assets/overview.png)

## Installation

Use Python 3.10 or later and a PyTorch build appropriate for the execution
hardware.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models,test]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

## Quick start

The CPU software fixture exercises legal graph execution and persistent run
state.

```bash
python -m cog_ttt check --case examples/protocol_check.json
python -m cog_ttt run --case examples/protocol_check.json \
  --output results/protocol-check
```

The selected execution and decision report are written to `result.json`.

## Task preparation

Choose a profile from [configs/backbones.yaml](configs/backbones.yaml), pin its
model revision, and place it under `owners` in the task case. Complete the
provider's access requirements for gated checkpoints.

| Backbone | Checkpoint |
| --- | --- |
| Qwen3-8B | `Qwen/Qwen3-8B` |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Gemma-3-12B-it | `google/gemma-3-12b-it` |

Owners use full-parameter updates by default. Optional LoRA specifications are
documented in the [environment reference](docs/environment.md).

Install the chosen benchmark from the upstream resources in
[configs/benchmarks.yaml](configs/benchmarks.yaml). Supply a Python factory
implementing task checkpointing, prompts, tools, authorized continuation scores,
and independent final evaluation.

```bash
python -m cog_ttt.worker --factory task_adapter:create --config environment.json
```

`task_adapter:create` and `environment.json` are supplied by the task
integration. Set `environment.command` to the worker's argument list. The runner
starts the process.

Prepare `task.json` using the [case schema](docs/environment.md). It declares
benchmark and model revisions, public task input, legal graphs, training
programs, and budgets. Integer graph costs have explicit GPU-second and utility
conversions; the execution reserve covers supplied witnesses.

## Calibration

Declare the locked development graphs, repetitions, and a valid structural
residual bound in `calibration`. Legal contrasts identify the signed features.
Calibration is refreshed for each parameter version and consumes the shared
probe and compute budgets.

Run the included calibration software fixture and continue from its saved state.

```bash
python -m cog_ttt calibrate --case examples/calibration_check.json \
  --output results/calibration-check

python -m cog_ttt run --case examples/calibration_check.json \
  --output results/calibration-check --resume
```

`run` performs this stage automatically for cases with a calibration plan. The
reference specifies feature identifiability and residual assumptions.

## Joint optimization

Validate the configured case and start the method.

```bash
python -m cog_ttt check --case task.json --config configs/paper.yaml
python -m cog_ttt run --case task.json --config configs/paper.yaml \
  --output results/task
```

Resume with the same case, configuration, model checkpoints, and output
directory.

```bash
python -m cog_ttt run --case task.json --config configs/paper.yaml \
  --output results/task --resume
```

Run state retains parameter versions, confidence observations, and charged work.
The final report includes the executable incumbent, decision gap, and stopping
reason.

## Evaluation

Score the completed execution through the native final evaluator.

```bash
python -m cog_ttt evaluate --case task.json --run results/task \
  --backbone qwen --output results/final.jsonl

python -m cog_ttt summarize --records results/final.jsonl \
  --output results/summary.json
```

Collect complete task records over seeds 0–4. Summaries report means and sample
standard deviations; MLGym requires matching No-TTT records. Paired comparisons
use matching task identities and native units.

```bash
python -m cog_ttt summarize \
  --records results/final.jsonl results/baseline.jsonl \
  --baseline No-TTT --draws 10000 --output results/paired-summary.json
```

## Configuration

| File | Contents |
| --- | --- |
| [configs/paper.yaml](configs/paper.yaml) | Loss, confidence, search, and budget defaults |
| [configs/backbones.yaml](configs/backbones.yaml) | Model checkpoint and execution specifications |
| [configs/benchmarks.yaml](configs/benchmarks.yaml) | Benchmark resources and native evaluation protocols |

Task-specific objectives, budgets, and search overrides belong in the case file.
Method settings are selected through `--config`.

## Software checks

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

The CPU suite checks graph bounds, training gradients, calibration, version
restoration, and evaluation accounting.

## License

[MIT License](LICENSE).
