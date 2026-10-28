# CoG-TTT

Code for **CoG-TTT: Joint Test-Time Training for Evolving Agentic Graphs**.

CoG-TTT jointly selects a graph-conditioned parameter update and a legal continuation graph under a shared compute budget. Legal continuation contrasts provide downstream-aware training credit. Versioned confidence bounds and a partition-budget dynamic program support conservative selection, acceptance, and stopping.

![CoG-TTT overview](assets/overview.png)

## Installation

Python 3.10 or later is required. Install the model dependencies for language-model execution and training.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[models,test]"
export HF_ENDPOINT=https://hf-mirror.com
```

## Usage

Validate the graph constraints and run the software fixture. This fixture checks the execution protocol; its outputs are excluded from benchmark summaries.

```bash
python -m cog_ttt check --case examples/protocol_check.json
python -m cog_ttt run --case examples/protocol_check.json --output results/protocol-check
python -m pytest -q
```

For a benchmark task, specify its native environment, graph, parameter owners, training programs, and cost units in a case file. The [environment interface](docs/environment.md) defines the required operations and input fields.

```bash
python -m cog_ttt run --case task.json --output results/task
python -m cog_ttt run --case task.json --output results/task --resume
python -m cog_ttt evaluate --case task.json --run results/task --backbone qwen --output results/final.jsonl
python -m cog_ttt summarize --records results/final.jsonl --output results/summary.json
```

Final benchmark evaluation is a separate operation. Online training uses authorized tool feedback and continuation scores from a frozen checkpoint.

## Configuration

[Paper settings](configs/paper.yaml) contain the loss coefficients, failure budget, tolerances, and probe/pilot limits. [Backbone identifiers](configs/backbones.yaml) and the [benchmark protocol](configs/benchmarks.yaml) specify the three models and seven benchmarks. Learning rates, update scope, task graphs, decoding settings, and cost discretization are explicit execution inputs.

The implementation includes legal graph constraints, graph-conditioned preference and verified-target losses, reversible parameter versions, anytime validation, residual certificates, signed interaction models, and partition-budget optimization. Run records retain the remaining decision gap, the evaluated-frontier gap, and costs for every attempted pilot, probe, verification, and execution.

## Resources and release scope

The release contains the method implementation, configurations, environment interface, executable protocol example, and software tests. Native benchmark environments and baseline implementations are obtained from their upstream projects. The benchmark case file connects the installed environment to CoG-TTT through the documented interface.

The [benchmark configuration](configs/benchmarks.yaml) links the upstream resources, and the [backbone configuration](configs/backbones.yaml) lists the public checkpoint identifiers. Select environment and evaluator revisions that match the paper protocol. DeepResearch evaluation uses the specified legacy evaluator and judge models.

Datasets, downloaded weights, runtime checkpoints, and generated results remain in local working directories.
