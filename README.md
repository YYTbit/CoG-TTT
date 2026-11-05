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
quality.

## Algorithm Flow

The joint procedure coordinates candidate generation, certified graph search, and budget-aware execution.

1. Initialize null program, feasible incumbent, and execution reserve.
2. While a legal probe or pilot fits the remaining budget, propose graphs and graph-conditioned training programs. Materialize updates on reversible copies; charge their costs.
3. Validate legal pairs and update anytime score intervals. Build certified graph bounds or range fallbacks. Select the feasible program-graph pair with the largest lower score bound and compute the reported gap.
4. Screen candidates using certified bounds. Stop when the gap is within tolerance and the declared family is complete.
5. If the candidate passes conservative acceptance, accept its version and invalidate affected graph intervals. Otherwise, retain the accepted parameters and probe the next legal witness.
6. Execute a feasible continuation and report the gap and all costs.

![CoG-TTT overview](assets/overview.png)

**Figure 1.** Graph-coupled adaptation during Web misinformation research, illustrated with a DeepResearch Bench question on need for closure and misinformation acceptance. Citation tracing supplies evidence for downstream analysis. Paired continuations guide node training, while joint selection updates the execution graph for subsequent research stages.

## Results

**Table 1.** Results on seven complete public benchmarks. Cells show five-seed mean ± sample standard deviation. Bold marks the best mean within each backbone. Win13 compares MLGym outcomes with No TTT; all columns favor higher values.

![Main results from Table 1](assets/main_results.png)

**CoG-TTT improves performance across model families.** Table 1 reports the highest mean in all 27 metric–backbone combinations. Across eight native metrics, the mean improvements over the strongest baseline in each column are 5.24 on Qwen3-8B, 4.50 on Llama, and 4.86 on Gemma. Their aggregate improvement of 4.87 supports the benefit of joint adaptation across backbone families.

**Web tasks benefit across research and interaction.** On Qwen3-8B, MARBLE KPI increases from 56.1 to 61.4 and RACE from 36.7 to 44.4. ToolHop accuracy rises from 49.4 to 52.4, with a paired 95% interval of [2.1, 3.9] for the gain; WebShop reward rises from 52.6 to 58.7, with an interval of [5.3, 6.9]. These tasks link intermediate evidence or actions to subsequent decisions. Their improvements are consistent with evaluating node updates through the continuations that use their outputs.

## Getting Started

### Environment Setup

Use Python 3.10 or later and install PyTorch for your system.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

### Data Preparation

Prepare `task.json` with the model checkpoint, environment adapter, graph, and
calibration inputs, following the [environment guide](docs/environment.md).
Model profiles and benchmark resources are in [configs/](configs).

### Training and Evaluation

Run joint optimization. The command collects the configured calibration
measurements before search:

```bash
python -m cog_ttt run --case task.json --config configs/paper.yaml \
  --output results/task
```

Evaluate the final execution and summarize the scores:

```bash
python -m cog_ttt evaluate --case task.json --run results/task \
  --backbone qwen --output results/final.jsonl

python -m cog_ttt summarize --records results/final.jsonl \
  --output results/summary.json
```

## License

[MIT License](LICENSE).
