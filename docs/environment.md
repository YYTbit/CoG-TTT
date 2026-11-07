# Native environment interface

The runner connects to an installed benchmark environment through JSON lines. Native tools, observations, task resets, and final evaluators remain in that environment. A Python environment object can be served with

```bash
python -m cog_ttt.worker --factory package.module:create --config environment.json
```

The factory accepts a configuration mapping and returns an object implementing the operations below. Each request has `id`, `method`, and `arguments`; each response has the same `id` and either `result` or `error`. Environment diagnostics use stderr.

| Operation | Inputs | Result |
| --- | --- | --- |
| `describe` | None | `protocol: 1`, benchmark `name`, `revision`, `data_kind`, `online_sources`, `final_labels_online: false` |
| `checkpoint` | `task` | Persistent `task_state`, serializable `snapshot`, and `public_task` |
| `restore` | `snapshot`, `task_state` | Matching `task_state` and `restored: true` |
| `prepare_node` | Checkpoint, public input, operation, dependencies, routed inputs, settings | Native prompt as a textual chat-message `history` |
| `tool` | Same node fields and `seed` | Native `output` and `external_gpu_seconds` |
| `continuation_score` | Checkpoint, node outputs, terminals, `seed` | `score` in [0,1], matching `task_state`, declared `source`, and `external_gpu_seconds` |
| `verified_targets` | Checkpoint, graph bits, execution trace | `targets`, each containing `node`, `requirement`, `target`, and `verification`; additional `external_gpu_seconds` |
| `final_evaluate` | Original task and completed execution | Native `metrics` and evaluator release information |

`prepare_node` receives `task_state`, `public_task`, `node`, `operation`, `dependencies`, `routed_inputs`, and `settings`. Each routed input records its route, port, source, and output. The selected graph is executed in topological order; cyclic workflows require an explicit finite unrolling. Restores reproduce the same authorized state. Changes in external state require a new checkpoint identity. Repeated continuation observations assume a constant conditional mean at that frozen state.

A case with several available sources selects one `benchmark.score_source` before validation. That source remains constant throughout the run and resume. Model identities retain resolved immutable checkpoint revisions, tokenizer fingerprints, library versions, and content hashes for local checkpoints.

Environment request deadlines include transmission and the complete JSON response.
A timeout invalidates the connection; subsequent requests require a new worker.

A verified target carries `passed: true`, a native `tool_result_id`, the matching `task_state`, and `source` equal to `tool`, `execution_check`, or `api_validation`. Targets originate from permitted execution checks. Independent final answers and judge labels remain in `final_evaluate`. The online runner never calls that operation.

## Case files

Benchmark cases use `data_kind: benchmark`, a benchmark identifier from [the protocol](../configs/benchmarks.yaml), and a pinned release in `revision`. Configure the environment command, task, and method inputs as follows.

- `owners` maps owner identifiers to model specifications. `--backbone qwen`, `--backbone llama`, and `--backbone gemma` select the corresponding checkpoint from [the backbone configuration](../configs/backbones.yaml) for every model owner. Owner-specific dtype, placement, and adaptation settings are retained; a pinned revision belongs to the selected checkpoint. Cases without this flag specify each owner's `checkpoint` directly. Models default to full-parameter updates. An optional `adaptation` mapping accepts `kind: lora` with explicit `rank`, `scale`, and `target_modules`. Shared operations refer to the same owner. Gemma's named language projections are selected within its language model.
- `graph` declares nodes, eligible routes, dependencies, required input ports, exclusive route groups, terminals, and optional linear constraints. Each operation names a native tool or model role. Costs are nonnegative integers in the declared cost quantum. Shared owners form ownership partitions. Tool nodes can declare `endpoint` and `route_rule` for task-based grouping, or an explicit `partition` label. Routing bits inherit their producer's group.
- `initial_graph` is a legal reference continuation. `witnesses` supplies further legal graphs. Exact enumeration is enabled within `max_graph_bits`; larger families use feasible witnesses and certified partition bounds.
- `programs` contains unique `id`, `owner`, active-bit `graph`, positive `steps`, and `learning_rate`. Every candidate begins at the accepted owner state. Tool-verified targets are collected through the native environment.
- `credit_requests` optionally names a model `node`, its active-bit `graph`, an optional `alternative_graph`, and positive `repetitions`. Alternatives are generated at the same node input and injected into restored legal continuations. An alternative graph preserves that input. Prefix generation seeds remain constant across repetitions; downstream seeds vary.
- `objective` specifies integer `total_budget` and `execution_budget`, positive `cost_quantum_gpu_seconds`, and positive `utility_per_cost_unit`. The latter converts integer cost units into utility penalties; configured lambda values are multiplied by this conversion. Learning rates, cost conversion, prompts, decoding, adapter scope, and profile-enumeration limits are execution choices.
- `budget` specifies `reserve_gpu_seconds`, `pilot_gpu_seconds`, and `verification_gpu_seconds`. The default total cap is 25,600 GPU-seconds. The reserve covers supplied execution witnesses. Optional `search` overrides are recorded with the case.

The case's `seed` controls model initialization and continuation sampling; `--seed` overrides it for a run. An owner can declare its own initialization `seed`. `run.json` stores the resolved task, model selection, method configuration, and immutable model identities. Resuming uses the same inputs and `--resume`, loading the recorded remote model revision. Local checkpoint contents remain subject to identity verification. Final evaluation reads this record and the completed execution, without loading model weights again. An optional evaluation `--backbone` checks the recorded selection. Earlier runs can be evaluated with their original `--case` and `--config`.

Node generation inherits the backbone's native generation configuration and chat
template. Owners accept `generation` sampling overrides and `chat_template_kwargs`.
An explicit node `temperature` samples the raw model policy at that temperature;
zero selects greedy decoding.
Model identities bind the native generation configuration and decoding protocol.

## Development calibration

A `calibration` mapping locks the development graph configurations before observation.

| Field | Meaning |
| --- | --- |
| `calibration.graphs` | Distinct legal configurations, each listed by its active node and route names; the first is the contrast reference |
| `calibration.repetitions` | Positive observation count per development configuration and parameter version |
| `calibration.residual_bound` | Finite nonnegative structural bound on the signed model's residual over the feasible graph family |
| `search.features` | Declared singleton or pair supports; the development contrasts identify every included coefficient |

The design has entries `D[i,j] = psi_j(x_i) - psi_j(x_0)`. Its column rank equals the number of declared features. Legality, rank, probe allowance, and execution reserve are checked before model loading. Graph constraints can make feature columns dependent; declare an independent basis for that graph family. The residual bound describes the declared graph family under the measured task and parameter state.

At each frozen checkpoint, calibration measures the predeclared continuation score for the locked configurations. It uses separate observation keys in the same persistent confidence registry as search and training credit. The first declared observation prefix is frozen. For contrast pseudoinverse `D^+`, the coefficient estimator has mean weights `A = [-D^+ 1, D^+]`. Each coefficient radius is the corresponding row of `abs(A) @ (r + residual_bound)`, where `r` contains the paper's anytime radii for those mean identifiers. Combining the shared-reference weights retains the cancellation in a four-configuration interaction.

Calibration is bound to the graph and feature design, task state, score source, accepted parameter version, and candidate snapshot identity. The null program and each materialized candidate receive their own observations. Accepted updates initiate fresh calibration. Resume preserves completed observations, interval allocations, and charged work. The `calibrate` command performs this stage and pauses; `run --resume` continues the same run. The `run` command also performs calibration automatically when the case declares a plan.

Search fits and verification retain their 70/30 assignment. Development measurements use distinct keys and are excluded from those two observation pools. Calibrated intervals are conservatively recentered around the current fit. Cross-partition profiles use the resulting coefficient upper endpoints. A fresh task/version residual certificate supplies the uniform model allowance. The 95th-percentile verification residual remains a diagnostic; a valid structural residual bound covers every feasible graph under the calibrated version. Calibration probes consume the existing probe cap and operational ledger.

## Accounting and certificates

An action's declared upper cost is reserved and persisted before execution. Completed actions record synchronized wall time across allocated local model GPUs, plus separately reported external GPU time. External time excludes GPUs already included in the local meter. Missing external usage retains the conservative reservation. Native prompt preparation and checkpoint serialization are CPU operations. Failed or interrupted actions retain their reserved charge. An exceeded cost bound is reported explicitly. Budget guarantees depend on the declared action bounds.

Execution bounds cover parameter-version activation and restoration as well as the graph's model and tool calls. These transfers run inside the action's GPU-time measurement. Accepted and resumed parameter states are materialized at the next measured action. Candidate probes restore the accepted state before their measurement ends.

Active checkpoints use the same accounting protocol throughout a run. A checkpoint from an earlier protocol requires a new output directory for further optimization. Completed earlier runs remain available for independent evaluation.

The objective attributes measured training and candidate-validation cost to each program. Shared data collection and discarded work remain in the operational ledger. Every search also limits graph execution to the remaining operational budget. Accepted versions retain all earlier physical costs. All confidence identifiers, observations, costs, and accepted/candidate parameter versions survive resume. Completed probe outcomes are persisted with their cost and committed idempotently to the confidence registry and per-action observation files. An interrupted pilot can retry within the remaining pilot and budget limits; its earlier reservation remains charged. Immutable snapshot identities preserve the committed parameter manifest across interrupted saves. A changed checkpoint or accepted version receives fresh score intervals. Accepted updates invalidate their owner's dependency closure; the implementation conservatively refits the complete signed model.

The initial graph pool uses a seeded 70/30 fit/verification partition with integer rounding. Additional legal witnesses use a stable seeded hash split with a 70% fit fraction, independent of discovery order and observed scores. Assignments are persisted. The uniform residual allowance uses the anytime radius and a certified Hamming covering bound. The 95th-percentile residual is diagnostic. Without a calibration plan, coefficient radii remain unbounded unless `coefficient_residual_bound` supplies a structural modeling assumption and the fit design is identifiable. DP values upper-bound the relaxed graph family; every executable witness is separately checked against the original constraints. Each certified DP report includes its feasible-witness relaxation-gap upper bound. An empty evaluated frontier is reported as unavailable.

The sequence KL term uses full-support current-policy completion samples at temperature 1 and a score-function gradient against the accepted reference. Sequence log probabilities sum the provided target tokens. Candidate optimization uses PyTorch SGD with momentum and weight decay both set to zero, applying the stated gradient step with the declared learning rate. Snapshot files contain the trainable owner scope. Training with full parameters requires memory for model weights, gradients, and activations.

Graph-decision intervals retain the paper's anytime radius. Training credit uses the conditional Hoeffding mean radius `sqrt(log(2 / delta_a_n) / (2 * n))` for bounded continuation scores, using the same persistent mean identifiers and failure allocations. Two-mean and four-mean contrast radii sum the corresponding mean radii. The paper requires valid credit radii; this implementation uses the bounded-score estimator above. This stronger mean event also covers the wider direct interval under the shared failure budget.

## Independent evaluation

Final metric records contain `value` and `unit` (`rate`, `score`, or `count`). Rates are converted to percentages during aggregation; counts retain their native units. MLGym records select one native metric, declare `direction` as `higher` or `lower`, and specify `reported_precision` as decimal places in the original value's units. Win13 gives a win 1, a tie at the reported precision 0.5, and a loss 0 across thirteen tasks within each seed, then reports `100 * credit / 13`. Its reference is No TTT, selected through `--mlgym-reference`, independently of the paired-comparison `--baseline`. FACT evidence citations count supported statement-citation pairs, including distinct statements sharing a source.

Summaries compute means and sample standard deviations over seeds 0–4. Paired intervals use 10,000 task bootstrap draws stratified by seed. Method and baseline records require identical task identities and native units. The external baselines listed in the protocol are installed from their upstream releases; the `--method` argument labels evaluation records. Baseline execution uses the upstream method's own entry point, and its metrics can be converted to the same final-evaluation record format for paired aggregation.
