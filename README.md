# FrogNano

FrogNano is a compact 4B coding agent built on Qwen3.5-4B for repository-level
software engineering. It uses the lightweight Leaf harness to navigate
codebases, debug issues, edit files, and run tests.
Its agent-specific post-training uses only reinforcement learning on around
1,500 synthetic software-engineering task environments. TaskPilot generates and
calibrates tasks to the evolving model's capabilities, targeting the frontier
of what it can learn. This training uses no solution trajectories distilled
from larger models.

This repository provides the Leaf harness and evaluation tooling for running
coding agents in isolated Kubernetes sandboxes. It supports OpenAI-compatible
model endpoints and five tools: `Read`, `Write`, `Edit`, `Glob`, and `Bash`.

**[Technical report](https://arxiv.org/abs/2609.07925)** | **[Model weights on Hugging Face](https://huggingface.co/microsoft/FrogNano-4B-2609)**

## Requirements

- Python 3.12 or newer and Git.
- A Kubernetes cluster, an existing namespace, and permission to manage pods,
  execute commands in them, and manage network policies.
- Pull access to the benchmark container images.
- An OpenAI-compatible endpoint with reasoning and tool-call parsers configured
  for the model. For Qwen3.5 with SGLang, use `--reasoning-parser qwen3` and
  `--tool-call-parser qwen3_coder`.

Task images need Bash, GNU coreutils, and Python 3.6 or newer. Public-network
images can bootstrap missing coreutils and Python through `apt-get` or `apk`
when package installation is permitted. Network-isolated images must include
these dependencies.

## Install

From a checkout of this repository, using uv:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Alternatively, using venv and pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Run evaluations

Select the served checkpoint, endpoint, and an accessible Kubernetes context
and namespace. Replace the placeholder values below:

```bash
export FROGNANO_MODEL_NAME="your-served-checkpoint"
export FROGNANO_MODEL_BASE_URL="https://your-model-endpoint.example/v1"
export OPENAI_API_KEY="your-endpoint-key"
export KUBE_CONTEXT="your-cluster-context"
export K8S_NAMESPACE="your-existing-namespace"
export FROGNANO_OUTPUT_ROOT="$HOME/frognano-eval-results/$(date -u +%Y%m%dT%H%M%SZ)"
```

For an unauthenticated endpoint, use any non-empty `OPENAI_API_KEY` placeholder.
Use a fresh output root for each independent experiment; keep the same root
when resuming one.

Each command below runs a full benchmark using its YAML configuration:

```bash
frognano-eval run --config frognano/configs/eval/swebench-verified.yaml
frognano-eval run --config frognano/configs/eval/swebench-pro.yaml
frognano-eval run --config frognano/configs/eval/terminal-bench-2-verified.yaml
frognano-eval run --config frognano/configs/eval/patch-eval-verified.yaml
```

| Benchmark | Tasks | Completion tokens per turn |
|---|---:|---:|
| SWE-bench Verified | 500 | 8,192 |
| SWE-bench Pro | 731 | 32,000 |
| Terminal-Bench 2.0 Verified | 89 | 32,000 |
| PatchEval Verified | 230 | 8,192 |

All presets use **three seeds and 150 shared workers**, 150 agent steps,
a 131,072-token context limit, and a 10,800-second agent budget that overrides
task-native time limits. Sampling uses temperature 0.6, `top_p=0.95`,
`top_k=20`, `min_p=0`, presence penalty 0, repetition penalty 1, thinking
enabled, and multiple tool calls per response. Task order uses shuffle seed 42.

Dataset revisions are pinned. Terminal uses the ZAI Verified catalog.
SWE-bench Verified also includes an image-digest lock; the other presets
do not pin image digests. Matching scores requires matching checkpoint,
tokenizer, serving configuration, task images, and evaluation protocol.

### Optional environment settings

| Variable | Purpose |
|---|---|
| `FROGNANO_MAX_WORKERS` | Override the shared worker count. |
| `FROGNANO_TOKENIZER` | Matching Hugging Face tokenizer ID/path or tiktoken encoding for context accounting. |
| `FROGNANO_CACHE_DIR` | Benchmark definition cache directory. |
| `KUBE_CONFIG_PATH` | Kubeconfig file; otherwise use the standard Kubernetes configuration. |
| `K8S_IMAGE_REGISTRY` | Image mirror prefix, replacing the registry while retaining repository paths, tags, and digests. |
| `K8S_PULL_SECRET` | Existing image-pull secret in the task namespace. |
| `K8S_SERVICE_ACCOUNT` | Service account for task pods. |

A mirror must contain every referenced image and digest; FrogNano does not
copy images or fall back to the original registry. Registry credentials belong
in the Kubernetes pull secret, not in configuration files.

### Custom configurations

Copy a preset, edit its settings, and run the copied file:

```bash
cp frognano/configs/eval/swebench-verified.yaml eval.yaml
# Edit eval.yaml before running.
frognano-eval run --config eval.yaml
```

For a one-task smoke run, set `num_tasks: 1`, `seeds_per_task: 1`, and
`max_workers: 1` in the copy. Use `task_ids` to select specific tasks.
Set `max_workers_per_seed` for separate per-seed pools; their combined capacity
must not exceed `max_workers`. An `image_digest_lock` JSON file can pin task
images using an `images` list of `task_id`/`digest` pairs.

## Results and recovery

Each benchmark writes its own subdirectory under `FROGNANO_OUTPUT_ROOT`:

```text
<benchmark>/
  config.json
  results.jsonl
  summary.json
  trajectories/<instance_id>/
    trajectory_seed-0.json
    generated_seed-0.patch
```

`results.jsonl` records status, reward, and exit reason per task and seed.
`summary.json` reports resolved outcomes over all scheduled task-seed pairs,
not pass@3. FrogNano grades the final workspace even when an agent reaches
its context, step, or time limit.

With `resume: true`, completed outcomes, including valid unresolved outcomes,
are preserved; failed and unstarted pairs are eligible to run. Each pair allows
two full attempts for execution errors. Set `resume_retry_error_contains` in
a custom config to restrict retries of recorded failures to a matching error.

Commands and tool requests use file transfers with checksummed output.
Lost execution acknowledgements trigger output retrieval, not command resubmission.
Pod recovery replays completed mutating actions, which can repeat external
side effects. Controller restarts begin unfinished rollouts from scratch,
not from saved partial conversations. The CLI runs in the foreground; use
an external process supervisor for unattended evaluations.

## Optional W&B tracking

Install the extra and provide credentials through the environment:

```bash
uv pip install -e '.[wandb]'
export WANDB_API_KEY="your-wandb-key"
```

For pip, use `python -m pip install -e '.[wandb]'` instead.

Add this block to a copied configuration such as `eval.yaml`:

```yaml
wandb:
  entity: your-team
  project: coding-agent-evaluations
```

Then run `frognano-eval run --config eval.yaml`. W&B tracks overall and per-seed
metrics and uploads configuration, results, and summary artifacts. The saved
run ID lets resumed evaluations continue the same W&B run.

With exactly three seeds, `overall/pass_at_3_percent` counts selected tasks
with at least one completed seed whose reward is at least 1. Each task counts
once; all selected tasks remain in the denominator. W&B also logs
`overall/pass_at_3_resolved_tasks` and `overall/pass_at_3_total_tasks`.
These values are restored on resume and remain provisional while work is pending.
