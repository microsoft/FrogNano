# FrogNano

FrogNano is a small, reproducible reference implementation for evaluating
coding agents on software engineering tasks. It uses the Leaf tool-calling
harness to evaluate agents on Harbor task datasets, with task execution
isolated in Kubernetes sandboxes. It includes integrations for SWE-bench
Verified, SWE-bench Pro, Terminal-Bench 2.0, and PatchEval Verified.
This code release accompanies the FrogNano technical report.

## What is included

- A Leaf coding-agent harness with `Read`, `Write`, `Edit`, `Glob`, and `Bash`
  tools.
- Isolated Kubernetes environments for task execution and verification.
- Integrations for SWE-bench Verified, SWE-bench Pro, Terminal-Bench 2.0, and
  PatchEval Verified.
- Support for configurable OpenAI-compatible model endpoints.
- Reproducible, commit-pinned Harbor benchmark acquisition and configuration.
- Parallel evaluation, retries, resume support, and structured result
  artifacts.

## Requirements

- Python 3.12 or newer.
- Git, used to retrieve benchmark task definitions.
- A Kubernetes cluster with permission to manage task pods and network
  policies. Minikube can be used for local evaluations.
- Pull access to the benchmark container images.
- An OpenAI-compatible model endpoint with structured tool-call support.

## Install

```bash
python -m pip install git+https://github.com/microsoft/FrogNano.git
```

## Run a benchmark

### Configure the model endpoint

Each benchmark configuration specifies an OpenAI-compatible model endpoint.
The endpoint's reasoning and tool-call parsers must match the model. For
example, Qwen3.5 served with SGLang can use `--reasoning-parser qwen3` and
`--tool-call-parser qwen3_coder`.

### Select a tokenizer

This step is optional. Context accounting first tries the configured model's
Hugging Face tokenizer, then tiktoken, and finally a lightweight local
estimate. To override the tokenizer, set a Hugging Face model or tiktoken model
name:

```bash
export FROGNANO_TOKENIZER=Qwen/Qwen3.5-32B
export FROGNANO_TOKENIZER=gpt-4o
```

An explicit tiktoken encoding name such as `o200k_base` is also accepted.

### Run a smoke evaluation

The bundled configurations evaluate one task with one worker by default:

```bash
export OPENAI_API_KEY=unused-for-local-endpoints
export K8S_NAMESPACE=default

frognano-eval run --config swebench-verified
```

### Choose a benchmark

| Configuration | Dataset name |
|---|---|
| `swebench-verified` | `swebench_verified` |
| `swebench-pro` | `swebench_pro` |
| `terminal-bench-2` | `terminal_bench_2` |
| `patch-eval` | `patch_eval` |

To inspect a registered dataset source, run:

```bash
frognano-eval dataset swebench_verified
```

### Customize an evaluation

Copy a YAML file from `frognano/configs/eval/` and pass its path to `--config`.
Update the model endpoint, task selection, concurrency, or Kubernetes settings
in the copied configuration.

To run the full dataset with 50 concurrent workers, set:

```yaml
num_tasks: null
max_workers: 50
```

Then run the copied configuration:

```bash
frognano-eval run --config swebench-verified-full.yaml
```

To evaluate specific tasks instead, set:

```yaml
task_ids:
  - astropy__astropy-12907
```

Benchmark images resolve through Docker Hub by default. The bundled
configurations read the registry from `K8S_IMAGE_REGISTRY`:

```bash
export K8S_IMAGE_REGISTRY=registry.example.com
```

Alternatively, set the registry in a copied YAML configuration:

```yaml
kubernetes:
  image_registry: registry.example.com
```

Use `image_registry: ${K8S_IMAGE_REGISTRY:-}` in a custom configuration to
read it from the environment. A command-line parameter overrides either form:

```bash
frognano-eval run --config eval.yaml --image-registry registry.example.com
```

Registry prefixes may include a port or mirror namespace, such as
`registry.example.com:5000/benchmarks`. They apply to task images without an
explicit registry; fully qualified image references are unchanged.

To pin each selected task image to an immutable digest, provide a Harbor image
lock containing an `images` list with `task_id` and `digest` fields:

```yaml
image_digest_lock: /path/to/sweb-v-20260904.json
```

FrogNano replaces each task image tag with the matching `@sha256:...` digest
and fails before launching work if a selected task is missing from the lock.
The configured image registry is preserved. Use
`image_digest_lock: sweb-v-20260904` to select the bundled Verified lock.
The lock contains only an `images` list of `task_id`/`digest` pairs. The pinned
task catalog supplies image repositories, and the YAML/environment/CLI setting
selects the registry. A mirror must provide the same image repositories and
immutable digests.

### Track an evaluation with W&B

Install the optional W&B integration:

```bash
python -m pip install \
  "frognano[wandb] @ git+https://github.com/microsoft/FrogNano.git"
```

Set `WANDB_API_KEY` and add a `wandb` block to the evaluation configuration:

```yaml
wandb:
  base_url: https://api.wandb.ai
  entity: example-team
  project: coding-agent-evaluations
  name: swebench-verified
  tags: [leaf, swebench]
```

FrogNano logs resolve, unresolve, and error percentages in an `overall` section
and one section per seed. Overall also includes completed percentage, result
totals, and stop-reason totals. Errors are rollouts that did not produce a valid
benchmark result. FrogNano uploads `config.json`, `results.jsonl`, and
`summary.json` at completion. The W&B run ID is stored in the output directory
so resumed evaluations continue writing to the same run.

## Outputs

Each run writes:

```text
<output_dir>/
  config.json
  results.jsonl
  summary.json
  trajectories/
    <instance_id>/
      trajectory_seed-0.json
      generated_seed-0.patch
```

With `resume: true`, completed `(instance_id, seed)` pairs in `results.jsonl`
are skipped.

## Adding datasets

Add a module under `frognano/datasets/` that defines an immutable
`DatasetSource`. A Harbor directory-backed dataset can reuse
`load_harbor_dataset`:

```python
SOURCE = DatasetSource(
    name="example",
    display_name="Example",
    source_url="https://github.com/example/harbor-datasets.git",
    revision="<full-commit-sha>",
    subpath="datasets/example",
    pod_prefix="example",
)
```

Import the source in `frognano/datasets/__init__.py` and add its loader to
`_DATASETS`:

```python
from .example import SOURCE as EXAMPLE

_DATASETS = {
    # Existing datasets...
    EXAMPLE.name: (EXAMPLE, load_harbor_dataset),
}
```

This keeps dataset identity and policy separate from the harness, allowing new
integrations to provide their own loaders or runtime requirements without
changing the orchestration layer.
