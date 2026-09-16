from pathlib import Path

import pytest
import yaml

from frognano.config import EvalConfig, KubernetesConfig, ModelConfig, load_config
from frognano.datasets import _DATASETS
from frognano.datasets.patch_eval_verified import SOURCE as PATCH_EVAL_VERIFIED_SOURCE

_EVAL_PRESETS = [
    ("swebench-verified", "swebench_verified", 5, 8192),
    ("swebench-pro", "swebench_pro", 2, 32000),
    ("terminal-bench-2-verified", "terminal_bench_2_verified", 2, 32000),
    ("patch-eval-verified", "patch_eval_verified", 5, 8192),
]
_PRESET_DIR = Path(__file__).resolve().parents[1] / "frognano/configs/eval"


def test_load_config_expands_environment_and_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_ROOT", str(tmp_path))
    monkeypatch.setenv("K8S_IMAGE_REGISTRY", "registry.test")
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        """
dataset: swebench_verified
output_dir: ${EVAL_ROOT}/results
image_digest_lock: ${EVAL_ROOT}/images.json
model:
  name: test-model
  base_url: http://model.test/v1/
kubernetes:
  namespace: ${MISSING_NAMESPACE:-eval}
  image_registry: ${K8S_IMAGE_REGISTRY:-}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.output_dir == tmp_path / "results"
    assert config.image_digest_lock == tmp_path / "images.json"
    assert config.model.base_url == "http://model.test/v1"
    assert config.model.max_tokens_per_turn == 8192
    assert config.kubernetes.namespace == "eval"
    assert config.kubernetes.image_registry == "registry.test"
    assert config.resume is True
    assert config.resume_retry_error_contains is None
    assert config.max_workers_per_seed is None


def test_model_config_defaults_to_8192_completion_tokens() -> None:
    assert (
        ModelConfig(name="model", base_url="http://model/v1").max_tokens_per_turn
        == 8192
    )


@pytest.mark.parametrize(
    "value,expected", [(None, None), (0, None), ("0", None), (1024, 1024)]
)
def test_model_config_preserves_explicit_completion_limits(value, expected) -> None:
    config = ModelConfig.from_dict(
        {
            "name": "model",
            "base_url": "http://model/v1",
            "max_tokens_per_turn": value,
        }
    )

    assert config.max_tokens_per_turn == expected


def test_bundled_presets_cover_each_dataset_once(monkeypatch) -> None:
    monkeypatch.setenv("FROGNANO_MODEL_NAME", "test-model")
    monkeypatch.setenv("FROGNANO_MODEL_BASE_URL", "http://model.test/v1")
    monkeypatch.delenv("FROGNANO_MAX_WORKERS", raising=False)
    presets = sorted(path.stem for path in _PRESET_DIR.glob("*.yaml"))
    assert set(presets) == {entry[0] for entry in _EVAL_PRESETS}
    datasets = [load_config(profile).dataset for profile in presets]
    assert len(datasets) == len(set(datasets))
    assert set(datasets) == set(_DATASETS)


@pytest.mark.parametrize(
    "profile",
    [
        "patch-eval-verified-3-seeds",
        "swebench-verified-3-seeds-300-workers",
        "swebench-verified-service-aligned",
        "swebench-pro-service-aligned",
        "terminal-bench-2",
        "terminal-bench-2-verified-service-aligned",
    ],
)
def test_superseded_presets_are_not_packaged(profile) -> None:
    with pytest.raises(FileNotFoundError, match="evaluation config does not exist"):
        load_config(profile)


def test_config_accepts_selective_resume() -> None:
    raw = {
        "dataset": "swebench_verified",
        "model": {"name": "model", "base_url": "http://model/v1"},
        "resume_retry_error_contains": "APIConnectionError",
    }

    config = EvalConfig.from_dict(raw)

    assert config.resume_retry_error_contains == "APIConnectionError"
    raw["resume"] = False
    with pytest.raises(ValueError, match="requires resume: true"):
        EvalConfig.from_dict(raw)


@pytest.mark.parametrize("value", ["", "   ", 123, False, []])
def test_config_rejects_invalid_resume_filter(value) -> None:
    with pytest.raises(ValueError, match="must be a non-empty string"):
        EvalConfig.from_dict(
            {
                "dataset": "swebench_verified",
                "model": {"name": "model", "base_url": "http://model/v1"},
                "resume_retry_error_contains": value,
            }
        )


def test_empty_image_registry_uses_docker_hub(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("K8S_IMAGE_REGISTRY", raising=False)
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        """
dataset: swebench_verified
model:
  name: test-model
  base_url: http://model.test/v1
kubernetes:
  image_registry: ${K8S_IMAGE_REGISTRY:-}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.kubernetes.image_registry is None


@pytest.mark.parametrize("suffix", ["", ".yaml"])
def test_load_config_reads_packaged_config_by_name(monkeypatch, suffix) -> None:
    monkeypatch.setenv("FROGNANO_MODEL_NAME", "test-model")
    monkeypatch.setenv("FROGNANO_MODEL_BASE_URL", "http://model.test/v1")

    config = load_config(f"swebench-verified{suffix}")

    assert config.dataset == "swebench_verified"
    assert config.model.name == "test-model"


def test_load_config_reads_patch_eval_verified_full_config(monkeypatch) -> None:
    monkeypatch.setenv("FROGNANO_MODEL_NAME", "test-model")
    monkeypatch.setenv("FROGNANO_MODEL_BASE_URL", "http://model.test/v1")
    monkeypatch.delenv("FROGNANO_MAX_WORKERS", raising=False)

    config = load_config("patch-eval-verified")

    assert config.dataset == "patch_eval_verified"
    assert config.num_tasks is None
    assert config.seeds_per_task == 3
    assert config.max_workers == 150
    assert config.max_workers_per_seed is None
    assert config.output_dir.name == "patch-eval-verified"


def test_config_resolves_packaged_image_digest_lock() -> None:
    config = EvalConfig.from_dict(
        {
            "dataset": "swebench_verified",
            "model": {"name": "model", "base_url": "http://model/v1"},
            "image_digest_lock": "sweb-v-20260904",
        }
    )

    assert config.image_digest_lock is not None
    assert config.image_digest_lock.name == "sweb-v-20260904.json"
    assert config.image_digest_lock.parent.name == "image_locks"
    assert config.image_digest_lock.is_file()


def test_kubernetes_config_accepts_resource_and_lifetime_overrides() -> None:
    config = KubernetesConfig.from_dict(
        {"memory_limit": "32Gi", "pod_lifetime_sec": 10800}
    )
    assert config.memory_limit == "32Gi"
    assert config.pod_lifetime_sec == 10800
    with pytest.raises(ValueError, match="pod_lifetime_sec must be positive"):
        KubernetesConfig.from_dict({"pod_lifetime_sec": 0})


def test_bundled_config_can_be_copied_for_smoke_evaluation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("FROGNANO_MODEL_NAME", "test-model")
    monkeypatch.setenv("FROGNANO_MODEL_BASE_URL", "http://model.test/v1")
    raw = yaml.safe_load((_PRESET_DIR / "swebench-verified.yaml").read_text())
    raw.update(
        num_tasks=1,
        seeds_per_task=1,
        max_workers=1,
        max_workers_per_seed=None,
        output_dir=str(tmp_path / "smoke-results"),
    )
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(path)

    assert config.num_tasks == config.seeds_per_task == config.max_workers == 1
    assert config.max_workers_per_seed is None
    assert config.output_dir == tmp_path / "smoke-results"
    assert config.max_steps == 150
    assert config.model.max_tokens_per_turn == 8192


@pytest.mark.parametrize(
    ("profile", "dataset", "retries", "completion_cap"), _EVAL_PRESETS
)
@pytest.mark.parametrize("worker_override", [None, 30, 300])
def test_load_bundled_config(
    tmp_path,
    monkeypatch,
    profile,
    dataset,
    retries,
    completion_cap,
    worker_override,
) -> None:
    monkeypatch.setenv("FROGNANO_MODEL_NAME", "test-checkpoint")
    monkeypatch.setenv("FROGNANO_MODEL_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("FROGNANO_OUTPUT_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("FROGNANO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("KUBE_CONTEXT", "test-cluster")
    monkeypatch.setenv("K8S_IMAGE_REGISTRY", "registry.example.test")
    monkeypatch.delenv("FROGNANO_MAX_WORKERS", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    if worker_override is not None:
        monkeypatch.setenv("FROGNANO_MAX_WORKERS", str(worker_override))

    config = load_config(profile)

    assert config.dataset == dataset
    assert config.output_dir == tmp_path / "results" / profile
    assert config.cache_dir == tmp_path / "cache"
    assert config.num_tasks is None
    assert config.task_ids == ()
    assert config.seed == 42
    assert config.seeds_per_task == 3
    assert config.max_workers == (worker_override if worker_override else 150)
    assert config.max_workers_per_seed is None
    assert config.max_attempts == 2
    assert config.max_steps == 150
    assert config.max_context_tokens == 131072
    assert config.max_total_time_sec == 10800
    assert config.resume is True
    assert config.resume_retry_error_contains is None
    assert config.model.name == "test-checkpoint"
    assert config.model.base_url == "http://model.test/v1"
    assert config.model.api_key_env == "OPENAI_API_KEY"
    assert config.model.temperature == 0.6
    assert config.model.max_tokens_per_turn == completion_cap
    assert config.model.max_retries == retries
    assert config.model.timeout_sec == 7200
    assert config.model.parallel_tool_calls is True
    assert config.model.extra_body == {
        "chat_template_kwargs": {"enable_thinking": True},
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    assert config.kubernetes.context == "test-cluster"
    assert config.kubernetes.pod_lifetime_sec is None
    assert config.kubernetes.pod_start_timeout_sec == 900
    assert config.kubernetes.keep_pods is False
    assert config.kubernetes.image_registry == "registry.example.test"
    assert config.wandb is None
    if dataset == "patch_eval_verified":
        assert config.kubernetes.memory_limit is None
    else:
        assert config.kubernetes.memory_limit == "32Gi"
    if dataset == "swebench_verified":
        assert config.image_digest_lock is not None
        assert config.image_digest_lock.name == "sweb-v-20260904.json"
        assert config.image_digest_lock.is_file()
    else:
        assert config.image_digest_lock is None
    if dataset == "patch_eval_verified":
        assert config.dataset == PATCH_EVAL_VERIFIED_SOURCE.name
        assert (
            PATCH_EVAL_VERIFIED_SOURCE.revision
            == "b43285cdde80cc04608d5f1178a330b740c91c2d"
        )
        assert PATCH_EVAL_VERIFIED_SOURCE.agent_network_mode == "no-network"
        assert PATCH_EVAL_VERIFIED_SOURCE.verifier_network_mode == "no-network"


@pytest.mark.parametrize("profile", [entry[0] for entry in _EVAL_PRESETS])
@pytest.mark.parametrize("missing", ["FROGNANO_MODEL_NAME", "FROGNANO_MODEL_BASE_URL"])
def test_bundled_configs_require_explicit_model(monkeypatch, profile, missing) -> None:
    monkeypatch.setenv("FROGNANO_MODEL_NAME", "test-checkpoint")
    monkeypatch.setenv("FROGNANO_MODEL_BASE_URL", "http://model.test/v1")
    monkeypatch.delenv(missing, raising=False)

    with pytest.raises(ValueError, match="model.name and model.base_url are required"):
        load_config(profile)


def test_load_config_reads_wandb_settings(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_HOST", "https://wandb.example.test")
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        """
dataset: swebench_verified
model:
  name: test-model
  base_url: http://model.test/v1
kubernetes: {}
wandb:
  base_url: ${WANDB_HOST}
  entity: research
  project: evaluations
  name: smoke-run
  tags: [smoke, leaf]
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.wandb is not None
    assert config.wandb.base_url == "https://wandb.example.test"
    assert config.wandb.entity == "research"
    assert config.wandb.project == "evaluations"
    assert config.wandb.name == "smoke-run"
    assert config.wandb.tags == ("smoke", "leaf")


@pytest.mark.parametrize("registry", [None, "", "mirror.example:5000/benchmarks/"])
@pytest.mark.parametrize(
    "name",
    [
        "swebench-verified",
        "swebench-pro",
        "terminal-bench-2-verified",
        "patch-eval-verified",
    ],
)
def test_packaged_configs_support_optional_image_registry(
    monkeypatch, name, registry
) -> None:
    monkeypatch.setenv("FROGNANO_MODEL_NAME", "test-model")
    monkeypatch.setenv("FROGNANO_MODEL_BASE_URL", "http://model.test/v1")
    if registry is None:
        monkeypatch.delenv("K8S_IMAGE_REGISTRY", raising=False)
    else:
        monkeypatch.setenv("K8S_IMAGE_REGISTRY", registry)

    config = load_config(name)

    expected = registry.rstrip("/") if registry else None
    assert config.kubernetes.image_registry == expected


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_workers", 0, "max_workers must be positive"),
        ("max_workers_per_seed", 0, "max_workers_per_seed must be positive"),
        ("max_workers_per_seed", -1, "max_workers_per_seed must be positive"),
        ("num_tasks", -1, "num_tasks cannot be negative"),
        (
            "max_total_time_sec",
            30,
            "max_total_time_sec must be between 60 and 10800",
        ),
        (
            "max_total_time_sec",
            10801,
            "max_total_time_sec must be between 60 and 10800",
        ),
    ],
)
def test_config_rejects_invalid_limits(field, value, message) -> None:
    raw = {
        "dataset": "swebench_verified",
        "model": {"name": "model", "base_url": "http://localhost/v1"},
        "kubernetes": {},
        field: value,
    }

    with pytest.raises(ValueError, match=message):
        EvalConfig.from_dict(raw)


def test_config_validates_per_seed_worker_budget() -> None:
    raw = {
        "dataset": "swebench_verified",
        "model": {"name": "model", "base_url": "http://localhost/v1"},
        "seeds_per_task": 3,
        "max_workers": 150,
        "max_workers_per_seed": 50,
    }

    config = EvalConfig.from_dict(raw)

    assert config.max_workers_per_seed == 50
    assert config.max_workers == 150

    raw["max_workers"] = 149
    with pytest.raises(ValueError, match="max_workers must be at least"):
        EvalConfig.from_dict(raw)


def test_config_requires_dataset_and_model() -> None:
    with pytest.raises(ValueError, match="dataset is required"):
        EvalConfig.from_dict({})
    with pytest.raises(ValueError, match="model.name"):
        EvalConfig.from_dict({"dataset": "swebench_verified"})


def test_config_requires_wandb_entity_and_project() -> None:
    raw = {
        "dataset": "swebench_verified",
        "model": {"name": "model", "base_url": "http://localhost/v1"},
        "kubernetes": {},
        "wandb": {"entity": "research"},
    }

    with pytest.raises(ValueError, match="wandb.entity and wandb.project"):
        EvalConfig.from_dict(raw)
