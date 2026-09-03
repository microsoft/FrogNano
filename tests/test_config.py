from pathlib import Path

import pytest

from frognano.config import EvalConfig, load_config


def test_load_config_expands_environment_and_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_ROOT", str(tmp_path))
    monkeypatch.setenv("K8S_IMAGE_REGISTRY", "registry.test")
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        """
dataset: swebench_verified
output_dir: ${EVAL_ROOT}/results
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
    assert config.model.base_url == "http://model.test/v1"
    assert config.kubernetes.namespace == "eval"
    assert config.kubernetes.image_registry == "registry.test"
    assert config.resume is True


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_workers", 0, "max_workers must be positive"),
        ("num_tasks", -1, "num_tasks cannot be negative"),
        (
            "max_total_time_sec",
            30,
            "max_total_time_sec must be between 60 and 7200",
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


def test_config_requires_dataset_and_model() -> None:
    with pytest.raises(ValueError, match="dataset is required"):
        EvalConfig.from_dict({})
    with pytest.raises(ValueError, match="model.name"):
        EvalConfig.from_dict({"dataset": "swebench_verified"})
