import json

import pytest

from frognano.cli import main
from frognano.datasets import load_dataset


@pytest.mark.parametrize(
    "name,subpath,display_name",
    [
        ("swebench_verified", "datasets/swebench-verified", "SWE-bench Verified"),
        ("patch_eval_verified", "patcheval/datasets", "PatchEval Verified"),
    ],
)
def test_dataset_command_prints_registered_source(
    capsys, name, subpath, display_name
) -> None:
    assert main(["dataset", name]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["name"] == name
    assert output["subpath"] == subpath
    assert output["display_name"] == display_name


def test_run_command_prints_summary(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "frognano.cli.load_config",
        lambda path: object(),
    )
    monkeypatch.setattr(
        "frognano.cli.run_evaluation",
        lambda config: {"jobs_failed": 0, "resolved": 1},
    )

    assert main(["run", "--config", "eval.yaml"]) == 0
    assert json.loads(capsys.readouterr().out)["resolved"] == 1


@pytest.mark.parametrize(
    "name",
    [
        "swebench-verified",
        "swebench-pro",
        "terminal-bench-2-verified",
        "patch-eval-verified",
    ],
)
def test_run_command_loads_latest_bundled_presets(monkeypatch, capsys, name) -> None:
    monkeypatch.setenv("FROGNANO_MODEL_NAME", "test-model")
    monkeypatch.setenv("FROGNANO_MODEL_BASE_URL", "http://model.test/v1")
    monkeypatch.delenv("FROGNANO_MAX_WORKERS", raising=False)

    def run(config):
        assert config.dataset == name.replace("-", "_")
        assert config.num_tasks is None
        assert config.seeds_per_task == 3
        assert config.max_workers == 150
        assert config.max_workers_per_seed is None
        assert config.resume_retry_error_contains is None
        return {"jobs_failed": 0, "resolved": 1}

    monkeypatch.setattr("frognano.cli.run_evaluation", run)

    assert main(["run", "--config", name]) == 0
    assert json.loads(capsys.readouterr().out)["resolved"] == 1


def test_dataset_command_rejects_removed_standard_terminal() -> None:
    with pytest.raises(ValueError, match="unknown dataset 'terminal_bench_2'"):
        main(["dataset", "terminal_bench_2"])


@pytest.mark.parametrize("argument", [None, "cli.example:5000/mirror"])
@pytest.mark.parametrize("environment_registry", [None, "", "env.example"])
def test_patch_eval_registry_override_reaches_task_images(
    tmp_path, monkeypatch, capsys, argument, environment_registry
) -> None:
    monkeypatch.setenv("FROGNANO_MODEL_NAME", "test-model")
    monkeypatch.setenv("FROGNANO_MODEL_BASE_URL", "http://model.test/v1")
    if environment_registry is None:
        monkeypatch.delenv("K8S_IMAGE_REGISTRY", raising=False)
    else:
        monkeypatch.setenv("K8S_IMAGE_REGISTRY", environment_registry)
    (tmp_path / "patcheval_verified.json").write_text(
        json.dumps(
            [{"cve_id": "example", "image_url": "ghcr.io/patcheval-cve/example:1"}]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "frognano.datasets.patch_eval_verified.materialize_source",
        lambda *args: tmp_path,
    )

    def run(config):
        [task] = load_dataset(
            config.dataset,
            cache_dir=tmp_path,
            image_registry=config.kubernetes.image_registry,
        )
        expected = argument or environment_registry or "ghcr.io"
        assert task["docker_image"] == f"{expected}/patcheval-cve/example:1"
        return {"jobs_failed": 0, "resolved": 1}

    monkeypatch.setattr("frognano.cli.run_evaluation", run)
    arguments = ["run", "--config", "patch-eval-verified"]
    if argument:
        arguments += ["--image-registry", argument]

    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["resolved"] == 1


@pytest.mark.parametrize(
    ("configured", "environment", "argument", "expected"),
    [
        ("yaml.example", "env.example", None, "yaml.example"),
        ("${K8S_IMAGE_REGISTRY:-}", "env.example/", None, "env.example"),
        ("${K8S_IMAGE_REGISTRY:-}", None, None, None),
        (
            "yaml.example",
            "env.example",
            "cli.example:5000/mirror/",
            "cli.example:5000/mirror",
        ),
        ("${K8S_IMAGE_REGISTRY:-}", "env.example", " cli.example/ ", "cli.example"),
    ],
)
def test_run_command_selects_image_registry(
    tmp_path, monkeypatch, capsys, configured, environment, argument, expected
) -> None:
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        json.dumps(
            {
                "dataset": "swebench_verified",
                "image_digest_lock": "sweb-v-20260904",
                "seeds_per_task": 3,
                "max_workers": 150,
                "max_workers_per_seed": 50,
                "resume_retry_error_contains": "APIConnectionError",
                "model": {"name": "model", "base_url": "http://model.test/v1"},
                "kubernetes": {
                    "namespace": "evaluation",
                    "image_registry": configured,
                    "memory_limit": "32Gi",
                    "pod_lifetime_sec": 10800,
                },
            }
        ),
        encoding="utf-8",
    )
    if environment is None:
        monkeypatch.delenv("K8S_IMAGE_REGISTRY", raising=False)
    else:
        monkeypatch.setenv("K8S_IMAGE_REGISTRY", environment)

    def run(config):
        assert config.kubernetes.image_registry == expected
        assert config.kubernetes.namespace == "evaluation"
        assert config.kubernetes.memory_limit == "32Gi"
        assert config.kubernetes.pod_lifetime_sec == 10800
        assert config.seeds_per_task == 3
        assert config.max_workers == 150
        assert config.max_workers_per_seed == 50
        assert config.resume_retry_error_contains == "APIConnectionError"
        assert config.image_digest_lock is not None
        assert config.image_digest_lock.is_file()
        return {"jobs_failed": 0, "resolved": 1}

    monkeypatch.setattr("frognano.cli.run_evaluation", run)
    arguments = ["run", "--config", str(config_path)]
    if argument is not None:
        arguments += ["--image-registry", argument]

    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["resolved"] == 1


@pytest.mark.parametrize("registry", ["", " ", "/", " /// "])
def test_run_command_rejects_empty_registry(registry, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", "eval.yaml", "--image-registry", registry])

    assert error.value.code == 2
    assert "image registry must be non-empty" in capsys.readouterr().err
