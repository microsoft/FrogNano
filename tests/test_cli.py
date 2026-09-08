import json

import pytest

from frognano.cli import main


def test_dataset_command_prints_registered_source(capsys) -> None:
    assert main(["dataset", "swebench_verified"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["name"] == "swebench_verified"
    assert output["subpath"] == "datasets/swebench-verified"


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
                "model": {"name": "model", "base_url": "http://model.test/v1"},
                "kubernetes": {
                    "namespace": "evaluation",
                    "image_registry": configured,
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
