import json

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
