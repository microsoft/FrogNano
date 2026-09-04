import json
from pathlib import Path

from frognano import runner
from frognano.config import EvalConfig


def _config(tmp_path: Path) -> EvalConfig:
    return EvalConfig.from_dict(
        {
            "dataset": "swebench_verified",
            "output_dir": str(tmp_path),
            "model": {
                "name": "model",
                "base_url": "http://model/v1",
                "api_key_env": "TEST_MODEL_KEY",
            },
            "kubernetes": {},
            "max_workers": 2,
        }
    )


def test_run_evaluation_writes_results_and_summary(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setenv("TEST_MODEL_KEY", "secret")
    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda *args, **kwargs: [
            {"instance_id": "task-a"},
            {"instance_id": "task-b"},
        ],
    )

    def run_job(config, task, seed, api_key, stop_event):
        return {
            "instance_id": task["instance_id"],
            "seed": seed,
            "status": "completed",
            "reward": 1.0 if task["instance_id"] == "task-a" else 0.0,
        }

    monkeypatch.setattr(runner, "_run_job", run_job)

    summary = runner.run_evaluation(config)

    assert summary["jobs_completed"] == 2
    assert summary["resolved"] == 1
    assert summary["unresolved"] == 1
    assert summary["resolve_rate"] == 0.5
    assert summary["error_rate"] == 0.0
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 2


def test_run_evaluation_resumes_completed_jobs(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setenv("TEST_MODEL_KEY", "secret")
    (tmp_path / "results.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "task-a",
                "seed": 0,
                "status": "completed",
                "reward": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda *args, **kwargs: [{"instance_id": "task-a"}],
    )
    monkeypatch.setattr(
        runner,
        "_run_job",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed job should not run")
        ),
    )

    summary = runner.run_evaluation(config)

    assert summary["jobs_recorded"] == 1
    assert summary["resolved"] == 1


def test_run_evaluation_requires_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODEL_KEY", raising=False)

    try:
        runner.run_evaluation(_config(tmp_path))
    except ValueError as exc:
        assert "TEST_MODEL_KEY" in str(exc)
    else:
        raise AssertionError("missing API key should fail")


def test_run_evaluation_uses_latest_result_and_total_denominator(
    tmp_path, monkeypatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setenv("TEST_MODEL_KEY", "secret")
    (tmp_path / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instance_id": "task-a",
                        "seed": 0,
                        "status": "completed",
                        "reward": 1,
                    }
                ),
                json.dumps(
                    {
                        "instance_id": "task-a",
                        "seed": 0,
                        "status": "failed",
                        "reward": 0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda *args, **kwargs: [
            {"instance_id": "task-a"},
            {"instance_id": "task-b"},
        ],
    )
    monkeypatch.setattr(
        runner,
        "_run_job",
        lambda config, task, seed, api_key, stop_event: {
            "instance_id": task["instance_id"],
            "seed": seed,
            "status": "failed",
            "reward": 0,
        },
    )

    summary = runner.run_evaluation(config)

    assert summary["jobs_recorded"] == 2
    assert summary["jobs_completed"] == 0
    assert summary["resolve_rate"] == 0.0
    assert summary["error_rate"] == 1.0


def test_run_job_retries_then_writes_trajectory(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    task = {
        "instance_id": "task-a",
        "agent_timeout_sec": 300,
        "source": {"revision": "abc"},
    }
    attempts = []

    class Runtime:
        def __init__(self, task, config, *, run_id):
            attempts.append(run_id)
            if len(attempts) == 1:
                raise RuntimeError("transient")

        def compute_reward(self):
            return 1.0, "passed"

        def close(self):
            pass

    class Agent:
        def __init__(self, config):
            pass

        def run(
            self,
            environment,
            *,
            instance_id,
            seed,
            checkpoint_callback,
            stop_event,
        ):
            checkpoint_callback({"partial": True})
            return {
                "exit_reason": "agent",
                "output_patch": "diff",
            }

    monkeypatch.setattr(runner, "KubernetesTaskRuntime", Runtime)
    monkeypatch.setattr(runner, "LeafEnvironment", lambda runtime: runtime)
    monkeypatch.setattr(runner, "LeafAgent", Agent)

    result = runner._run_job(config, task, 0, "key")

    assert result["status"] == "completed"
    assert result["attempt"] == 2
    assert (tmp_path / "trajectories/task-a/trajectory_seed-0.json").is_file()
    assert (tmp_path / "trajectories/task-a/generated_seed-0.patch").is_file()


def test_run_job_returns_all_attempt_errors(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    task = {
        "instance_id": "task-a",
        "agent_timeout_sec": 300,
        "source": {},
    }

    class Runtime:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("cannot create pod")

    monkeypatch.setattr(runner, "KubernetesTaskRuntime", Runtime)

    result = runner._run_job(config, task, 0, "key")

    assert result["status"] == "failed"
    assert result["exit_reason"] == "infrastructure_error"
    assert result["error"].count("cannot create pod") == config.max_attempts


def test_run_job_retries_leaf_infrastructure_failures(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    task = {
        "instance_id": "task-a",
        "agent_timeout_sec": 300,
        "source": {},
    }
    calls = []

    class Runtime:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    class Agent:
        def __init__(self, config):
            pass

        def run(
            self,
            environment,
            *,
            instance_id,
            seed,
            checkpoint_callback,
            stop_event,
        ):
            calls.append(instance_id)
            return {
                "exit_reason": "llm_query_error",
                "error": "model unavailable",
                "output_patch": "",
            }

    monkeypatch.setattr(runner, "KubernetesTaskRuntime", Runtime)
    monkeypatch.setattr(runner, "LeafEnvironment", lambda runtime: runtime)
    monkeypatch.setattr(runner, "LeafAgent", Agent)

    result = runner._run_job(config, task, 0, "key")

    assert result["status"] == "failed"
    assert result["exit_reason"] == "llm_query_error"
    assert len(calls) == config.max_attempts
    assert result["error"].count("retryable Leaf failure") == config.max_attempts
