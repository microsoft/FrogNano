import json
from dataclasses import replace
from pathlib import Path

import pytest

from frognano import runner
from frognano.config import EvalConfig, WandbConfig


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


def test_run_evaluation_applies_image_digest_lock(tmp_path, monkeypatch) -> None:
    digest = "sha256:" + ("a" * 64)
    lock = tmp_path / "images.json"
    lock.write_text(
        json.dumps({"images": [{"task_id": "task-a", "digest": digest}]}),
        encoding="utf-8",
    )
    base_config = _config(tmp_path)
    registry = "registry.test:5000/mirror"
    config = replace(
        base_config,
        image_digest_lock=lock,
        kubernetes=replace(base_config.kubernetes, image_registry=registry),
    )
    tasks = [{"instance_id": "task-a", "docker_image": f"{registry}/example:latest"}]
    monkeypatch.setenv("TEST_MODEL_KEY", "secret")

    def load_dataset(*args, **kwargs):
        assert kwargs["image_registry"] == registry
        return tasks

    monkeypatch.setattr(runner, "load_dataset", load_dataset)

    def run_job(config, task, seed, api_key, stop_event):
        assert task["docker_image"] == f"{registry}/example@{digest}"
        return {
            "instance_id": task["instance_id"],
            "seed": seed,
            "status": "completed",
            "reward": 1,
        }

    monkeypatch.setattr(runner, "_run_job", run_job)

    summary = runner.run_evaluation(config)

    assert summary["resolved"] == 1
    assert tasks[0]["docker_image"] == f"{registry}/example:latest"
    saved_config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved_config["image_digest_lock"] == str(lock)
    assert saved_config["kubernetes"]["image_registry"] == registry


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


@pytest.mark.parametrize("resume", [True, False])
def test_wandb_initial_results_match_selected_tasks_and_seeds(
    tmp_path, monkeypatch, resume
):
    config = replace(
        _config(tmp_path),
        seeds_per_task=3,
        resume=resume,
        wandb=WandbConfig(entity="research", project="evaluations"),
    )
    original = [
        {
            "instance_id": task,
            "seed": seed,
            "status": status,
            "reward": reward,
        }
        for task, seed, status, reward in [
            ("task-a", 0, "failed", 0),
            ("task-a", 0, "completed", 1),
            ("task-a", 3, "completed", 1),
            ("unselected-task", 0, "completed", 1),
        ]
    ]
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in original), encoding="utf-8"
    )
    monkeypatch.setenv("TEST_MODEL_KEY", "secret")
    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda *args, **kwargs: [{"instance_id": "task-a"}, {"instance_id": "task-b"}],
    )
    monkeypatch.setattr(
        runner,
        "_run_job",
        lambda config, task, seed, *args: {
            "instance_id": task["instance_id"],
            "seed": seed,
            "status": "completed",
            "reward": 0,
        },
    )
    from frognano.wandb import WandbTracker

    class Tracker(WandbTracker):
        def __init__(self, config, **kwargs):
            assert kwargs["jobs_total"] == 6
            assert kwargs["seeds_per_task"] == 3
            expected = {("task-a", 0): original[1]} if resume else {}
            assert kwargs["initial_results"] == expected
            self.jobs_total = kwargs["jobs_total"]
            self.seeds_per_task = kwargs["seeds_per_task"]
            self.results = dict(kwargs["initial_results"])
            self.log()

        def log(self):
            metrics = self._metrics()
            assert metrics["overall/pass_at_3_total_tasks"] == 2
            assert metrics["overall/pass_at_3_percent"] == (50 if resume else 0)

        def finish(self, summary):
            self.log()
            assert len(self.results) == 6

    monkeypatch.setattr(runner, "WandbTracker", Tracker)

    summary = runner.run_evaluation(config)

    assert summary["jobs_recorded"] == 6
    assert summary["resolved"] == (1 if resume else 0)


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
