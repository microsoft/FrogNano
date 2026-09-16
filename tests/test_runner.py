import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize(
    "resume,error_filter",
    [(True, None), (True, "APIConnectionError"), (False, None)],
)
@pytest.mark.parametrize("workers_per_seed", [None, 1])
def test_wandb_initial_results_match_selected_tasks_and_seeds(
    tmp_path, monkeypatch, resume, error_filter, workers_per_seed
):
    config = replace(
        _config(tmp_path),
        seeds_per_task=3,
        resume=resume,
        resume_retry_error_contains=error_filter,
        max_workers=3,
        max_workers_per_seed=workers_per_seed,
        wandb=WandbConfig(entity="research", project="evaluations"),
    )
    original = [
        {
            "instance_id": task,
            "seed": seed,
            "status": status,
            "reward": reward,
            "error": error,
        }
        for task, seed, status, reward, error in [
            ("task-a", 0, "failed", 0, "APIConnectionError"),
            ("task-a", 0, "completed", 1, None),
            ("task-a", 3, "completed", 1, None),
            ("unselected-task", 0, "completed", 1, None),
            ("task-b", 0, "failed", 0, "APIConnectionError"),
            ("task-b", 1, "failed", 0, "setup failed"),
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
    calls = []

    def run_job(config, task, seed, *args):
        calls.append((task["instance_id"], seed))
        return {
            "instance_id": task["instance_id"],
            "seed": seed,
            "status": "completed",
            "reward": 0,
        }

    monkeypatch.setattr(runner, "_run_job", run_job)
    from frognano.wandb import WandbTracker

    class Tracker(WandbTracker):
        def __init__(self, config, **kwargs):
            assert kwargs["jobs_total"] == 6
            assert kwargs["seeds_per_task"] == 3
            expected = (
                {
                    ("task-a", 0): original[1],
                    ("task-b", 0): original[4],
                    ("task-b", 1): original[5],
                }
                if resume
                else {}
            )
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
    assert summary["jobs_failed"] == (1 if error_filter else 0)
    expected_calls = {
        (task, seed) for task in ("task-a", "task-b") for seed in range(3)
    }
    if resume:
        expected_calls.remove(("task-a", 0))
    if error_filter:
        expected_calls.remove(("task-b", 1))
    assert set(calls) == expected_calls
    assert len(calls) == len(expected_calls)


@pytest.mark.parametrize("error_filter", [None, "APIConnectionError"])
def test_run_evaluation_selectively_resumes_failed_pairs(
    tmp_path, monkeypatch, error_filter
) -> None:
    config = replace(
        _config(tmp_path),
        seeds_per_task=2,
        resume_retry_error_contains=error_filter,
    )
    original = [
        {
            "instance_id": task,
            "seed": seed,
            "status": status,
            "reward": reward,
            "exit_reason": reason,
            "error": error,
        }
        for task, seed, status, reward, reason, error in [
            ("task-a", 0, "failed", 0, "llm_query_error", "APIConnectionError"),
            ("task-a", 0, "completed", 1, "agent", None),
            ("task-a", 1, "completed", 0, "max_time", None),
            ("task-b", 0, "failed", 0, "llm_query_error", "APIConnectionError"),
            ("task-b", 1, "failed", 0, "infrastructure_error", "setup failed"),
            ("task-c", 0, "failed", 0, "cancelled", "evaluation cancelled"),
            ("task-c", 1, "failed", 0, "internal_error", None),
            ("task-d", 1, "failed", 0, "llm_query_error", "apiconnectionerror"),
        ]
    ]
    results_path = tmp_path / "results.jsonl"
    original_text = "".join(json.dumps(row) + "\n" for row in original)
    results_path.write_text(original_text, encoding="utf-8")
    monkeypatch.setenv("TEST_MODEL_KEY", "secret")
    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda *args, **kwargs: [
            {"instance_id": f"task-{suffix}"} for suffix in "abcd"
        ],
    )
    calls = []

    def run_job(config, task, seed, api_key, stop_event):
        calls.append((task["instance_id"], seed))
        return {
            "instance_id": task["instance_id"],
            "seed": seed,
            "status": "completed",
            "reward": 0,
        }

    monkeypatch.setattr(runner, "_run_job", run_job)

    summary = runner.run_evaluation(config)

    expected = {("task-b", 0), ("task-c", 0), ("task-d", 0)}
    if error_filter is None:
        expected |= {("task-b", 1), ("task-c", 1), ("task-d", 1)}
    assert set(calls) == expected
    assert len(calls) == len(expected)
    assert results_path.read_text(encoding="utf-8").startswith(original_text)
    latest = runner._latest_results(results_path)
    assert latest[("task-a", 0)] == original[1]
    assert latest[("task-a", 1)] == original[2]
    if error_filter is not None:
        for row in (original[4], original[6], original[7]):
            assert latest[(row["instance_id"], row["seed"])] == row
    assert summary["jobs_total"] == 8
    assert summary["jobs_recorded"] == 8
    assert summary["jobs_completed"] == 2 + len(expected)
    assert summary["jobs_failed"] == 6 - len(expected)
    assert summary["resolved"] == 1
    assert summary["resolve_rate"] == 1 / 8
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["resume_retry_error_contains"] == error_filter


@pytest.mark.parametrize("workers_per_seed", [None, 2])
def test_run_evaluation_worker_pools(tmp_path, monkeypatch, workers_per_seed) -> None:
    config = replace(
        _config(tmp_path),
        seeds_per_task=3,
        max_workers=6,
        max_workers_per_seed=workers_per_seed,
    )
    monkeypatch.setenv("TEST_MODEL_KEY", "secret")
    monkeypatch.setattr(
        runner,
        "load_dataset",
        lambda *args, **kwargs: [
            {"instance_id": f"task-{index}"} for index in range(6)
        ],
    )
    pools = []

    class Pool(ThreadPoolExecutor):
        def __init__(self, *, max_workers):
            super().__init__(max_workers=max_workers)
            self.seeds = set()
            self.limit = max_workers
            pools.append(self)

        def submit(self, fn, config, task, seed, *args):
            self.seeds.add(seed)
            return super().submit(fn, config, task, seed, *args)

    barrier = threading.Barrier(6, timeout=10)
    lock = threading.Lock()
    active = [0, 0, 0]
    peaks = [0, 0, 0]
    total_peak = 0

    def run_job(config, task, seed, api_key, stop_event):
        nonlocal total_peak
        with lock:
            active[seed] += 1
            peaks[seed] = max(peaks[seed], active[seed])
            total_peak = max(total_peak, sum(active))
        try:
            barrier.wait()
            return {
                "instance_id": task["instance_id"],
                "seed": seed,
                "status": "completed",
                "reward": 1,
            }
        finally:
            with lock:
                active[seed] -= 1

    monkeypatch.setattr(runner, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(runner, "_run_job", run_job)

    summary = runner.run_evaluation(config)

    assert summary["jobs_completed"] == 18
    assert total_peak == 6
    if workers_per_seed is None:
        assert len(pools) == 1
        assert pools[0].limit == 6
        assert pools[0].seeds == {0, 1, 2}
    else:
        assert len(pools) == 3
        assert [pool.limit for pool in pools] == [2, 2, 2]
        assert [pool.seeds for pool in pools] == [{0}, {1}, {2}]
        assert peaks == [2, 2, 2]


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


@pytest.mark.parametrize("time_budget", [None, 3000, 10800])
def test_run_job_retries_then_writes_trajectory(
    tmp_path, monkeypatch, time_budget
) -> None:
    config = replace(_config(tmp_path), max_total_time_sec=time_budget)
    task = {
        "instance_id": "task-a",
        "agent_timeout_sec": 300,
        "source": {"revision": "abc"},
    }
    expected_timeout = time_budget if time_budget is not None else 300
    attempts = []

    class Runtime:
        def __init__(self, task, config, *, run_id):
            assert task["agent_timeout_sec"] == expected_timeout
            attempts.append(run_id)
            if len(attempts) == 1:
                raise RuntimeError("transient")

        def compute_reward(self):
            return 1.0, "passed"

        def close(self):
            pass

    class Agent:
        def __init__(self, config):
            assert config.max_total_time_sec == expected_timeout

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
    assert task["agent_timeout_sec"] == 300
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


def _offline_harbor_runtime(monkeypatch, rewards, query_errors=None):
    created = []
    closed = []
    pending_rewards = iter(rewards)

    class Runtime(runner.KubernetesTaskRuntime):
        def __init__(self, task, config, *, run_id):
            self.task = task
            self.reward = next(pending_rewards)
            self.commands = []
            created.append(self)

        def run(self, command, **kwargs):
            self.commands.append(command)
            if command == "bash /tests/test.sh":
                return "verifier output", 0 if self.reward > 0 else 1
            if "reward.json" in command:
                return json.dumps({"reward": self.reward}), 0
            return "", 0

        def copy_to_container(self, *args, **kwargs):
            pass

        def _has_git_checkout(self):
            return False

        def close(self):
            closed.append(self)

    class Environment:
        def __init__(self, runtime):
            self.runtime = runtime

        def instruction(self):
            return "Fix the task."

        def patch(self):
            return "diff --git a/file.py b/file.py"

        def compute_reward(self):
            return self.runtime.compute_reward()

    class Client:
        def __init__(self, config):
            assert config.max_tokens_per_turn == 8192
            self.error = query_errors[len(created) - 1] if query_errors else None

        def complete(self, messages, *, tools):
            if self.error is not None:
                raise self.error
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Done.", tool_calls=[]),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(total_tokens=1),
            )

    monkeypatch.setattr(runner, "KubernetesTaskRuntime", Runtime)
    monkeypatch.setattr(runner, "LeafEnvironment", Environment)
    monkeypatch.setattr("frognano.harness.leaf.agent.OpenAIChatClient", Client)
    task = {
        "instance_id": "task-a",
        "agent_timeout_sec": 300,
        "verifier_timeout_sec": 10,
        "repo_path": "/testbed",
        "tests_dir": "/host/tests",
        "source": {},
    }
    return task, created, closed


@pytest.mark.parametrize("reward", [0, 1])
def test_run_job_grades_context_limited_work_without_retry(
    tmp_path, monkeypatch, reward
) -> None:
    task, created, closed = _offline_harbor_runtime(
        monkeypatch, [reward], [ValueError("context_length_exceeded")]
    )

    result = runner._run_job(_config(tmp_path), task, 0, "key")

    assert result["status"] == "completed"
    assert result["exit_reason"] == "max_context_len"
    assert result["reward"] == reward
    assert result["attempt"] == 1
    assert len(created) == len(closed) == 1
    assert created[0].commands.count("bash /tests/test.sh") == 1
    trajectory = json.loads(Path(result["trajectory"]).read_text())
    assert trajectory["error"] is None
    assert trajectory["output_patch"] == "diff --git a/file.py b/file.py"
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    assert runner._processed_jobs(results_path) == {("task-a", 0)}


@pytest.mark.parametrize("rewards", [(-1, 1), (-1, 0), (-1, -1)])
def test_run_job_retries_negative_verifier_rewards(
    tmp_path, monkeypatch, rewards
) -> None:
    task, created, closed = _offline_harbor_runtime(monkeypatch, rewards)

    result = runner._run_job(_config(tmp_path), task, 0, "key")

    assert result["attempt"] == 2
    assert len(created) == len(closed) == 2
    assert all(
        runtime.commands.count("bash /tests/test.sh") == 1 for runtime in created
    )
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    if rewards[-1] >= 0:
        assert result["status"] == "completed"
        assert result["reward"] == rewards[-1]
        assert runner._processed_jobs(results_path) == {("task-a", 0)}
    else:
        assert result["status"] == "failed"
        assert result["exit_reason"] == "infrastructure_error"
        assert result["error"].count("negative reward") == 2
        assert runner._processed_jobs(results_path) == set()


def test_run_job_reports_verifier_failure_after_query_retry(
    tmp_path, monkeypatch
) -> None:
    task, created, closed = _offline_harbor_runtime(
        monkeypatch, [-1, -1], [TimeoutError("model unavailable"), None]
    )

    result = runner._run_job(_config(tmp_path), task, 0, "key")

    assert result["status"] == "failed"
    assert result["exit_reason"] == "infrastructure_error"
    assert "model unavailable" in result["error"]
    assert "negative reward" in result["error"]
    assert len(created) == len(closed) == 2
