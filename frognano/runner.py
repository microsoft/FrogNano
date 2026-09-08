from __future__ import annotations

import json
import logging
import os
import random
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from frognano.config import EvalConfig
from frognano.datasets import load_dataset
from frognano.datasets.harbor import apply_image_digest_lock
from frognano.harness.leaf import LeafAgent, LeafConfig
from frognano.harness.leaf.environment import LeafEnvironment
from frognano.runtimes import KubernetesTaskRuntime
from frognano.wandb import WandbTracker

logger = logging.getLogger(__name__)


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    api_key = os.environ.get(config.model.api_key_env)
    if not api_key:
        raise ValueError(
            f"model API key environment variable is unset: "
            f"{config.model.api_key_env}"
        )
    output_dir = config.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    _write_json(output_dir / "config.json", _config_dict(config))

    tasks = load_dataset(
        config.dataset,
        cache_dir=config.cache_dir,
        image_registry=config.kubernetes.image_registry,
        task_ids=config.task_ids,
        limit=None,
    )
    random.Random(config.seed).shuffle(tasks)
    if config.num_tasks is not None:
        tasks = tasks[: config.num_tasks]
    if config.image_digest_lock is not None:
        tasks = apply_image_digest_lock(tasks, config.image_digest_lock)
    jobs = [(task, seed) for task in tasks for seed in range(config.seeds_per_task)]
    latest_results = _latest_results(results_path)
    processed = _processed_jobs(results_path) if config.resume else set()
    tracker = (
        WandbTracker(
            config.wandb,
            run_config=_config_dict(config),
            output_dir=output_dir,
            jobs_total=len(jobs),
            seeds_per_task=config.seeds_per_task,
            initial_results=latest_results,
        )
        if config.wandb is not None
        else None
    )
    jobs = [
        (task, seed)
        for task, seed in jobs
        if (str(task["instance_id"]), seed) not in processed
    ]

    write_lock = threading.Lock()
    stop_event = threading.Event()
    previous_handlers = _install_stop_handlers(stop_event)
    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = {
                pool.submit(
                    _run_job,
                    config,
                    task,
                    seed,
                    api_key,
                    stop_event,
                ): (
                    str(task["instance_id"]),
                    seed,
                )
                for task, seed in jobs
            }
            for future in as_completed(futures):
                instance_id, seed = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "instance_id": instance_id,
                        "seed": seed,
                        "status": "failed",
                        "reward": 0.0,
                        "exit_reason": "internal_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                with write_lock:
                    with results_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result, default=str) + "\n")
                    if tracker is not None:
                        tracker.update(result)
    finally:
        _restore_stop_handlers(previous_handlers)

    expected_jobs = {
        (str(task["instance_id"]), seed)
        for task in tasks
        for seed in range(config.seeds_per_task)
    }
    results = [
        row
        for key, row in _latest_results(results_path).items()
        if key in expected_jobs
    ]
    completed = [row for row in results if row.get("status") == "completed"]
    resolved = sum(float(row.get("reward") or 0) >= 1 for row in completed)
    jobs_total = len(expected_jobs)
    summary = {
        "dataset": config.dataset,
        "tasks_selected": len(tasks),
        "jobs_total": jobs_total,
        "jobs_recorded": len(results),
        "jobs_completed": len(completed),
        "jobs_failed": len(results) - len(completed),
        "resolved": resolved,
        "unresolved": len(completed) - resolved,
        "resolve_rate": resolved / jobs_total if jobs_total else 0.0,
        "error_rate": (
            (jobs_total - len(completed)) / jobs_total if jobs_total else 0.0
        ),
        "duration_sec": time.time() - started,
    }
    _write_json(output_dir / "summary.json", summary)
    if tracker is not None:
        tracker.finish(summary)
    return summary


def _run_job(
    config: EvalConfig,
    task: dict[str, Any],
    seed: int,
    api_key: str,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    instance_id = str(task["instance_id"])
    task_dir = config.output_dir.expanduser() / "trajectories" / instance_id
    task_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    last_exit_reason = "infrastructure_error"
    for attempt in range(1, config.max_attempts + 1):
        if stop_event is not None and stop_event.is_set():
            return _cancelled_result(task, seed, attempt)
        runtime: KubernetesTaskRuntime | None = None
        try:
            run_id = f"{seed}-{attempt}-{uuid.uuid4().hex[:6]}"
            runtime = KubernetesTaskRuntime(
                task,
                config.kubernetes,
                run_id=run_id,
            )
            environment = LeafEnvironment(runtime)
            max_total_time = (
                config.max_total_time_sec
                if config.max_total_time_sec is not None
                else int(task["agent_timeout_sec"])
            )
            agent = LeafAgent(
                LeafConfig(
                    model=config.model.name,
                    base_url=config.model.base_url,
                    api_key=api_key,
                    temperature=config.model.temperature,
                    max_tokens_per_turn=config.model.max_tokens_per_turn,
                    timeout_sec=config.model.timeout_sec,
                    max_retries=config.model.max_retries,
                    parallel_tool_calls=config.model.parallel_tool_calls,
                    max_steps=config.max_steps,
                    max_context_tokens=config.max_context_tokens,
                    max_total_time_sec=max_total_time,
                    extra_body=config.model.extra_body,
                )
            )
            trajectory = agent.run(
                environment,
                instance_id=instance_id,
                seed=seed,
                checkpoint_callback=lambda value: _write_json(
                    task_dir / f"trajectory_seed-{seed}.partial.json",
                    value,
                ),
                stop_event=stop_event,
            )
            if trajectory["exit_reason"] == "cancelled":
                return _cancelled_result(task, seed, attempt)
            if trajectory["exit_reason"] in {
                "llm_query_error",
                "tool_error",
                "unknown",
            }:
                last_exit_reason = trajectory["exit_reason"]
                raise RuntimeError(
                    "retryable Leaf failure: "
                    f"{trajectory['exit_reason']}: {trajectory.get('error')}"
                )
            reward, test_output = runtime.compute_reward()
            trajectory["reward"] = reward
            trajectory["test_output"] = test_output
            trajectory["attempt"] = attempt
            _write_json(
                task_dir / f"trajectory_seed-{seed}.json",
                trajectory,
            )
            (task_dir / f"trajectory_seed-{seed}.partial.json").unlink(missing_ok=True)
            patch = str(trajectory.get("output_patch") or "")
            if patch:
                (task_dir / f"generated_seed-{seed}.patch").write_text(
                    patch,
                    encoding="utf-8",
                )
            return {
                "instance_id": instance_id,
                "seed": seed,
                "attempt": attempt,
                "status": "completed",
                "reward": reward,
                "exit_reason": trajectory["exit_reason"],
                "trajectory": str(task_dir / f"trajectory_seed-{seed}.json"),
                "source": task["source"],
            }
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            logger.exception(
                "Evaluation failed for %s seed %s attempt %s",
                instance_id,
                seed,
                attempt,
            )
        finally:
            if runtime is not None:
                runtime.close()
    return {
        "instance_id": instance_id,
        "seed": seed,
        "attempt": config.max_attempts,
        "status": "failed",
        "reward": 0.0,
        "exit_reason": last_exit_reason,
        "error": "\n".join(errors),
        "source": task["source"],
    }


def _cancelled_result(
    task: dict[str, Any],
    seed: int,
    attempt: int,
) -> dict[str, Any]:
    return {
        "instance_id": str(task["instance_id"]),
        "seed": seed,
        "attempt": attempt,
        "status": "failed",
        "reward": 0.0,
        "exit_reason": "cancelled",
        "error": "evaluation cancelled",
        "source": task["source"],
    }


def _install_stop_handlers(
    stop_event: threading.Event,
) -> dict[signal.Signals, Any]:
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous = {}
    signal_count = 0

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal signal_count
        signal_count += 1
        if signal_count == 1:
            stop_event.set()
            return
        for signal_name, handler in previous.items():
            if handler is not None:
                signal.signal(signal_name, handler)
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)
    return previous


def _restore_stop_handlers(previous: dict[signal.Signals, Any]) -> None:
    for signum, handler in previous.items():
        if handler is not None:
            signal.signal(signum, handler)


def _processed_jobs(path: Path) -> set[tuple[str, int]]:
    return {
        key
        for key, row in _latest_results(path).items()
        if row.get("status") == "completed"
    }


def _latest_results(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _read_results(path):
        key = (str(row["instance_id"]), int(row.get("seed", 0)))
        latest[key] = row
    return latest


def _read_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _config_dict(config: EvalConfig) -> dict[str, Any]:
    value = asdict(config)
    value["output_dir"] = str(config.output_dir)
    value["cache_dir"] = str(config.cache_dir)
    if config.image_digest_lock is not None:
        value["image_digest_lock"] = str(config.image_digest_lock)
    return value
