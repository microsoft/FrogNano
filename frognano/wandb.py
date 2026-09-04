from __future__ import annotations

import importlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from frognano.config import WandbConfig

_EXIT_REASONS = (
    "agent",
    "max_turns",
    "max_time",
    "max_context_len",
    "max_tokens_per_turn",
    "llm_query_error",
    "tool_error",
    "cancelled",
    "infrastructure_error",
    "internal_error",
    "unknown",
)


class WandbTracker:
    def __init__(
        self,
        config: WandbConfig,
        *,
        run_config: dict[str, Any],
        output_dir: Path,
        jobs_total: int,
        seeds_per_task: int,
        initial_results: dict[tuple[str, int], dict[str, Any]],
    ) -> None:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"W&B API key environment variable is unset: {config.api_key_env}"
            )
        try:
            wandb = importlib.import_module("wandb")
        except ModuleNotFoundError as exc:
            if exc.name != "wandb":
                raise
            raise RuntimeError(
                "W&B tracking requires the optional dependency; "
                "install FrogNano with the wandb extra"
            ) from exc

        self.wandb = wandb
        self.output_dir = output_dir
        self.jobs_total = jobs_total
        self.seeds_per_task = seeds_per_task
        self.results = dict(initial_results)
        self.state_path = output_dir / "wandb.json"
        run_id = config.run_id or self._stored_run_id() or uuid.uuid4().hex
        self.run = wandb.init(
            entity=config.entity,
            project=config.project,
            name=config.name,
            group=config.group,
            tags=list(config.tags),
            id=run_id,
            resume="allow",
            job_type="evaluation",
            config=run_config,
            settings=wandb.Settings(base_url=config.base_url, api_key=api_key),
        )
        self._write_state(config, run_id)
        self.log()

    def update(self, result: dict[str, Any]) -> None:
        key = (str(result["instance_id"]), int(result.get("seed", 0)))
        self.results[key] = result
        self.log()

    def log(self) -> None:
        self.run.log(self._metrics())

    def _metrics(self) -> dict[str, int | float]:
        completed = [
            result
            for result in self.results.values()
            if result.get("status") == "completed"
        ]
        resolved = sum(float(result.get("reward") or 0) >= 1 for result in completed)
        recorded = len(self.results)
        failed = recorded - len(completed)
        metrics: dict[str, int | float] = {
            "overall/completed_percent": self._percent(
                len(completed),
                self.jobs_total,
            ),
            "overall/error_percent": self._percent(failed, self.jobs_total),
            "overall/resolve_rate_percent": self._percent(
                resolved,
                len(completed),
            ),
            "overall/unresolve_rate_percent": self._percent(
                len(completed) - resolved,
                len(completed),
            ),
        }
        for seed in range(self.seeds_per_task):
            seed_results = [
                result
                for result in self.results.values()
                if int(result.get("seed", 0)) == seed
            ]
            seed_completed = [
                result for result in seed_results if result.get("status") == "completed"
            ]
            seed_resolved = sum(
                float(result.get("reward") or 0) >= 1 for result in seed_completed
            )
            prefix = f"seed-{seed}"
            metrics[f"{prefix}/error_percent"] = self._percent(
                len(seed_results) - len(seed_completed),
                self.jobs_total // self.seeds_per_task,
            )
            metrics[f"{prefix}/resolve_rate_percent"] = self._percent(
                seed_resolved,
                len(seed_completed),
            )
            metrics[f"{prefix}/unresolve_rate_percent"] = self._percent(
                len(seed_completed) - seed_resolved,
                len(seed_completed),
            )
        return metrics

    def finish(self, summary: dict[str, Any]) -> None:
        self.run.summary.update(self._metrics())
        self._log_total_charts()
        for filename in ("config.json", "results.jsonl", "summary.json"):
            path = self.output_dir / filename
            if path.is_file():
                self.run.save(str(path), base_path=str(self.output_dir), policy="end")
        self.run.finish()

    def _log_total_charts(self) -> None:
        outcomes = Counter(self._outcome(result) for result in self.results.values())
        self._log_bar_chart(
            "overall/result_totals",
            "Result totals",
            "result",
            outcomes,
            ("resolved", "unresolved", "error"),
        )
        stop_reasons = Counter(
            self._exit_reason(result) for result in self.results.values()
        )
        self._log_bar_chart(
            "overall/stop_reason_totals",
            "Stop reason totals",
            "stop_reason",
            stop_reasons,
            _EXIT_REASONS,
        )

    def _log_bar_chart(
        self,
        chart_key: str,
        title: str,
        category_name: str,
        counts: Counter[str],
        categories: tuple[str, ...],
    ) -> None:
        ordered = list(dict.fromkeys((*categories, *sorted(counts))))
        table = self.wandb.Table(
            columns=[category_name, "count"],
            data=[
                [category, counts[category]]
                for category in ordered
                if counts[category] or category == "error"
            ],
        )
        chart = self.wandb.plot.bar(
            table,
            category_name,
            "count",
            title=title,
        )
        table_key = f"{chart_key}_table"
        self._remove_summary_key(table_key)
        self.run.define_metric(
            table_key,
            hidden=True,
            summary="none",
            overwrite=True,
        )
        self.run.log({chart_key: chart})

    def _remove_summary_key(self, key: str) -> None:
        if key not in self.run.summary.keys():
            return
        if isinstance(self.run.summary, dict):
            del self.run.summary[key]
            return
        self.run.summary.__delattr__(key)

    @staticmethod
    def _percent(numerator: int, denominator: int) -> float:
        return 100.0 * numerator / denominator if denominator else 0.0

    @staticmethod
    def _exit_reason(result: dict[str, Any]) -> str:
        reason = str(result.get("exit_reason") or "").strip()
        if reason:
            return reason
        return (
            "infrastructure_error" if result.get("status") != "completed" else "unknown"
        )

    @staticmethod
    def _outcome(result: dict[str, Any]) -> str:
        if result.get("status") != "completed":
            return "error"
        return "resolved" if float(result.get("reward") or 0) >= 1 else "unresolved"

    def _stored_run_id(self) -> str | None:
        if not self.state_path.is_file():
            return None
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        run_id = str(value.get("run_id") or "").strip()
        return run_id or None

    def _write_state(self, config: WandbConfig, run_id: str) -> None:
        self.state_path.write_text(
            json.dumps(
                {
                    "base_url": config.base_url,
                    "entity": config.entity,
                    "project": config.project,
                    "run_id": run_id,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
