import json
import sys
from types import SimpleNamespace

from frognano.config import WandbConfig
from frognano.wandb import WandbTracker


def test_wandb_tracker_logs_progress_and_resumes_run(tmp_path, monkeypatch) -> None:
    runs = []

    class Run:
        def __init__(self):
            self.logs = []
            self.summary = {}
            self.saved = []
            self.finished = False
            self.defined_metrics = []
            self.no_summary = set()

        def log(self, metrics):
            self.logs.append(metrics)
            for key, value in metrics.items():
                if (
                    isinstance(value, dict)
                    and "table" in value
                    and f"{key}_table" not in self.no_summary
                ):
                    self.summary[f"{key}_table"] = value["table"]

        def define_metric(self, name, **kwargs):
            self.defined_metrics.append((name, kwargs))
            if kwargs.get("summary") == "none":
                self.no_summary.add(name)

        def save(self, path, *, base_path, policy):
            self.saved.append((path, base_path, policy))

        def finish(self):
            self.finished = True

    class Table:
        def __init__(self, *, columns, data):
            self.columns = columns
            self.data = data

    def init(**kwargs):
        run = Run()
        run.init_kwargs = kwargs
        runs.append(run)
        return run

    fake_wandb = SimpleNamespace(
        Settings=lambda **kwargs: kwargs,
        Table=Table,
        plot=SimpleNamespace(
            bar=lambda table, x, y, *, title: {
                "table": table,
                "x": x,
                "y": y,
                "title": title,
            },
        ),
        init=init,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    config = WandbConfig(
        entity="research",
        project="evaluations",
        base_url="https://wandb.example.test",
    )
    initial = {
        ("task-a", 0): {
            "instance_id": "task-a",
            "seed": 0,
            "status": "completed",
            "reward": 1,
            "exit_reason": "agent",
        }
    }

    tracker = WandbTracker(
        config,
        run_config={"dataset": "test"},
        output_dir=tmp_path,
        jobs_total=4,
        seeds_per_task=2,
        initial_results=initial,
    )
    tracker.update(
        {
            "instance_id": "task-b",
            "seed": 1,
            "status": "failed",
            "reward": 0,
            "exit_reason": "llm_query_error",
            "error": "model unavailable",
        }
    )
    for filename in ("config.json", "results.jsonl", "summary.json"):
        (tmp_path / filename).write_text("{}\n", encoding="utf-8")
    tracker.finish({"resolved": 1})

    run = runs[0]
    assert run.init_kwargs["resume"] == "allow"
    assert run.init_kwargs["settings"]["base_url"] == "https://wandb.example.test"
    progress = run.logs[1]
    assert list(progress)[:3] == [
        "overall/completed_percent",
        "overall/error_percent",
        "overall/resolve_rate_percent",
    ]
    assert progress["overall/completed_percent"] == 25.0
    assert progress["overall/error_percent"] == 25.0
    assert progress["overall/execution_task_resolve_percent"] == 50.0
    assert progress["overall/execution_task_unresolve_percent"] == 0.0
    assert progress["overall/execution_task_error_percent"] == 50.0
    assert progress["overall/valid_task_resolve_percent"] == 100.0
    assert progress["overall/valid_task_unresolve_percent"] == 0.0
    assert progress["overall/resolve_rate_percent"] == 100.0
    assert progress["seed-0/resolve_rate_percent"] == 100.0
    assert progress["seed-1/error_percent"] == 50.0
    assert set(progress) == {
        "overall/completed_percent",
        "overall/error_percent",
        "overall/execution_task_resolve_percent",
        "overall/execution_task_unresolve_percent",
        "overall/execution_task_error_percent",
        "overall/valid_task_resolve_percent",
        "overall/valid_task_unresolve_percent",
        "overall/resolve_rate_percent",
        "overall/unresolve_rate_percent",
        "seed-0/error_percent",
        "seed-0/execution_task_resolve_percent",
        "seed-0/execution_task_unresolve_percent",
        "seed-0/execution_task_error_percent",
        "seed-0/valid_task_resolve_percent",
        "seed-0/valid_task_unresolve_percent",
        "seed-0/resolve_rate_percent",
        "seed-0/unresolve_rate_percent",
        "seed-1/error_percent",
        "seed-1/execution_task_resolve_percent",
        "seed-1/execution_task_unresolve_percent",
        "seed-1/execution_task_error_percent",
        "seed-1/valid_task_resolve_percent",
        "seed-1/valid_task_unresolve_percent",
        "seed-1/resolve_rate_percent",
        "seed-1/unresolve_rate_percent",
    }
    assert (
        "overall/result_totals_table",
        {"hidden": True, "summary": "none", "overwrite": True},
    ) in run.defined_metrics
    result_totals = run.logs[2]["overall/result_totals"]
    assert ["resolved", 1] in result_totals["table"].data
    assert ["error", 1] in result_totals["table"].data
    stop_totals = run.logs[3]["overall/stop_reason_totals"]
    assert ["agent", 1] in stop_totals["table"].data
    assert ["llm_query_error", 1] in stop_totals["table"].data
    assert run.summary["overall/resolve_rate_percent"] == 100.0
    assert "resolved" not in run.summary
    assert "overall/result_totals_table" not in run.summary
    assert len(run.saved) == 3
    assert run.finished is True

    state = json.loads((tmp_path / "wandb.json").read_text(encoding="utf-8"))
    second = WandbTracker(
        config,
        run_config={"dataset": "test"},
        output_dir=tmp_path,
        jobs_total=4,
        seeds_per_task=2,
        initial_results=initial,
    )

    assert runs[1].init_kwargs["id"] == state["run_id"]
    second.finish({})
