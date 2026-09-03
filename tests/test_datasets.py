from pathlib import Path

import pytest

from frognano.datasets import DatasetSource, get_dataset, harbor
from frognano.datasets.harbor import (
    load_harbor_dataset,
    materialize_source,
    parse_dockerfile,
    parse_harbor_task,
)


def _task_dir(tmp_path: Path) -> Path:
    task = tmp_path / "owner__repo-1"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        """
[agent]
timeout_sec = 300
network_mode = "public"

[verifier]
timeout_sec = 600

[environment]
cpus = 2
memory = "4G"
storage = "10G"
""",
        encoding="utf-8",
    )
    (task / "instruction.md").write_text("Fix the bug.\n", encoding="utf-8")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (task / "environment" / "Dockerfile").write_text(
        """
FROM swebench/example:latest
WORKDIR /testbed
ENV EXAMPLE=value
RUN mkdir -p /logs
""",
        encoding="utf-8",
    )
    return task


def test_swebench_verified_is_registered() -> None:
    source, _ = get_dataset("swebench_verified")

    assert source.revision == "86723674f04e4209ac479d0fb75d9d9f44b4377e"
    assert source.default_image_registry is None
    assert source.verifier_success_marker == "SWEBench results ends here"


def test_parse_harbor_task_normalizes_runtime_fields(tmp_path) -> None:
    source = DatasetSource(
        name="test",
        display_name="Test",
        source_url="https://example.test/tasks.git",
        revision="a" * 40,
        subpath="tasks",
        pod_prefix="test",
        agent_network_mode="public",
        verifier_network_mode="public",
    )

    task = parse_harbor_task(
        _task_dir(tmp_path),
        source=source,
        image_registry="registry.test",
    )

    assert task["instance_id"] == "owner__repo-1"
    assert task["docker_image"] == "registry.test/swebench/example:latest"
    assert task["repo_path"] == "/testbed"
    assert task["setup_commands"] == ["mkdir -p /logs"]
    assert task["environment_env"] == {"EXAMPLE": "value"}
    assert task["agent_network_mode"] == "public"


def test_parse_harbor_task_qualifies_single_segment_image(tmp_path) -> None:
    task_dir = _task_dir(tmp_path)
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM ubuntu\nWORKDIR /testbed\n",
        encoding="utf-8",
    )
    source = DatasetSource(
        name="test",
        display_name="Test",
        source_url="https://example.test/tasks.git",
        revision="a" * 40,
        subpath="tasks",
        pod_prefix="test",
    )

    task = parse_harbor_task(
        task_dir,
        source=source,
        image_registry="registry.test",
    )

    assert task["docker_image"] == "registry.test/ubuntu"


def test_parse_dockerfile_rejects_multiple_stages(tmp_path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM one\nFROM two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="single-stage"):
        parse_dockerfile(dockerfile)


def test_get_dataset_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        get_dataset("missing")


def test_materialize_source_uses_immutable_cached_checkout(
    tmp_path, monkeypatch
) -> None:
    source = DatasetSource(
        name="test",
        display_name="Test",
        source_url="https://example.test/tasks.git",
        revision="b" * 40,
        subpath="datasets/test",
        pod_prefix="test",
    )
    calls = []

    def fake_git(root, *args, env, capture=False):
        calls.append(args)
        if args[0] == "init":
            (root / ".git").mkdir()
        if args[:2] == ("checkout", "--quiet"):
            (root / source.subpath).mkdir(parents=True)
        return source.revision if args[:2] == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(harbor, "_git", fake_git)

    selected = materialize_source(source, tmp_path)
    again = materialize_source(source, tmp_path)

    assert selected == again
    assert selected.is_dir()
    assert calls.count(("rev-parse", "HEAD")) == 1


def test_load_harbor_dataset_filters_tasks(tmp_path, monkeypatch) -> None:
    root = tmp_path / "tasks"
    root.mkdir()
    _task_dir(root)
    source = DatasetSource(
        name="test",
        display_name="Test",
        source_url="https://example.test/tasks.git",
        revision="c" * 40,
        subpath="tasks",
        pod_prefix="test",
    )
    monkeypatch.setattr(harbor, "materialize_source", lambda *args: root)

    tasks = load_harbor_dataset(
        source,
        cache_dir=tmp_path,
        image_registry=None,
        task_ids=("owner__repo-1",),
        limit=1,
    )

    assert [task["instance_id"] for task in tasks] == ["owner__repo-1"]
    with pytest.raises(ValueError, match="unknown task IDs"):
        load_harbor_dataset(
            source,
            cache_dir=tmp_path,
            image_registry=None,
            task_ids=("missing",),
            limit=None,
        )


def test_parse_dockerfile_rejects_unsupported_instruction(tmp_path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM image\nCOPY . /app\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported"):
        parse_dockerfile(dockerfile)
