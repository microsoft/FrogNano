import json
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import pytest

from frognano.datasets import DatasetSource, get_dataset, harbor, load_dataset
from frognano.datasets.harbor import (
    apply_image_digest_lock,
    load_harbor_dataset,
    materialize_source,
    parse_dockerfile,
    parse_harbor_task,
)
from frognano.datasets.patch_eval_verified import load_patch_eval_verified


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


@pytest.mark.parametrize(
    ("name", "revision", "subpath"),
    [
        (
            "swebench_pro",
            "c8e8f3fac7097accaacf261d74c3d6f441de45b1",
            "datasets/swebenchpro",
        ),
        (
            "patch_eval_verified",
            "b43285cdde80cc04608d5f1178a330b740c91c2d",
            "patcheval/datasets",
        ),
        (
            "terminal_bench_2_verified",
            "7bbd2fef45db2f9ee9d9e3aae8f7253f56bf4e2b",
            ".",
        ),
    ],
)
def test_additional_datasets_are_registered(name, revision, subpath) -> None:
    source, _ = get_dataset(name)

    assert source.revision == revision
    assert source.subpath == subpath


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


@pytest.mark.parametrize("image_field", [None, "docker_image", "base_image"])
@pytest.mark.parametrize("image_registry", [None, "mirror.example:5000/prefix"])
def test_harbor_registry_selection_preserves_digest(
    tmp_path, image_field, image_registry
) -> None:
    image = "ghcr.io/owner/image@sha256:" + "a" * 64
    task_dir = _task_dir(tmp_path)
    (task_dir / "environment/Dockerfile").write_text(
        f"FROM {image}\nWORKDIR /testbed\n", encoding="utf-8"
    )
    if image_field:
        with (task_dir / "task.toml").open("a", encoding="utf-8") as stream:
            stream.write(f'\n{image_field} = "{image}"\n')
    source, _ = get_dataset("swebench_verified")
    source = replace(source, default_image_registry="fallback.example")

    task = parse_harbor_task(task_dir, source=source, image_registry=image_registry)

    expected = (
        f"{image_registry}/owner/image@sha256:" + "a" * 64 if image_registry else image
    )
    assert task["docker_image"] == expected


@pytest.mark.parametrize("registry", ["registry.test", "localhost:5000/mirror"])
def test_apply_image_digest_lock_pins_task_images(tmp_path, registry) -> None:
    digest = "sha256:" + ("a" * 64)
    lock = tmp_path / "images.json"
    lock.write_text(
        json.dumps(
            {
                "registry": "capture.example",
                "images": [
                    {
                        "task_id": "owner__repo-1",
                        "digest": digest,
                        "image": "capture.example/example:latest",
                        "digest_image": f"capture.example/example@{digest}",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    [task] = apply_image_digest_lock(
        [
            {
                "instance_id": "owner__repo-1",
                "docker_image": f"{registry}/example:latest",
                "verifier_docker_image": f"{registry}/verifier:1",
            }
        ],
        lock,
    )

    assert task["docker_image"] == f"{registry}/example@{digest}"
    assert task["verifier_docker_image"] == f"{registry}/verifier@{digest}"


def test_packaged_digest_lock_is_registry_neutral() -> None:
    packaged = files("frognano.configs.eval.image_locks").joinpath(
        "sweb-v-20260904.json"
    )
    content = packaged.read_text(encoding="utf-8")
    payload = json.loads(content)

    assert set(payload) == {"images"}
    assert len(payload["images"]) == 500
    assert len({row["task_id"] for row in payload["images"]}) == 500
    for row in payload["images"]:
        assert set(row) == {"task_id", "digest"}
    digests = harbor._load_image_digest_lock(Path(str(packaged)))
    tasks = [
        {
            "instance_id": task_id,
            "docker_image": "mirror.example:5000/swebench/example:latest",
        }
        for task_id in digests
    ]

    pinned = apply_image_digest_lock(tasks, Path(str(packaged)))

    assert len(pinned) == 500
    assert all(
        task["docker_image"]
        == f"mirror.example:5000/swebench/example@{digests[task['instance_id']]}"
        for task in pinned
    )


def test_apply_image_digest_lock_rejects_missing_task(tmp_path) -> None:
    lock = tmp_path / "images.json"
    lock.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "task_id": "other",
                        "digest": "sha256:" + ("a" * 64),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no digest for task 'missing'"):
        apply_image_digest_lock(
            [{"instance_id": "missing", "docker_image": "example:latest"}],
            lock,
        )


def test_parse_dockerfile_rejects_multiple_stages(tmp_path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM one\nFROM two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="single-stage"):
        parse_dockerfile(dockerfile)


def test_parse_harbor_task_uses_final_image_runtime_metadata(tmp_path) -> None:
    task_dir = _task_dir(tmp_path)
    (task_dir / "task.toml").write_text(
        """
[agent]
timeout_sec = 300

[verifier]
timeout_sec = 600

[environment]
docker_image = "owner/final-task:1"
cpus = 2
memory_mb = 4096
storage_mb = 10240
allow_internet = false
""",
        encoding="utf-8",
    )
    (task_dir / "environment" / "Dockerfile").write_text(
        """
FROM base:1
COPY files /app
WORKDIR /workspace
ENV LEGACY VALUE
RUN setup-command
""",
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

    task = parse_harbor_task(task_dir, source=source, image_registry=None)

    assert task["docker_image"] == "owner/final-task:1"
    assert task["repo_path"] == "/workspace"
    assert task["setup_commands"] == []
    assert task["resources"] == {
        "cpu": "2",
        "memory": "4096Mi",
        "storage": "10240Mi",
    }
    assert task["agent_network_mode"] == "no-network"
    assert task["verifier_network_mode"] == "no-network"


@pytest.mark.parametrize("name", ["missing", "terminal_bench_2"])
def test_get_dataset_rejects_unknown_name(name) -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        get_dataset(name)


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
    assert "--filter=blob:none" in next(args for args in calls if args[0] == "fetch")


def test_materialize_root_dataset_skips_sparse_checkout(tmp_path, monkeypatch) -> None:
    source = DatasetSource(
        name="root",
        display_name="Root",
        source_url="https://example.test/root.git",
        revision="d" * 40,
        subpath=".",
        pod_prefix="root",
    )
    calls = []

    def fake_git(root, *args, env, capture=False):
        calls.append(args)
        if args[0] == "init":
            (root / ".git").mkdir()
        return source.revision if args[:2] == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(harbor, "_git", fake_git)

    selected = materialize_source(source, tmp_path)

    assert selected.is_dir()
    assert not any(call[0] == "sparse-checkout" for call in calls)


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


@pytest.mark.parametrize("image_registry", [None, "mirror.example"])
def test_terminal_verified_uses_harbor_catalog_defaults(
    tmp_path, monkeypatch, image_registry
) -> None:
    source, loader = get_dataset("terminal_bench_2_verified")
    assert loader is load_harbor_dataset
    assert source.source_url == (
        "https://huggingface.co/datasets/zai-org/terminal-bench-2-verified.git"
    )
    assert source.git_filter is None
    assert source.default_image_registry is None
    _task_dir(tmp_path)
    monkeypatch.setattr(harbor, "materialize_source", lambda *args: tmp_path)

    [task] = load_dataset(
        source.name,
        cache_dir=tmp_path,
        image_registry=image_registry,
        task_ids=("owner__repo-1",),
        limit=1,
    )

    prefix = f"{image_registry}/" if image_registry else ""
    assert task["docker_image"] == f"{prefix}swebench/example:latest"
    assert task["dataset"] == "terminal_bench_2_verified"
    assert task["repo_path"] == "/testbed"
    assert task["agent_network_mode"] == "public"
    assert task["verifier_network_mode"] == "public"
    assert task["source"]["revision"] == source.revision
    with pytest.raises(ValueError, match="unknown task IDs"):
        load_dataset(
            source.name,
            cache_dir=tmp_path,
            image_registry=None,
            task_ids=("missing",),
        )


def test_terminal_verified_materializes_without_partial_clone(
    tmp_path, monkeypatch
) -> None:
    source, _ = get_dataset("terminal_bench_2_verified")
    calls = []

    def fake_git(root, *args, env, capture=False):
        calls.append(args)
        if args[0] == "init":
            (root / ".git").mkdir()
        return source.revision if args[:2] == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(harbor, "_git", fake_git)

    assert materialize_source(source, tmp_path).is_dir()
    fetch = next(args for args in calls if args[0] == "fetch")
    assert fetch == ("fetch", "--quiet", "--depth=1", "origin", source.revision)
    assert not any(args[0] == "sparse-checkout" for args in calls)


@pytest.mark.parametrize(
    "text", ["\n\nFix it.\n", "# canary GUID example\n\nFix it.\n"]
)
def test_terminal_verified_normalizes_instruction_without_changing_other_datasets(
    tmp_path, text
) -> None:
    task_dir = _task_dir(tmp_path)
    (task_dir / "instruction.md").write_text(text, encoding="utf-8")
    source, _ = get_dataset("terminal_bench_2_verified")
    other_source, _ = get_dataset("swebench_verified")

    task = parse_harbor_task(task_dir, source=source, image_registry=None)
    other_task = parse_harbor_task(task_dir, source=other_source, image_registry=None)

    assert task["instruction"] == "Fix it.\n"
    assert other_task["instruction"] == text


@pytest.mark.parametrize("image_registry", [None, "mirror.example"])
def test_load_patch_eval_verified_builds_isolated_tasks(
    tmp_path, monkeypatch, image_registry
) -> None:
    root = tmp_path / "patcheval"
    root.mkdir()
    (root / "patcheval_verified.json").write_text(
        json.dumps(
            [
                {
                    "cve_id": "CVE-2021-23376",
                    "cve_description": "Command injection.",
                    "repo": "https://github.com/example/project",
                    "image_url": "ghcr.io/patcheval-cve/example:1",
                }
            ]
        ),
        encoding="utf-8",
    )
    source, _ = get_dataset("patch_eval_verified")
    monkeypatch.setattr(
        "frognano.datasets.patch_eval_verified.materialize_source",
        lambda *args: root,
    )

    tasks = load_patch_eval_verified(
        source,
        cache_dir=tmp_path,
        image_registry=image_registry,
        task_ids=("CVE-2021-23376",),
        limit=None,
    )

    assert len(tasks) == 1
    registry = image_registry or "ghcr.io"
    assert tasks[0]["docker_image"] == f"{registry}/patcheval-cve/example:1"
    assert tasks[0]["repo_name"] == "project"
    assert tasks[0]["dataset"] == tasks[0]["dataset_type"] == "patch_eval_verified"
    assert tasks[0]["verifier_protocol"] == "patch_eval_verified"
    assert tasks[0]["pod_prefix"] == "patch-eval-verified"
    assert (
        tasks[0]["agent_network_mode"]
        == tasks[0]["verifier_network_mode"]
        == "no-network"
    )
    assert tasks[0]["detect_repo_path"] is True
    assert tasks[0]["hide_workspace_payload"] is True
    assert tasks[0]["resources"] == {"cpu": "1", "memory": "4G", "storage": "20G"}
    assert "Command injection." in tasks[0]["instruction"]
