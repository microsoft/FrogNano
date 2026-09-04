from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from frognano.datasets.harbor import materialize_source
from frognano.datasets.source import DatasetSource

SOURCE = DatasetSource(
    name="patch_eval",
    display_name="PatchEval Verified",
    source_url="https://github.com/bytedance/PatchEval.git",
    revision="b43285cdde80cc04608d5f1178a330b740c91c2d",
    subpath="patcheval/datasets",
    pod_prefix="patch-eval",
    agent_network_mode="no-network",
    verifier_network_mode="no-network",
)


def load_patch_eval(
    source: DatasetSource,
    *,
    cache_dir: Path,
    image_registry: str | None,
    task_ids: tuple[str, ...],
    limit: int | None,
) -> list[dict[str, Any]]:
    root = materialize_source(source, cache_dir)
    samples = json.loads((root / "patcheval_verified.json").read_text(encoding="utf-8"))
    selected = set(task_ids)
    tasks = []
    for sample in samples:
        cve = str(sample["cve_id"])
        if selected and cve not in selected:
            continue
        tasks.append(_parse_sample(sample, source, image_registry))
        if limit is not None and len(tasks) >= limit:
            break
    if selected:
        missing = selected - {task["instance_id"] for task in tasks}
        if missing:
            raise ValueError(f"unknown task IDs: {sorted(missing)}")
    return tasks


def _parse_sample(
    sample: dict[str, Any],
    source: DatasetSource,
    image_registry: str | None,
) -> dict[str, Any]:
    cve = str(sample["cve_id"])
    image = str(sample.get("image_url") or "").strip()
    if not image:
        raise ValueError(f"PatchEval sample {cve} has no image_url")
    if image_registry and _is_unqualified_image(image):
        image = f"{image_registry.rstrip('/')}/{image}"
    repo_name = Path(urlparse(str(sample.get("repo") or "")).path).stem
    return {
        "dataset_type": "patch_eval",
        "dataset": source.name,
        "instance_id": cve,
        "instruction": _instruction(sample),
        "docker_image": image,
        "repo_path": "/workspace",
        "repo_name": repo_name,
        "detect_repo_path": True,
        "hide_workspace_payload": True,
        "verifier_protocol": "patch_eval",
        "environment_env": {},
        "setup_commands": [],
        "tests_dir": None,
        "agent_timeout_sec": 3600,
        "verifier_timeout_sec": 600,
        "agent_network_mode": source.agent_network_mode,
        "verifier_network_mode": source.verifier_network_mode,
        "verifier_success_marker": None,
        "resources": {
            "cpu": "1",
            "memory": "4G",
            "storage": "20G",
        },
        "source": {
            "url": source.source_url,
            "revision": source.revision,
            "subpath": source.subpath,
        },
        "pod_prefix": source.pod_prefix,
    }


def _instruction(sample: dict[str, Any]) -> str:
    description = str(sample.get("cve_description") or "").strip()
    return (
        "Please fix the vulnerabilities in the code repository based on the "
        f"following information: {description}\n\n"
        "The target repository is under /workspace. Locate its Git root before "
        "editing, and keep all tool paths inside that repository.\n\n"
        "Do not search the web for this vulnerability, CVE, advisory, GHSA, "
        "release note, issue, pull request, or upstream patch. Do not run "
        "network commands to find the fix."
    )


def _is_unqualified_image(image: str) -> bool:
    first = image.split("/", 1)[0]
    return "." not in first and ":" not in first and first != "localhost"
