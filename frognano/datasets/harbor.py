from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frognano.datasets.source import DatasetSource

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


@dataclass(frozen=True)
class DockerfileRuntime:
    image: str
    workdir: str
    environment: dict[str, str]
    setup_commands: tuple[str, ...]


def materialize_source(source: DatasetSource, cache_dir: Path) -> Path:
    revision = source.revision.lower()
    if not _REVISION_RE.fullmatch(revision):
        raise ValueError("dataset revision must be a full SHA-1 commit")
    digest = hashlib.sha256(source.source_url.encode()).hexdigest()
    root = cache_dir.expanduser() / digest / revision
    root_dataset = source.subpath in {"", "."}
    selected = root if root_dataset else root / source.subpath
    marker = root / ".frognano-source"
    if marker.is_file() and marker.read_text().strip() == revision:
        if selected.is_dir():
            return selected
    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = root.with_name(f".{root.name}.tmp-{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        _git(stage, "init", "--quiet", env=env)
        _git(stage, "remote", "add", "origin", source.source_url, env=env)
        _git(
            stage,
            "fetch",
            "--quiet",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            revision,
            env=env,
        )
        if not root_dataset:
            _git(stage, "sparse-checkout", "init", "--cone", env=env)
            _git(stage, "sparse-checkout", "set", "--cone", source.subpath, env=env)
        _git(stage, "checkout", "--quiet", "--detach", "FETCH_HEAD", env=env)
        resolved = _git(stage, "rev-parse", "HEAD", env=env, capture=True)
        if resolved.lower() != revision:
            raise ValueError(f"dataset resolved to {resolved}, expected {revision}")
        shutil.rmtree(stage / ".git")
        (stage / ".frognano-source").write_text(f"{revision}\n")
        stage.rename(root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    if not selected.is_dir():
        raise FileNotFoundError(f"dataset subpath does not exist: {selected}")
    return selected


def _git(
    root: Path,
    *args: str,
    env: dict[str, str],
    capture: bool = False,
) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=capture,
        text=True,
        env=env,
    )
    return result.stdout.strip() if capture else ""


def load_harbor_dataset(
    source: DatasetSource,
    *,
    cache_dir: Path,
    image_registry: str | None,
    task_ids: tuple[str, ...],
    limit: int | None,
) -> list[dict]:
    root = materialize_source(source, cache_dir)
    selected = set(task_ids)
    tasks: list[dict] = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir() or not (task_dir / "task.toml").is_file():
            continue
        if selected and task_dir.name not in selected:
            continue
        tasks.append(
            parse_harbor_task(
                task_dir,
                source=source,
                image_registry=image_registry,
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    if selected:
        missing = selected - {task["instance_id"] for task in tasks}
        if missing:
            raise ValueError(f"unknown task IDs: {sorted(missing)}")
    return tasks


def parse_harbor_task(
    task_dir: Path,
    *,
    source: DatasetSource,
    image_registry: str | None,
) -> dict[str, Any]:
    with (task_dir / "task.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    instruction_path = task_dir / "instruction.md"
    tests_dir = task_dir / "tests"
    if not instruction_path.is_file() or not (tests_dir / "test.sh").is_file():
        raise ValueError(
            f"Harbor task {task_dir.name} requires instruction.md and tests/test.sh"
        )
    environment = dict(manifest.get("environment") or {})
    agent = dict(manifest.get("agent") or {})
    verifier = dict(manifest.get("verifier") or {})
    configured_image = environment.get("docker_image") or environment.get("base_image")
    runtime = parse_dockerfile(
        task_dir / "environment" / "Dockerfile",
        final_image=bool(configured_image),
    )
    image = str(configured_image or runtime.image)
    effective_registry = image_registry or source.default_image_registry
    if effective_registry and _is_unqualified_image(image):
        image = f"{effective_registry}/{image}"
    workdir = str(environment.get("workdir") or runtime.workdir or "/app")
    agent_timeout = int(agent.get("timeout_sec") or 3600)
    verifier_timeout = int(verifier.get("timeout_sec") or 600)
    return {
        "dataset_type": "harbor",
        "dataset": source.name,
        "instance_id": task_dir.name,
        "instruction": instruction_path.read_text(encoding="utf-8"),
        "docker_image": image,
        "repo_path": workdir,
        "environment_env": {
            **runtime.environment,
            **{
                str(key): _expand_env(str(value))
                for key, value in dict(environment.get("env") or {}).items()
            },
        },
        "setup_commands": ([] if configured_image else list(runtime.setup_commands)),
        "tests_dir": str(tests_dir),
        "agent_timeout_sec": agent_timeout,
        "verifier_timeout_sec": verifier_timeout,
        "agent_network_mode": _network_mode(
            agent,
            environment,
            source.agent_network_mode,
        ),
        "verifier_network_mode": _network_mode(
            verifier,
            environment,
            source.verifier_network_mode,
        ),
        "verifier_success_marker": source.verifier_success_marker,
        "resources": {
            "cpu": str(environment.get("cpus") or "0.5"),
            "memory": _resource_quantity(environment, "memory", "memory_mb", "8G"),
            "storage": _resource_quantity(
                environment,
                "storage",
                "storage_mb",
                "20G",
            ),
        },
        "source": {
            "url": source.source_url,
            "revision": source.revision,
            "subpath": source.subpath,
        },
        "pod_prefix": source.pod_prefix,
    }


def parse_dockerfile(
    path: Path,
    *,
    final_image: bool = False,
) -> DockerfileRuntime:
    if not path.is_file():
        raise ValueError(f"Harbor Dockerfile does not exist: {path}")
    image = ""
    workdir = "/app"
    environment: dict[str, str] = {}
    setup_commands: list[str] = []
    for instruction, value in _dockerfile_instructions(path.read_text()):
        if final_image:
            if instruction == "FROM" and not image:
                image = value.split()[0]
            elif instruction == "WORKDIR":
                workdir = value.strip()
            continue
        if instruction == "FROM":
            if image:
                raise ValueError(f"Harbor Dockerfile must be single-stage: {path}")
            image = value.split()[0]
        elif instruction == "WORKDIR":
            workdir = value.strip()
        elif instruction == "ENV":
            key, separator, item = value.partition("=")
            if not separator:
                raise ValueError(f"Dockerfile ENV must use KEY=VALUE: {path}")
            environment[key.strip()] = item.strip()
        elif instruction == "RUN":
            if not final_image:
                setup_commands.append(value)
        elif instruction in {
            "ARG",
            "CMD",
            "ENTRYPOINT",
            "LABEL",
            "SHELL",
            "USER",
        }:
            continue
        else:
            raise ValueError(
                f"unsupported Harbor Dockerfile instruction {instruction}: {path}"
            )
    if not image:
        raise ValueError(f"Harbor Dockerfile has no FROM image: {path}")
    return DockerfileRuntime(
        image=image,
        workdir=workdir,
        environment=environment,
        setup_commands=tuple(setup_commands),
    )


def _dockerfile_instructions(text: str) -> list[tuple[str, str]]:
    logical: list[str] = []
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        current = f"{current} {line}".strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        logical.append(current)
        current = ""
    if current:
        raise ValueError("Dockerfile ends with a continuation")
    return [
        (line.split(None, 1)[0].upper(), line.split(None, 1)[1])
        for line in logical
        if len(line.split(None, 1)) == 2
    ]


def _is_unqualified_image(image: str) -> bool:
    first = image.split("/", 1)[0]
    return "." not in first and ":" not in first and first != "localhost"


def _expand_env(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.groups()
        return os.environ.get(name, default or "")

    return _ENV_RE.sub(replace, value)


def _network_mode(
    role: dict[str, Any],
    environment: dict[str, Any],
    default: str,
) -> str:
    if role.get("network_mode"):
        return str(role["network_mode"])
    if environment.get("network_mode"):
        return str(environment["network_mode"])
    if environment.get("allow_internet") is not None:
        return "public" if bool(environment["allow_internet"]) else "no-network"
    return default


def _resource_quantity(
    environment: dict[str, Any],
    value_key: str,
    megabytes_key: str,
    default: str,
) -> str:
    if environment.get(value_key) is not None:
        return str(environment[value_key])
    if environment.get(megabytes_key) is not None:
        return f"{environment[megabytes_key]}Mi"
    return default
