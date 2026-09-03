from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        expanded = _ENV_RE.sub(
            lambda match: os.environ.get(match.group(1), match.group(2) or ""),
            value,
        )
        return os.path.expanduser(os.path.expandvars(expanded))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.6
    max_tokens_per_turn: int | None = None
    timeout_sec: int = 7200
    max_retries: int = 5
    parallel_tool_calls: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        name = str(value.get("name") or "").strip()
        base_url = str(value.get("base_url") or "").strip()
        if not name or not base_url:
            raise ValueError("model.name and model.base_url are required")
        max_tokens = value.get("max_tokens_per_turn")
        return cls(
            name=name,
            base_url=base_url.rstrip("/"),
            api_key_env=str(value.get("api_key_env") or "OPENAI_API_KEY"),
            temperature=float(value.get("temperature", 0.6)),
            max_tokens_per_turn=(
                None if max_tokens in (None, 0, "0") else int(max_tokens)
            ),
            timeout_sec=int(value.get("timeout_sec", 7200)),
            max_retries=int(value.get("max_retries", 5)),
            parallel_tool_calls=bool(value.get("parallel_tool_calls", True)),
            extra_body=dict(value.get("extra_body") or {}),
        )


@dataclass(frozen=True)
class KubernetesConfig:
    namespace: str = "default"
    context: str | None = None
    kubeconfig: str | None = None
    image_registry: str | None = None
    service_account: str | None = None
    pull_secret: str | None = None
    pod_start_timeout_sec: int = 900
    keep_pods: bool = False
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KubernetesConfig":
        return cls(
            namespace=str(value.get("namespace") or "default"),
            context=value.get("context"),
            kubeconfig=value.get("kubeconfig"),
            image_registry=(
                str(value["image_registry"]).rstrip("/")
                if value.get("image_registry")
                else None
            ),
            service_account=value.get("service_account"),
            pull_secret=value.get("pull_secret"),
            pod_start_timeout_sec=int(value.get("pod_start_timeout_sec", 900)),
            keep_pods=bool(value.get("keep_pods", False)),
            labels={
                str(key): str(item)
                for key, item in dict(value.get("labels") or {}).items()
            },
        )


@dataclass(frozen=True)
class EvalConfig:
    dataset: str
    output_dir: Path
    model: ModelConfig
    kubernetes: KubernetesConfig
    task_ids: tuple[str, ...] = ()
    num_tasks: int | None = None
    seed: int = 42
    seeds_per_task: int = 1
    max_workers: int = 1
    max_attempts: int = 2
    max_steps: int = 100
    max_context_tokens: int = 65536
    max_total_time_sec: int | None = None
    resume: bool = True
    cache_dir: Path = Path("~/.cache/frognano/harbor").expanduser()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvalConfig":
        dataset = str(value.get("dataset") or "").strip()
        if not dataset:
            raise ValueError("dataset is required")
        num_tasks = value.get("num_tasks")
        max_total_time = value.get("max_total_time_sec")
        config = cls(
            dataset=dataset,
            output_dir=Path(str(value.get("output_dir") or "eval-results")),
            model=ModelConfig.from_dict(dict(value.get("model") or {})),
            kubernetes=KubernetesConfig.from_dict(dict(value.get("kubernetes") or {})),
            task_ids=tuple(str(item) for item in value.get("task_ids") or ()),
            num_tasks=None if num_tasks is None else int(num_tasks),
            seed=int(value.get("seed", 42)),
            seeds_per_task=int(value.get("seeds_per_task", 1)),
            max_workers=int(value.get("max_workers", 1)),
            max_attempts=int(value.get("max_attempts", 2)),
            max_steps=int(value.get("max_steps", 100)),
            max_context_tokens=int(value.get("max_context_tokens", 65536)),
            max_total_time_sec=(
                None if max_total_time is None else int(max_total_time)
            ),
            resume=bool(value.get("resume", True)),
            cache_dir=Path(str(value.get("cache_dir") or "~/.cache/frognano/harbor")),
        )
        for name in (
            "seeds_per_task",
            "max_workers",
            "max_attempts",
            "max_steps",
        ):
            if getattr(config, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if config.num_tasks is not None and config.num_tasks < 0:
            raise ValueError("num_tasks cannot be negative")
        if (
            config.max_total_time_sec is not None
            and not 60 <= config.max_total_time_sec <= 7200
        ):
            raise ValueError("max_total_time_sec must be between 60 and 7200")
        return config


def load_config(path: str | Path) -> EvalConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install FrogNano") from exc
    raw = yaml.safe_load(_read_config(path))
    if not isinstance(raw, dict):
        raise ValueError("evaluation config must be a YAML object")
    return EvalConfig.from_dict(_expand(raw))


def _read_config(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    name = str(path)
    if candidate.name == name:
        filename = name if name.endswith(".yaml") else f"{name}.yaml"
        packaged = files("frognano.configs.eval").joinpath(filename)
        if packaged.is_file():
            return packaged.read_text(encoding="utf-8")
    raise FileNotFoundError(f"evaluation config does not exist: {path}")
