from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class DatasetSource:
    name: str
    display_name: str
    source_url: str
    revision: str
    subpath: str
    pod_prefix: str
    agent_network_mode: str = "no-network"
    verifier_network_mode: str = "no-network"
    default_image_registry: str | None = None
    verifier_success_marker: str | None = None


DatasetLoader = Callable[..., list[dict]]

_DATASETS: dict[str, tuple[DatasetSource, DatasetLoader]] = {}


def register_dataset(source: DatasetSource, loader: DatasetLoader) -> None:
    if source.name in _DATASETS:
        raise ValueError(f"dataset already registered: {source.name}")
    _DATASETS[source.name] = (source, loader)


def get_dataset(name: str) -> tuple[DatasetSource, DatasetLoader]:
    try:
        return _DATASETS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown dataset {name!r}; available: {sorted(_DATASETS)}"
        ) from exc


def load_dataset(
    name: str,
    *,
    cache_dir: Path,
    image_registry: str | None,
    task_ids: tuple[str, ...] = (),
    limit: int | None = None,
) -> list[dict]:
    source, loader = get_dataset(name)
    return loader(
        source,
        cache_dir=cache_dir,
        image_registry=image_registry,
        task_ids=task_ids,
        limit=limit,
    )


from . import swebench_verified as _swebench_verified  # noqa: E402,F401

__all__ = [
    "DatasetSource",
    "get_dataset",
    "load_dataset",
    "register_dataset",
]
