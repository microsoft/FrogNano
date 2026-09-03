from __future__ import annotations

from pathlib import Path
from typing import Callable

from frognano.datasets.harbor import load_harbor_dataset
from frognano.datasets.patch_eval import SOURCE as PATCH_EVAL
from frognano.datasets.patch_eval import load_patch_eval
from frognano.datasets.source import DatasetSource
from frognano.datasets.swebench_pro import SOURCE as SWEBENCH_PRO
from frognano.datasets.swebench_verified import SOURCE as SWEBENCH_VERIFIED
from frognano.datasets.terminal_bench_2 import SOURCE as TERMINAL_BENCH_2

DatasetLoader = Callable[..., list[dict]]

_DATASETS: dict[str, tuple[DatasetSource, DatasetLoader]] = {
    PATCH_EVAL.name: (PATCH_EVAL, load_patch_eval),
    SWEBENCH_PRO.name: (SWEBENCH_PRO, load_harbor_dataset),
    SWEBENCH_VERIFIED.name: (SWEBENCH_VERIFIED, load_harbor_dataset),
    TERMINAL_BENCH_2.name: (TERMINAL_BENCH_2, load_harbor_dataset),
}


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


__all__ = [
    "DatasetSource",
    "get_dataset",
    "load_dataset",
]
