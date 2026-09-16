from __future__ import annotations

from dataclasses import dataclass


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
    git_filter: str | None = "blob:none"
    strip_instruction_canary: bool = False
    require_git_patch: bool = True
