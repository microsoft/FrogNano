from __future__ import annotations

from frognano.datasets.source import DatasetSource

SOURCE = DatasetSource(
    name="terminal_bench_2_verified",
    display_name="Terminal-Bench 2.0 Verified",
    source_url="https://huggingface.co/datasets/zai-org/terminal-bench-2-verified.git",
    revision="7bbd2fef45db2f9ee9d9e3aae8f7253f56bf4e2b",
    subpath=".",
    pod_prefix="t-bench-2",
    agent_network_mode="public",
    verifier_network_mode="public",
    # Hugging Face's Git endpoint cannot serve the deferred blob fetches.
    git_filter=None,
    strip_instruction_canary=True,
    require_git_patch=False,
)
