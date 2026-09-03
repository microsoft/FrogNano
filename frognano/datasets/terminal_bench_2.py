from __future__ import annotations

from frognano.datasets.source import DatasetSource

SOURCE = DatasetSource(
    name="terminal_bench_2",
    display_name="Terminal-Bench 2.0",
    source_url="https://github.com/laude-institute/terminal-bench-2.git",
    revision="69671fbaac6d67a7ef0dfec016cc38a64ef7a77c",
    subpath=".",
    pod_prefix="tb2",
    agent_network_mode="public",
    verifier_network_mode="public",
)
