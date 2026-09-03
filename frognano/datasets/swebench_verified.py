from __future__ import annotations

from .source import DatasetSource

SOURCE = DatasetSource(
    name="swebench_verified",
    display_name="SWE-bench Verified",
    source_url="https://github.com/laude-institute/harbor-datasets.git",
    revision="86723674f04e4209ac479d0fb75d9d9f44b4377e",
    subpath="datasets/swebench-verified",
    pod_prefix="sweb-v",
    agent_network_mode="public",
    verifier_network_mode="public",
    verifier_success_marker="SWEBench results ends here",
)
