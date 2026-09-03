from __future__ import annotations

from frognano.datasets.source import DatasetSource

SOURCE = DatasetSource(
    name="swebench_pro",
    display_name="SWE-bench Pro",
    source_url="https://github.com/laude-institute/harbor-datasets.git",
    revision="c8e8f3fac7097accaacf261d74c3d6f441de45b1",
    subpath="datasets/swebenchpro",
    pod_prefix="sweb-p",
    agent_network_mode="public",
    verifier_network_mode="public",
)
