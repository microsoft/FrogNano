"""Leaf task environment backed by a FrogNano runtime."""

from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path
from typing import Any

from frognano.runtimes.kubernetes import KubernetesTaskRuntime

_RUNNER_PATH = "/tmp/frognano_leaf_tool_runner.py"
_MUTATING_TOOLS = frozenset({"Write", "Edit", "Bash"})
_POD_ERROR_MARKERS = (
    "unreachable",
    "consecutive exec failures",
    "proxy returned 502",
    "proxy returned 503",
    "proxy returned 504",
    "urlopen error",
    "connection refused",
    "timed out",
    "eof",
    "pod command returned no exit marker",
)
_MAX_POD_RECOVERIES = 2


class LeafEnvironment:
    def __init__(self, runtime: KubernetesTaskRuntime) -> None:
        self.runtime = runtime
        runner = Path(__file__).with_name("tool_runner.py")
        self._runner = runner
        self._action_history: list[tuple[str, dict[str, Any]]] = []
        self._pod_recoveries = 0
        self._provision()

    def instruction(self) -> str:
        return self.runtime.get_task_instruction()

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            output = self._execute(name, arguments)
        except Exception as exc:
            if not self._is_pod_error(exc) or (
                self._pod_recoveries >= _MAX_POD_RECOVERIES
            ):
                raise
            self._pod_recoveries += 1
            try:
                self.runtime.recreate()
                self._provision()
            except Exception as recovery_exc:
                raise RuntimeError("pod recovery failed") from recovery_exc
            for previous_name, previous_arguments in self._action_history:
                try:
                    self._execute(previous_name, previous_arguments)
                except Exception:
                    self.runtime.logger.warning(
                        "Pod recovery: replay of %s failed (continuing)",
                        previous_name,
                        exc_info=True,
                    )
            output = self._execute(name, arguments)
        if name in _MUTATING_TOOLS:
            self._action_history.append((name, dict(arguments)))
        return output

    def _provision(self) -> None:
        self.runtime.copy_to_container(self._runner, _RUNNER_PATH)

    def _execute(self, name: str, arguments: dict[str, Any]) -> str:
        payload = base64.b64encode(
            json.dumps(
                {
                    "tool": name,
                    "args": arguments,
                    "workdir": self.runtime.task["repo_path"],
                }
            ).encode()
        ).decode()
        command = (
            "python_bin=$(test -x /usr/bin/python3 && echo /usr/bin/python3 "
            "|| command -v python3 || command -v python); "
            f"echo {shlex.quote(payload)} | base64 -d | "
            f'"$python_bin" {shlex.quote(_RUNNER_PATH)}'
        )
        timeout = _timeout(name, arguments)
        output, exit_code = self.runtime.run(
            command,
            timeout=timeout,
            workdir="/",
        )
        if exit_code != 0:
            raise RuntimeError(f"tool runner failed ({exit_code}): {output}")
        return output

    def patch(self) -> str:
        return self.runtime.get_patch()

    @staticmethod
    def _is_pod_error(error: Exception) -> bool:
        message = str(error).lower()
        return any(marker in message for marker in _POD_ERROR_MARKERS)


def _timeout(name: str, arguments: dict[str, Any]) -> int:
    if name != "Bash":
        return 120
    if arguments.get("timeout_ms") is not None:
        value = float(arguments["timeout_ms"]) / 1000.0
    else:
        raw = arguments.get("timeout")
        value = 120.0 if raw is None else float(raw)
        if value > 1000:
            value /= 1000.0
    return max(1, int(max(0.1, value)) + 60)
