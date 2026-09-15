"""Leaf task environment backed by a FrogNano runtime."""

from __future__ import annotations

import json
import re
import shlex
import tempfile
import uuid
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

from frognano.runtimes.errors import PodExecutionError
from frognano.runtimes.kubernetes import KubernetesTaskRuntime
from frognano.runtimes.python import FIND_PYTHON

_RUNNER_PATH = "/tmp/frognano_leaf_tool_runner.py"
_REQUEST_PREFIX = "/tmp/frognano_leaf_request_"
_MUTATING_TOOLS = frozenset({"Write", "Edit", "Bash"})
_MAX_POD_RECOVERIES = 2
_Result = TypeVar("_Result")


class LeafEnvironment:
    def __init__(self, runtime: KubernetesTaskRuntime) -> None:
        self.runtime = runtime
        runner = Path(__file__).with_name("tool_runner.py")
        self._runner = runner
        self._action_history: list[tuple[str, dict[str, Any], tuple[int, str]]] = []
        self._pod_recoveries = 0
        self._provision()

    def instruction(self) -> str:
        return self.runtime.get_task_instruction()

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        output, exit_code = self._with_pod_recovery(
            lambda: self._execute(name, arguments)
        )
        if name in _MUTATING_TOOLS:
            self._action_history.append(
                (name, deepcopy(arguments), _replay_outcome(name, output, exit_code))
            )
        return output

    def _with_pod_recovery(self, operation: Callable[[], _Result]) -> _Result:
        while True:
            try:
                return operation()
            except PodExecutionError as exc:
                if self._pod_recoveries >= _MAX_POD_RECOVERIES:
                    raise
                self._pod_recoveries += 1
                self.runtime.logger.warning(
                    "Pod recovery %s/%s after execution failure: %s",
                    self._pod_recoveries,
                    _MAX_POD_RECOVERIES,
                    exc,
                )
                try:
                    self.runtime.recreate()
                    self._provision()
                except Exception as recovery_exc:
                    raise PodExecutionError(
                        f"pod recovery failed: {type(recovery_exc).__name__}: "
                        f"{recovery_exc}; original failure: {exc}"
                    ) from recovery_exc
                for previous_name, previous_arguments, expected in self._action_history:
                    try:
                        output, code = self._execute(previous_name, previous_arguments)
                        if _replay_outcome(previous_name, output, code) != expected:
                            raise RuntimeError(
                                "replayed tool returned a different outcome"
                            )
                    except Exception as replay_exc:
                        raise PodExecutionError(
                            f"pod recovery replay of {previous_name} failed: "
                            f"{type(replay_exc).__name__}: {replay_exc}"
                        ) from replay_exc

    def _provision(self) -> None:
        self.runtime.copy_to_container(self._runner, _RUNNER_PATH)

    def _execute(self, name: str, arguments: dict[str, Any]) -> tuple[str, int]:
        timeout = _timeout(name, arguments)
        request_path = f"{_REQUEST_PREFIX}{uuid.uuid4().hex}.json"
        with tempfile.TemporaryDirectory(prefix="frognano-leaf-") as directory:
            local_path = Path(directory) / "request.json"
            local_path.write_text(
                json.dumps(
                    {
                        "tool": name,
                        "args": arguments,
                        "workdir": self.runtime.task["repo_path"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.runtime.copy_to_container(local_path, request_path)
        cleanup = shlex.quote(f"rm -f -- {shlex.quote(request_path)}")
        command = (
            f"trap {cleanup} EXIT; {FIND_PYTHON}; "
            f'"$python_bin" {shlex.quote(_RUNNER_PATH)} {shlex.quote(request_path)}'
        )
        output, exit_code = self.runtime.run(
            command,
            timeout=timeout,
            workdir="/",
        )
        if exit_code != 0:
            self.runtime.logger.warning(
                "Leaf tool runner exited with code %s", exit_code
            )
            return f"Tool runner exited with code {exit_code}.\n{output}", exit_code
        return output, exit_code

    def patch(self) -> str:
        return self._with_pod_recovery(self.runtime.get_patch)

    def compute_reward(self) -> tuple[float, str]:
        return self._with_pod_recovery(self.runtime.compute_reward)


def _replay_outcome(name: str, output: str, exit_code: int) -> tuple[int, str]:
    if name == "Bash" and exit_code == 0:
        status = re.match(r"(Exit code: -?\d+|Timed out after [\d.]+s)\n", output)
        if status:
            return exit_code, status.group(1)
    return exit_code, output


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
