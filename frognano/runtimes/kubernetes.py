"""Kubernetes task execution runtime."""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
import shlex
import tarfile
import tempfile
import time
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any

from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import HTTPError
from websocket import WebSocketException
from yaml import YAMLError

from frognano.config import KubernetesConfig
from frognano.runtimes.errors import CommandTimeoutError, PodExecutionError
from frognano.runtimes.python import CHECK_PYTHON, INSTALL_PYTHON

_SAFE_NAME_RE = re.compile(r"[^a-z0-9-]+")
_TRANSPORT_ERRORS = (ApiException, HTTPError, OSError, WebSocketException)
_COMMAND_CONTROL_TIMEOUT = 30
_COMMAND_CONTROL_ATTEMPTS = 3
_COMMAND_START_GRACE = 60
_OUTPUT_CHUNK_BYTES = 49152

_CHECK_COREUTILS = r"""
set -eo pipefail
for tool in timeout cat dd head base64 sha256sum wc mkdir mv rm; do
    if ! command -v "$tool" >/dev/null; then
        printf 'Missing required GNU coreutils executable: %s\n' "$tool" >&2
        exit 1
    fi
    if ! version=$("$tool" --version 2>&1); then
        printf 'Cannot verify GNU coreutils executable %s: %s\n' "$tool" "$version" >&2
        exit 1
    fi
    case "$version" in
        *"(GNU coreutils)"*|*"(coreutils)"*) ;;
        *) printf 'Required executable %s is not GNU coreutils: %s\n' "$tool" "$version" >&2; exit 1 ;;
    esac
done
if ! timeout --kill-after=1s 1s /bin/bash -c 'exit 0'; then
    printf 'GNU timeout kill-after probe failed\n' >&2
    exit 1
fi
if [ "$(printf abc | dd iflag=skip_bytes,count_bytes skip=1 count=1 status=none | base64)" != Yg== ]; then
    printf 'GNU dd/base64 byte-range read probe failed\n' >&2
    exit 1
fi
printf 'GNU coreutils verified\n'
"""

_INSTALL_COREUTILS = (
    "if command -v apt-get >/dev/null 2>&1; then "
    "apt-get update -qq && "
    "DEBIAN_FRONTEND=noninteractive apt-get install "
    "-y -qq --no-install-recommends coreutils; "
    "elif command -v apk >/dev/null 2>&1; then "
    "apk add --no-cache coreutils; "
    "else echo 'No supported GNU coreutils package manager (apt-get or apk)' "
    ">&2; exit 1; fi"
)

_COMMAND_EXEC_SCRIPT = r"""
set -eu
umask 077
directory=$1
duration=$2
shift 2
mkdir "$directory/claimed"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "shell wrapper exited with code %s\n" "$rc" > "$directory/error"; fi' EXIT
set +e
timeout --kill-after=1s "$duration" /bin/bash -c '
    directory=$1
    shift
    "$@" 2>&1 | cat > "$directory/output"
    codes=("${PIPESTATUS[@]}")
    if [ "${codes[1]}" -ne 0 ]; then exit "${codes[1]}"; fi
    printf "%s\n" "${codes[0]}" > "$directory/exit-code.tmp" &&
        mv "$directory/exit-code.tmp" "$directory/exit-code"
' frognano-capture "$directory" "$@"
status=$?
set -e
timed_out=0
if [ "$status" -eq 124 ]; then
    timed_out=1
    code=124
elif [ "$status" -eq 0 ] && [ -f "$directory/exit-code" ]; then
    code=$(cat "$directory/exit-code")
else
    if [ "$status" -eq 0 ]; then exit 125; fi
    exit "$status"
fi
size=$(wc -c < "$directory/output")
digest=$(sha256sum "$directory/output")
printf 'completed %s %s %s %s\n' "$code" "$timed_out" "$size" "${digest%% *}" > "$directory/result.tmp"
mv "$directory/result.tmp" "$directory/result"
"""

_COMMAND_STATUS_SCRIPT = r"""
if [ -f "$1/result" ]; then
    head -c 1024 "$1/result"
elif [ -f "$1/error" ]; then
    printf 'error '
    head -c 2000 "$1/error"
elif [ -d "$1/claimed" ]; then
    printf 'running\n'
else
    printf 'missing\n'
fi
"""

_COMMAND_READ_SCRIPT = r"""
set -eo pipefail
dd if="$1/output" iflag=skip_bytes,count_bytes skip="$2" count="$3" status=none | base64
"""


class KubernetesTaskRuntime:
    def __init__(
        self,
        task: dict[str, Any],
        config: KubernetesConfig,
        *,
        run_id: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.task = task
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.namespace = config.namespace
        self._pod_prefix = str(task.get("pod_prefix") or "frognano")
        self._run_id = run_id
        self._recovery_count = 0
        self._verifier_mode = False
        self.pod_name = _pod_name(
            self._pod_prefix,
            str(task["instance_id"]),
            run_id,
        )
        self._network_policy_name = f"{self.pod_name}-deny-egress"
        self._load_clients()
        try:
            self._create_pod()
        except BaseException:
            self.close()
            raise

    def _load_clients(self) -> None:
        try:
            from kubernetes import client, config
            from kubernetes.stream import stream
        except ImportError as exc:
            raise RuntimeError("kubernetes is required; install FrogNano") from exc
        if self.config.kubeconfig:
            config.load_kube_config(
                config_file=self.config.kubeconfig,
                context=self.config.context,
            )
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config(context=self.config.context)
        self._client_module = client
        self._core = client.CoreV1Api()
        self._networking = client.NetworkingV1Api()
        self._stream = stream

    def _create_pod(self) -> None:
        from kubernetes.utils.quantity import parse_quantity

        client = self._client_module
        resources = self.task.get("resources") or {}
        quantities = {
            "cpu": str(resources.get("cpu") or "0.5"),
            "memory": _quantity(str(resources.get("memory") or "8G")),
            "ephemeral-storage": _quantity(str(resources.get("storage") or "20G")),
        }
        limits = dict(quantities)
        if "cpu_limit" in resources:
            if resources["cpu_limit"] is None:
                limits.pop("cpu")
            else:
                limits["cpu"] = str(resources["cpu_limit"])
        memory_limit = self.config.memory_limit or resources.get("memory_limit")
        if memory_limit is not None:
            limits["memory"] = _quantity(str(memory_limit))
            if parse_quantity(limits["memory"]) <= 0:
                raise ValueError("memory_limit must be positive")
            if parse_quantity(quantities["memory"]) > parse_quantity(limits["memory"]):
                quantities["memory"] = limits["memory"]
        labels = {
            "app.kubernetes.io/name": "frognano-eval",
            "app.kubernetes.io/component": "leaf",
            "frognano-run": _run_label(self.pod_name),
            **self.config.labels,
        }
        container = client.V1Container(
            name="task",
            image=str(self.task["docker_image"]),
            command=["/bin/sh", "-c"],
            args=["trap : TERM INT; while :; do sleep 300; done"],
            working_dir=str(self.task["repo_path"]),
            env=[
                client.V1EnvVar(name=str(key), value=str(value))
                for key, value in dict(self.task.get("environment_env") or {}).items()
            ],
            resources=client.V1ResourceRequirements(
                requests=quantities,
                limits=limits,
            ),
        )
        spec_kwargs: dict[str, Any] = {
            "containers": [container],
            "restart_policy": "Never",
            "active_deadline_seconds": self.config.pod_lifetime_sec
            or (
                int(self.task["agent_timeout_sec"])
                + int(self.task["verifier_timeout_sec"])
                + 600
            ),
        }
        if self.config.service_account:
            spec_kwargs["service_account_name"] = self.config.service_account
        if self.config.pull_secret:
            spec_kwargs["image_pull_secrets"] = [
                client.V1LocalObjectReference(name=self.config.pull_secret)
            ]
        pod = client.V1Pod(
            metadata=client.V1ObjectMeta(name=self.pod_name, labels=labels),
            spec=client.V1PodSpec(**spec_kwargs),
        )
        self.logger.info("Creating pod %s", self.pod_name)
        self._core.create_namespaced_pod(self.namespace, pod)
        deadline = time.monotonic() + self.config.pod_start_timeout_sec
        while time.monotonic() < deadline:
            current = self._core.read_namespaced_pod(self.pod_name, self.namespace)
            phase = str(current.status.phase or "")
            if phase == "Running":
                self._bootstrap_executor()
                self._initialize()
                if self.task.get("agent_network_mode") == "no-network":
                    self._enable_network_isolation()
                return
            if phase in {"Failed", "Succeeded"}:
                raise RuntimeError(
                    f"pod {self.pod_name} entered terminal phase {phase}: "
                    f"{self._pod_status_detail()}"
                )
            time.sleep(2)
        raise TimeoutError(
            f"pod {self.pod_name} did not become ready: {self._pod_status_detail()}"
        )

    def _bootstrap_executor(self) -> None:
        output, exit_code = self._exec(
            ["/bin/bash", "-c", _CHECK_COREUTILS],
            timeout=30,
            operation="shell executor bootstrap",
        )
        if exit_code != 0 or "GNU coreutils" not in output:
            if self.task.get("agent_network_mode") != "public":
                raise RuntimeError(
                    "task image requires Bash and GNU coreutils for file-backed "
                    "execution; cannot install GNU coreutils without public "
                    f"agent network access: {output[-2000:]}"
                )
            self.logger.info("Installing GNU coreutils for file-backed execution")
            output, exit_code = self._exec(
                ["/bin/sh", "-c", _INSTALL_COREUTILS],
                timeout=300,
                operation="GNU coreutils installation",
            )
            if exit_code != 0:
                raise RuntimeError(
                    "could not install GNU coreutils for file-backed execution: "
                    f"{output[-2000:]}"
                )
            output, exit_code = self._exec(
                ["/bin/bash", "-c", _CHECK_COREUTILS],
                timeout=30,
                operation="shell executor bootstrap",
            )
            if exit_code != 0 or "GNU coreutils" not in output:
                raise RuntimeError(
                    "GNU coreutils unavailable after installation: " f"{output[-2000:]}"
                )
        probe = f'{CHECK_PYTHON} && printf "%s\\n" "$python_bin"'
        output, exit_code = self._exec(
            ["/bin/sh", "-c", probe], timeout=30, operation="Python bootstrap"
        )
        if exit_code != 0:
            if self.task.get("agent_network_mode") != "public":
                raise RuntimeError(
                    "task image requires Python 3.6 or newer for Leaf tools; "
                    "cannot install it without public agent network access"
                )
            self.logger.info("Installing Python 3 for Leaf tools")
            output, exit_code = self._exec(
                ["/bin/sh", "-c", INSTALL_PYTHON],
                timeout=300,
                operation="Python installation",
            )
            if exit_code != 0:
                raise RuntimeError(
                    f"could not install Python 3 for Leaf tools: {output[-2000:]}"
                )
            output, exit_code = self._exec(
                ["/bin/sh", "-c", probe], timeout=30, operation="Python bootstrap"
            )
            if exit_code != 0:
                raise RuntimeError(
                    "Python 3.6 or newer is unavailable after installation: "
                    f"{output[-2000:]}"
                )
        python_path = output.strip()
        if not python_path.startswith("/") or "\n" in python_path:
            raise RuntimeError(f"invalid task Python executable: {python_path!r}")
        self._command_output_root = f"/tmp/frognano-commands-{uuid.uuid4().hex}"

    def _initialize(self) -> None:
        output, exit_code = self.run(
            "mkdir -p /logs/verifier /logs/agent /tests /solution", workdir="/"
        )
        if exit_code != 0:
            raise RuntimeError(f"could not initialize task directories: {output}")
        if self.task.get("detect_repo_path") and not self._verifier_mode:
            self._detect_repo_path()
        if self.task.get("hide_workspace_payload") and not self._verifier_mode:
            self._hide_workspace_payload()
        for command in self.task.get("setup_commands") or ():
            output, exit_code = self.run(
                str(command),
                timeout=int(self.task["agent_timeout_sec"]),
                workdir=str(self.task["repo_path"]),
            )
            if exit_code != 0:
                raise RuntimeError(
                    f"task image setup failed ({exit_code}): {output[-2000:]}"
                )

    def run(
        self,
        command: str,
        *,
        timeout: float = 120,
        workdir: str | None = None,
    ) -> tuple[str, int]:
        if not isinstance(command, str) or "\0" in command:
            raise ValueError("command must be a string without NUL bytes")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("command timeout must be positive and finite")
        directory = workdir or str(self.task["repo_path"])
        if not isinstance(directory, str) or "\0" in directory:
            raise ValueError("workdir must be a string without NUL bytes")
        interpreter = self.task.get("command_interpreter")
        if interpreter is None:
            interpreter = ["/bin/bash", "-c"]
        if (
            not isinstance(interpreter, (list, tuple))
            or not interpreter
            or not all(
                isinstance(argument, str) and argument and "\0" not in argument
                for argument in interpreter
            )
        ):
            raise ValueError("command_interpreter must be a non-empty argument list")
        command_id = uuid.uuid4().hex
        command_directory = f"{self._command_output_root}/{command_id}"
        command_path = f"{command_directory}/command.sh"
        with tempfile.TemporaryDirectory(prefix="frognano-exec-") as directory_path:
            local_path = Path(directory_path) / "command.sh"
            local_path.write_text(command, encoding="utf-8")
            self.copy_to_container(local_path, command_path)
        deadline = time.monotonic() + timeout + _COMMAND_START_GRACE
        try:
            output, exit_code = self._exec(
                [
                    "/bin/bash",
                    "-c",
                    _COMMAND_EXEC_SCRIPT,
                    "frognano-execute",
                    command_directory,
                    str(timeout),
                    *interpreter,
                    f"cd -- {shlex.quote(directory)} && "
                    f". {shlex.quote(command_path)}",
                ],
                timeout=timeout + _COMMAND_START_GRACE,
                operation=f"command {command_id} execute",
            )
            if exit_code != 0:
                raise PodExecutionError(
                    f"command {command_id} wrapper exited with code {exit_code}: "
                    f"{output[-2000:]}"
                )
        except (PodExecutionError, CommandTimeoutError) as exc:
            self.logger.warning(
                "Command %s execution response was interrupted; checking its "
                "saved state without re-executing: %s",
                command_id,
                exc,
            )
        result = self._wait_for_command(command_id, deadline)
        output = self._read_command_output(command_id, result)
        try:
            self._command_request("cleanup", command_id)
        except PodExecutionError as exc:
            self.logger.warning(
                "Could not remove verified command output %s: %s", command_id, exc
            )
        if result["timed_out"]:
            raise CommandTimeoutError(
                f"command {command_id} in pod {self.namespace}/{self.pod_name} "
                f"timed out after {timeout:g}s"
                f"\nOutput:\n{output[-2000:]}",
                output=output,
            )
        return output, result["returncode"]

    def _wait_for_command(self, command_id: str, deadline: float) -> dict[str, Any]:
        delay = 0.1
        while True:
            result = self._command_request("status", command_id)
            if result["state"] == "completed":
                return result
            if result["state"] != "running":
                raise PodExecutionError(
                    f"command {command_id} has no completion record "
                    f"(state={result['state']}, detail={result.get('error', '')}); "
                    f"output directory {self._command_output_root}/{command_id}; "
                    f"{self._pod_status_detail()}"
                )
            if time.monotonic() >= deadline:
                raise PodExecutionError(
                    f"command {command_id} has no completion record after its "
                    "deadline; the outcome is unknown and the pod must be "
                    f"recreated before replay; {self._pod_status_detail()}"
                )
            time.sleep(delay)
            delay = min(delay * 2, 2.0)

    def _command_request(
        self,
        operation: str,
        command_id: str,
        *arguments: str,
    ) -> dict[str, Any]:
        if re.fullmatch(r"[0-9a-f]{32}", command_id) is None:
            raise ValueError("invalid command ID")
        directory = f"{self._command_output_root}/{command_id}"
        if operation == "status":
            script = _COMMAND_STATUS_SCRIPT
        elif operation == "read":
            script = _COMMAND_READ_SCRIPT
        elif operation == "cleanup":
            script = 'rm -rf -- "$1"'
        else:
            raise ValueError(f"unsupported command operation: {operation}")
        command = [
            "/bin/bash",
            "-c",
            script,
            f"frognano-{operation}",
            directory,
            *arguments,
        ]
        for attempt in range(_COMMAND_CONTROL_ATTEMPTS):
            try:
                output, exit_code = self._exec(
                    command,
                    timeout=_COMMAND_CONTROL_TIMEOUT,
                    operation=f"command {command_id} {operation}",
                )
                if exit_code != 0:
                    raise PodExecutionError(
                        f"command {command_id} {operation} exited with code "
                        f"{exit_code}: {output[-2000:]}"
                    )
                if operation == "status":
                    return _parse_command_state(output)
                if operation == "read":
                    result = {
                        "offset": int(arguments[0]),
                        "length": int(arguments[1]),
                        "data": "".join(output.splitlines()),
                    }
                    _decode_output_chunk(result, int(arguments[0]), int(arguments[1]))
                    return result
                return {"removed": True}
            except (PodExecutionError, CommandTimeoutError, ValueError) as exc:
                if attempt + 1 == _COMMAND_CONTROL_ATTEMPTS:
                    raise PodExecutionError(
                        f"command {command_id} {operation} failed after "
                        f"{_COMMAND_CONTROL_ATTEMPTS} control requests: {exc}"
                    ) from exc
                self.logger.warning(
                    "Retrying command %s %s control request: %s",
                    command_id,
                    operation,
                    exc,
                )
                time.sleep(attempt + 1)
        raise AssertionError("command control attempts exhausted")

    def _read_command_output(self, command_id: str, result: dict[str, Any]) -> str:
        for attempt in range(_COMMAND_CONTROL_ATTEMPTS):
            output = bytearray()
            size = result["output_size"]
            while len(output) < size:
                offset = len(output)
                length = min(_OUTPUT_CHUNK_BYTES, size - offset)
                chunk = self._command_request(
                    "read", command_id, str(offset), str(length)
                )
                output.extend(_decode_output_chunk(chunk, offset, length))
            if sha256(output).hexdigest() == result["sha256"]:
                return output.decode("utf-8", errors="replace")
            self.logger.warning(
                "Command %s output checksum mismatch on read %s/%s",
                command_id,
                attempt + 1,
                _COMMAND_CONTROL_ATTEMPTS,
            )
        raise PodExecutionError(
            f"command {command_id} output failed SHA-256 verification; "
            "the command will not be re-executed in this pod"
        )

    def _exec(
        self,
        command: list[str],
        *,
        timeout: float,
        operation: str,
        input_text: str | None = None,
    ) -> tuple[str, int]:
        if timeout <= 0:
            raise ValueError("command timeout must be positive")
        response = None
        output = ""
        context = f"{operation} in pod {self.namespace}/{self.pod_name}"
        try:
            response = self._stream(
                self._core.connect_get_namespaced_pod_exec,
                self.pod_name,
                self.namespace,
                container="task",
                command=command,
                stderr=True,
                stdin=input_text is not None,
                stdout=True,
                tty=False,
                _preload_content=False,
                _request_timeout=timeout,
            )
            if input_text is not None:
                for index in range(0, len(input_text), 65536):
                    response.write_stdin(input_text[index : index + 65536])
            response.run_forever(timeout=timeout)
            if response.is_open():
                output = response.read_all()
                raise CommandTimeoutError(
                    f"{context} timed out after {timeout}s; "
                    f"{self._pod_status_detail()}\nOutput:\n{output[-2000:]}",
                    output=output,
                )
            try:
                # read_all() clears every channel, including the SDK exit status.
                exit_code = response.returncode
            except (IndexError, KeyError, TypeError, ValueError, YAMLError) as exc:
                output = response.read_all()
                raise PodExecutionError(
                    f"{context} returned an invalid Kubernetes exit status: "
                    f"{type(exc).__name__}: {exc}; {self._pod_status_detail()}"
                    f"\nOutput:\n{output[-2000:]}"
                ) from exc
            output = response.read_all()
            if not isinstance(exit_code, int) or isinstance(exit_code, bool):
                raise PodExecutionError(
                    f"{context} closed without a Kubernetes exit status; "
                    f"{self._pod_status_detail()}\nOutput:\n{output[-2000:]}"
                )
            return output, exit_code
        except CommandTimeoutError:
            raise
        except (
            ApiException,
            HTTPError,
            OSError,
            WebSocketException,
            AttributeError,
        ) as exc:
            causes = _exception_chain(exc)
            if not any(isinstance(cause, _TRANSPORT_ERRORS) for cause in causes):
                raise
            if response is not None:
                output += response.read_all()
            detail = "; caused by ".join(
                f"{type(cause).__name__}: {cause}" for cause in causes
            )
            raise PodExecutionError(
                f"{context} transport failed: {detail}; "
                f"{self._pod_status_detail()}\nOutput:\n{output[-2000:]}"
            ) from exc
        finally:
            if response is not None:
                response.close()

    def _pod_status_detail(self) -> str:
        try:
            pod = self._core.read_namespaced_pod(
                self.pod_name, self.namespace, _request_timeout=10
            )
        except ApiException as exc:
            if exc.status == 404:
                return "pod not found (404)"
            self.logger.warning("Could not inspect pod %s: %s", self.pod_name, exc)
            return f"pod status unavailable: {type(exc).__name__}: {exc}"
        except (HTTPError, OSError) as exc:
            self.logger.warning("Could not inspect pod %s: %s", self.pod_name, exc)
            return f"pod status unavailable: {type(exc).__name__}: {exc}"
        status = pod.status
        details = [f"phase={status.phase}"]
        for field in ("reason", "message"):
            value = getattr(status, field, None)
            if value:
                details.append(f"{field}={value}")
        for container in (getattr(status, "init_container_statuses", None) or []) + (
            getattr(status, "container_statuses", None) or []
        ):
            for state_name in ("state", "last_state"):
                state = getattr(container, state_name, None)
                for kind in ("waiting", "terminated"):
                    value = getattr(state, kind, None)
                    if value is not None:
                        fields = [
                            f"{field}={getattr(value, field)}"
                            for field in ("reason", "exit_code", "message")
                            if getattr(value, field, None) is not None
                        ]
                        details.append(
                            f"{container.name} {state_name}.{kind}: {', '.join(fields)}"
                        )
        for condition in getattr(status, "conditions", None) or []:
            if condition.status == "False" and condition.reason:
                details.append(
                    f"{condition.type}: {condition.reason} {condition.message or ''}"
                )
        return "; ".join(details)

    def copy_to_container(
        self,
        source: str | Path,
        destination: str,
        *,
        timeout: int = 300,
    ) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        parent = str(destination_path.parent)
        arcname = destination_path.name
        if (
            not destination_path.is_absolute()
            or not arcname
            or ".." in destination_path.parts
        ):
            raise ValueError("copy destination must be an absolute non-root path")
        destination = str(destination_path)
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as stream:
            stream.add(source_path, arcname=arcname)
        archive_bytes = archive.getvalue()
        digest = sha256(archive_bytes).hexdigest()
        payload = base64.b64encode(archive_bytes).decode()
        staging = f"/tmp/frognano-copy-{uuid.uuid4().hex}"
        archive_path = f"{staging}/archive.tar.gz"
        content_path = f"{staging}/content"
        checksum = shlex.quote(f"{digest}  {archive_path}")
        cleanup = shlex.quote(f"rm -rf -- {shlex.quote(staging)}")
        # A bounded input avoids depending on stdin half-close support in the
        # negotiated Kubernetes WebSocket protocol.
        script = (
            f"set -o pipefail; trap {cleanup} EXIT; "
            f"mkdir -m 700 {shlex.quote(staging)} && "
            f"mkdir {shlex.quote(content_path)} && "
            f"head -c {len(payload)} | base64 -d > {shlex.quote(archive_path)} && "
            f"printf '%s\\n' {checksum} | sha256sum -c - && "
            f"tar -xzf {shlex.quote(archive_path)} -C {shlex.quote(content_path)} && "
            f"mkdir -p {shlex.quote(parent)} && "
            f"rm -rf -- {shlex.quote(destination)} && "
            f"mv -- {shlex.quote(f'{content_path}/{arcname}')} {shlex.quote(destination)}"
        )
        output, exit_code = self._exec(
            ["/bin/bash", "-c", script],
            timeout=timeout,
            operation=f"copy into {destination}",
            input_text=payload,
        )
        if exit_code != 0:
            raise RuntimeError(
                f"failed to copy into pod ({exit_code}): {output[-2000:]}"
            )

    def get_task_instruction(self) -> str:
        return str(self.task["instruction"])

    def _has_git_checkout(self) -> bool:
        if self.task.get("require_git_patch", True):
            return True
        output, exit_code = self.run(
            "command -v git >/dev/null 2>&1 && "
            f"git -C {shlex.quote(str(self.task['repo_path']))} "
            "rev-parse --is-inside-work-tree",
            timeout=30,
            workdir="/",
        )
        if exit_code == 0 and output.strip() == "true":
            return True
        self.logger.info(
            "Skipping optional Git artifact at %s (probe exit %s): %s",
            self.task["repo_path"],
            exit_code,
            output[-1000:],
        )
        return False

    def get_patch(self) -> str:
        if not self._has_git_checkout():
            return ""
        repo = shlex.quote(str(self.task["repo_path"]))
        output, exit_code = self.run(
            f"git -C {repo} add -A 2>/dev/null && "
            f"git -C {repo} diff --cached --binary 2>/dev/null; "
            f"rc=$?; git -C {repo} reset --mixed HEAD >/dev/null 2>&1; "
            f"exit $rc",
            workdir="/",
        )
        if exit_code != 0:
            raise RuntimeError(f"failed to capture patch: {output[-2000:]}")
        return output

    def compute_reward(self) -> tuple[float, str]:
        if self.task.get("verifier_protocol") == "patch_eval_verified":
            return self._compute_patch_eval_verified_reward()
        if self.task.get("verifier_network_mode") == "no-network":
            self._enable_network_isolation()
        output, exit_code = self.run(
            "rm -rf /tests /logs/verifier && " "mkdir -p /tests /logs/verifier",
            workdir="/",
        )
        if exit_code != 0:
            raise RuntimeError(f"could not prepare verifier directories: {output}")
        # Harbor verifiers clean untracked files; preserve agent-created files
        # after the non-mutating patch capture has reset the index.
        if self._has_git_checkout():
            output, exit_code = self.run("git add -A", timeout=120)
            if exit_code != 0:
                raise RuntimeError(
                    f"could not stage agent changes for grading: {output}"
                )
        self.copy_to_container(self.task["tests_dir"], "/tests")
        output, _ = self.run(
            "bash /tests/test.sh",
            timeout=int(self.task["verifier_timeout_sec"]),
            workdir=str(self.task["repo_path"]),
        )
        marker = self.task.get("verifier_success_marker")
        if marker and marker not in output:
            raise RuntimeError(
                "Harbor verifier exited before producing its completion marker"
            )
        reward_json, _ = self.run(
            "cat /logs/verifier/reward.json 2>/dev/null || true",
            workdir="/",
        )
        if reward_json.strip():
            payload = json.loads(reward_json)
            if isinstance(payload, dict):
                value = payload.get("reward")
                if value is None and len(payload) == 1:
                    value = next(iter(payload.values()))
            else:
                value = payload
            return _parse_reward(value, "reward.json"), output
        reward_text, _ = self.run(
            "cat /logs/verifier/reward.txt 2>/dev/null || true",
            workdir="/",
        )
        if reward_text.strip():
            return _parse_reward(reward_text.strip(), "reward.txt"), output
        raise RuntimeError("Harbor verifier did not write a reward")

    def _detect_repo_path(self) -> None:
        repo_name = str(self.task.get("repo_name") or "")
        candidates = [
            f"/workspace/{repo_name}",
            f"/workspace/{repo_name.lower()}",
            "/workspace",
        ]
        for candidate in dict.fromkeys(candidates):
            _, exit_code = self.run(
                f"test -d {shlex.quote(candidate)}/.git",
                timeout=30,
                workdir="/",
            )
            if exit_code == 0:
                self.task["repo_path"] = candidate
                return
        output, exit_code = self.run(
            "find /workspace -mindepth 2 -maxdepth 3 -type d -name .git "
            "2>/dev/null | head -n 2",
            timeout=60,
            workdir="/",
        )
        candidates = [
            line.rsplit("/.git", 1)[0] for line in output.splitlines() if line.strip()
        ]
        if exit_code == 0 and len(candidates) == 1:
            self.task["repo_path"] = candidates[0]
            return
        raise RuntimeError(
            f"could not locate PatchEval Verified Git repository for {repo_name!r}"
        )

    def _hide_workspace_payload(self) -> None:
        workdir = str(self.task["repo_path"]).rstrip("/")
        if workdir == "/workspace":
            command = "rm -f /workspace/fix.patch"
        elif workdir.startswith("/workspace/"):
            top_name = workdir.removeprefix("/workspace/").split("/", 1)[0]
            command = (
                f"find /workspace -mindepth 1 -maxdepth 1 "
                f"! -name {shlex.quote(top_name)} -exec rm -rf -- {{}} +"
            )
        else:
            raise RuntimeError(
                f"PatchEval Verified workdir is outside /workspace: {workdir}"
            )
        output, exit_code = self.run(command, timeout=300, workdir="/")
        if exit_code != 0:
            raise RuntimeError(
                f"failed to hide PatchEval Verified verifier payload: {output[-2000:]}"
            )

    def _compute_patch_eval_verified_reward(self) -> tuple[float, str]:
        patch = self.get_patch()
        self.recreate(verifier=True)
        with tempfile.TemporaryDirectory(
            prefix="frognano-patch-eval-verified-"
        ) as directory:
            patch_path = Path(directory) / "fix.patch"
            patch_path.write_text(patch, encoding="utf-8")
            self.copy_to_container(patch_path, "/workspace/fix.patch")
        output, exit_code = self.run(
            "bash fix-run.sh",
            timeout=int(self.task["verifier_timeout_sec"]),
            workdir="/workspace",
        )
        return (1.0 if exit_code == 0 else 0.0), output

    def _enable_network_isolation(self) -> None:
        client = self._client_module
        policy = client.V1NetworkPolicy(
            metadata=client.V1ObjectMeta(name=self._network_policy_name),
            spec=client.V1NetworkPolicySpec(
                pod_selector=client.V1LabelSelector(
                    match_labels={
                        "app.kubernetes.io/name": "frognano-eval",
                        "frognano-run": _run_label(self.pod_name),
                    }
                ),
                policy_types=["Egress"],
                egress=[],
            ),
        )
        try:
            self._networking.create_namespaced_network_policy(self.namespace, policy)
        except client.exceptions.ApiException as exc:
            if exc.status != 409:
                raise
        time.sleep(2)

    def recreate(self, *, verifier: bool = False) -> None:
        old_pod_name = self.pod_name
        if self.config.keep_pods:
            self.logger.warning(
                "Deleting pod %s for recreation despite keep_pods; "
                "old execution must stop before replay",
                old_pod_name,
            )
        self._delete_resources(grace_period_seconds=1)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                self._core.read_namespaced_pod(
                    old_pod_name, self.namespace, _request_timeout=10
                )
            except self._client_module.exceptions.ApiException as exc:
                if exc.status == 404:
                    break
                raise
            time.sleep(1)
        else:
            raise TimeoutError(f"pod {old_pod_name} was not deleted for recovery")
        self._recovery_count += 1
        self.pod_name = _pod_name(
            self._pod_prefix,
            str(self.task["instance_id"]),
            f"{self._run_id}-recovery-{self._recovery_count}",
        )
        self._network_policy_name = f"{self.pod_name}-deny-egress"
        self._verifier_mode = verifier
        self._create_pod()

    def close(self) -> None:
        if self.config.keep_pods:
            self.logger.info("Keeping pod %s", self.pod_name)
            return
        self._delete_resources()

    def _delete_resources(self, *, grace_period_seconds: int = 0) -> None:
        client = self._client_module
        try:
            self._networking.delete_namespaced_network_policy(
                self._network_policy_name,
                self.namespace,
                body=client.V1DeleteOptions(),
            )
        except client.exceptions.ApiException as exc:
            if exc.status != 404:
                self.logger.warning("Failed to delete network policy: %s", exc)
        try:
            self._core.delete_namespaced_pod(
                self.pod_name,
                self.namespace,
                body=client.V1DeleteOptions(grace_period_seconds=grace_period_seconds),
            )
        except client.exceptions.ApiException as exc:
            if exc.status != 404:
                self.logger.warning("Failed to delete pod: %s", exc)


def _parse_command_state(output: str) -> dict[str, Any]:
    value = output.strip()
    if value in {"missing", "running"}:
        return {"state": value}
    if value.startswith("error "):
        return {"state": "error", "error": value.removeprefix("error ")}
    match = re.fullmatch(r"completed (\d+) ([01]) (\d+) ([0-9a-f]{64})", value)
    if match is None or int(match[1]) > 255:
        raise ValueError("completion record has invalid exit or output metadata")
    return {
        "state": "completed",
        "returncode": int(match[1]),
        "timed_out": match[2] == "1",
        "output_size": int(match[3]),
        "sha256": match[4],
    }


def _decode_output_chunk(result: dict[str, Any], offset: int, length: int) -> bytes:
    if (
        type(result.get("offset")) is not int
        or result["offset"] != offset
        or type(result.get("length")) is not int
        or result["length"] != length
        or not isinstance(result.get("data"), str)
    ):
        raise ValueError("output chunk has an incorrect offset or byte count")
    output = base64.b64decode(result["data"], validate=True)
    if len(output) != length:
        raise ValueError("output chunk was truncated")
    return output


def _exception_chain(error: BaseException) -> list[BaseException]:
    causes = []
    seen = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        causes.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return causes


def _parse_reward(value: Any, source: str) -> float:
    reward = float(value)
    if reward < 0:
        raise RuntimeError(
            f"Harbor verifier returned negative reward {reward} from {source}; "
            "the verifier did not produce a valid benchmark result"
        )
    return reward


def _pod_name(prefix: str, task_id: str, run_id: str) -> str:
    raw = f"{prefix}-{task_id}-{run_id}".lower()
    value = _SAFE_NAME_RE.sub("-", raw).strip("-")
    suffix = sha256(raw.encode()).hexdigest()[:10]
    return f"{value[:52].rstrip('-')}-{suffix}"


def _run_label(pod_name: str) -> str:
    return sha256(pod_name.encode()).hexdigest()[:20]


def _quantity(value: str) -> str:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([GMK])", value.strip(), re.I)
    if match:
        return f"{match.group(1)}{match.group(2).upper()}i"
    return value
