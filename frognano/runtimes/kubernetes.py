"""Kubernetes task execution runtime."""

from __future__ import annotations

import base64
import io
import json
import logging
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

from frognano.config import KubernetesConfig

_SAFE_NAME_RE = re.compile(r"[^a-z0-9-]+")


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
        client = self._client_module
        resources = self.task.get("resources") or {}
        quantities = {
            "cpu": str(resources.get("cpu") or "0.5"),
            "memory": _quantity(str(resources.get("memory") or "8G")),
            "ephemeral-storage": _quantity(str(resources.get("storage") or "20G")),
        }
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
                limits=quantities,
            ),
        )
        spec_kwargs: dict[str, Any] = {
            "containers": [container],
            "restart_policy": "Never",
            "active_deadline_seconds": (
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
                self._initialize()
                if self.task.get("agent_network_mode") == "no-network":
                    self._enable_network_isolation()
                return
            if phase in {"Failed", "Succeeded"}:
                raise RuntimeError(
                    f"pod {self.pod_name} entered terminal phase {phase}"
                )
            time.sleep(2)
        raise TimeoutError(f"pod {self.pod_name} did not become ready")

    def _initialize(self) -> None:
        self.run("mkdir -p /logs/verifier /logs/agent /tests /solution", workdir="/")
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
        timeout: int = 120,
        workdir: str | None = None,
    ) -> tuple[str, int]:
        sentinel = f"__FROGNANO_RC_{uuid.uuid4().hex}__"
        directory = workdir or str(self.task["repo_path"])
        script = (
            f"cd {shlex.quote(directory)} && ({command}); "
            f"rc=$?; printf '\\n{sentinel}%s\\n' \"$rc\""
        )
        output = self._stream(
            self._core.connect_get_namespaced_pod_exec,
            self.pod_name,
            self.namespace,
            container="task",
            command=["/bin/bash", "-lc", script],
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _request_timeout=timeout,
        )
        marker = re.search(rf"\n{re.escape(sentinel)}(\d+)\n?$", output)
        if marker is None:
            raise RuntimeError(f"pod command returned no exit marker: {output[-2000:]}")
        return output[: marker.start()], int(marker.group(1))

    def copy_to_container(
        self,
        source: str | Path,
        destination: str,
        *,
        timeout: int = 300,
    ) -> None:
        source_path = Path(source)
        parent = str(Path(destination).parent)
        arcname = Path(destination).name
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as stream:
            stream.add(source_path, arcname=arcname)
        payload = base64.b64encode(archive.getvalue()).decode()
        script = (
            f"mkdir -p {shlex.quote(parent)} && "
            f"base64 -d | tar -xzf - -C {shlex.quote(parent)}"
        )
        response = self._stream(
            self._core.connect_get_namespaced_pod_exec,
            self.pod_name,
            self.namespace,
            container="task",
            command=["/bin/bash", "-lc", script],
            stderr=True,
            stdin=True,
            stdout=True,
            tty=False,
            _preload_content=False,
            _request_timeout=timeout,
        )
        try:
            for index in range(0, len(payload), 65536):
                response.write_stdin(payload[index : index + 65536])
        finally:
            response.close()
        output, exit_code = self.run(
            f"test -e {shlex.quote(destination)}",
            timeout=30,
            workdir="/",
        )
        if exit_code != 0:
            raise RuntimeError(f"failed to copy into pod: {output}")

    def get_task_instruction(self) -> str:
        return str(self.task["instruction"])

    def get_patch(self) -> str:
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
        if self.task.get("verifier_protocol") == "patch_eval":
            return self._compute_patch_eval_reward()
        if self.task.get("verifier_network_mode") == "no-network":
            self._enable_network_isolation()
        self.run(
            "rm -rf /tests /logs/verifier && " "mkdir -p /tests /logs/verifier",
            workdir="/",
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
            return float(value), output
        reward_text, _ = self.run(
            "cat /logs/verifier/reward.txt 2>/dev/null || true",
            workdir="/",
        )
        if reward_text.strip():
            return float(reward_text.strip()), output
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
            f"could not locate PatchEval Git repository for {repo_name!r}"
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
            raise RuntimeError(f"PatchEval workdir is outside /workspace: {workdir}")
        output, exit_code = self.run(command, timeout=300, workdir="/")
        if exit_code != 0:
            raise RuntimeError(
                f"failed to hide PatchEval verifier payload: {output[-2000:]}"
            )

    def _compute_patch_eval_reward(self) -> tuple[float, str]:
        patch = self.get_patch()
        self._verifier_mode = True
        self.recreate()
        with tempfile.TemporaryDirectory(prefix="frognano-patch-eval-") as directory:
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

    def recreate(self) -> None:
        old_pod_name = self.pod_name
        if not self.config.keep_pods:
            self._delete_resources()
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                try:
                    self._core.read_namespaced_pod(old_pod_name, self.namespace)
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
        self._create_pod()

    def close(self) -> None:
        if self.config.keep_pods:
            self.logger.info("Keeping pod %s", self.pod_name)
            return
        self._delete_resources()

    def _delete_resources(self) -> None:
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
                body=client.V1DeleteOptions(grace_period_seconds=0),
            )
        except client.exceptions.ApiException as exc:
            if exc.status != 404:
                self.logger.warning("Failed to delete pod: %s", exc)


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
