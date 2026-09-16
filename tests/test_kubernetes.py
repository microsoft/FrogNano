import base64
import json
import logging
import subprocess
from types import SimpleNamespace

import pytest

from frognano.config import KubernetesConfig
from frognano.runtimes.kubernetes import (
    KubernetesTaskRuntime,
    _pod_name,
    _run_label,
)


def test_pod_name_is_stable_unique_and_within_limit() -> None:
    first = _pod_name("sweb-v", "a" * 100, "run-1")
    second = _pod_name("sweb-v", "a" * 100, "run-2")

    assert first != second
    assert first == _pod_name("sweb-v", "a" * 100, "run-1")
    assert len(first) <= 63


def test_run_label_is_valid_and_stable() -> None:
    label = _run_label("-invalid-looking-pod-name")

    assert label == _run_label("-invalid-looking-pod-name")
    assert len(label) == 20
    assert label.isalnum()


@pytest.mark.parametrize("override", [None, "32Gi"])
@pytest.mark.parametrize("pod_lifetime", [None, 10800])
@pytest.mark.parametrize("verifier_timeout", [600, 3600, 12000])
def test_pod_preserves_requests_and_applies_limit_overrides(
    override, pod_lifetime, verifier_timeout
) -> None:
    from kubernetes import client

    runtime = object.__new__(KubernetesTaskRuntime)
    runtime._client_module = client
    runtime.config = KubernetesConfig(
        memory_limit=override, pod_lifetime_sec=pod_lifetime
    )
    runtime.namespace = "default"
    runtime.pod_name = "preflight"
    runtime.logger = logging.getLogger(__name__)
    runtime.task = {
        "docker_image": "example:1",
        "repo_path": "/workspace/project",
        "agent_timeout_sec": 10800,
        "verifier_timeout_sec": verifier_timeout,
        "resources": {
            "cpu": "0.5",
            "cpu_limit": None,
            "memory": "1G",
            "memory_limit": "8G",
        },
    }
    pods = []
    runtime._core = SimpleNamespace(
        create_namespaced_pod=lambda namespace, pod: pods.append(pod),
        read_namespaced_pod=lambda *args: SimpleNamespace(
            status=SimpleNamespace(phase="Running")
        ),
    )
    runtime._bootstrap_executor = lambda: None
    runtime._initialize = lambda: None

    runtime._create_pod()

    resources = pods[0].spec.containers[0].resources
    assert resources.requests["memory"] == "1Gi"
    assert resources.limits["memory"] == (override or "8Gi")
    assert "cpu" not in resources.limits
    assert pods[0].spec.active_deadline_seconds == (
        pod_lifetime or 10800 + verifier_timeout + 600
    )


def test_get_patch_resets_index_after_capture() -> None:
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.task = {"repo_path": "/testbed"}
    commands = []

    def run(command, *, timeout=120, workdir=None):
        commands.append(command)
        return "diff --git a/file b/file", 0

    runtime.run = run

    assert runtime.get_patch().startswith("diff --git")
    assert "git -C /testbed reset --mixed HEAD" in commands[0]


@pytest.mark.parametrize(
    "probe,code,has_patch",
    [("", 127, False), ("not a git repository", 128, False), ("true\n", 0, True)],
)
def test_optional_patch_capture_handles_non_git_tasks(probe, code, has_patch) -> None:
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.task = {"repo_path": "/app", "require_git_patch": False}
    runtime.logger = logging.getLogger(__name__)

    def run(command, **kwargs):
        if "rev-parse" in command:
            return probe, code
        return "diff --git a/file b/file", 0

    runtime.run = run
    assert bool(runtime.get_patch()) == has_patch


def test_optional_patch_capture_does_not_hide_transport_failure() -> None:
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.task = {"repo_path": "/app", "require_git_patch": False}

    def run(*args, **kwargs):
        raise RuntimeError("pod unreachable")

    runtime.run = run
    with pytest.raises(RuntimeError, match="pod unreachable"):
        runtime.get_patch()


def test_non_git_task_can_be_graded() -> None:
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.task = {
        "repo_path": "/app",
        "require_git_patch": False,
        "tests_dir": "/host/tests",
        "verifier_timeout_sec": 10,
    }
    runtime.logger = logging.getLogger(__name__)
    runtime.copy_to_container = lambda *args: None

    def run(command, **kwargs):
        assert command != "git add -A"
        if "rev-parse" in command:
            return "not a git repository", 128
        if command == "bash /tests/test.sh":
            return "valid failed tests", 1
        if "reward.txt" in command:
            return "0\n", 0
        return "", 0

    runtime.run = run
    assert runtime.compute_reward() == (0.0, "valid failed tests")


def test_compute_reward_requires_verifier_marker() -> None:
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.task = {
        "repo_path": "/testbed",
        "tests_dir": "/host/tests",
        "verifier_timeout_sec": 10,
        "verifier_network_mode": "public",
        "verifier_success_marker": "completed marker",
    }
    runtime.copy_to_container = lambda *args: None

    def run(command, *, timeout=120, workdir=None):
        if command == "bash /tests/test.sh":
            return "incomplete output", 0
        return "", 0

    runtime.run = run

    try:
        runtime.compute_reward()
    except RuntimeError as exc:
        assert "completion marker" in str(exc)
    else:
        raise AssertionError("missing verifier marker should fail")


def test_compute_reward_accepts_completed_zero_reward() -> None:
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.task = {
        "repo_path": "/testbed",
        "tests_dir": "/host/tests",
        "verifier_timeout_sec": 10,
        "verifier_network_mode": "public",
        "verifier_success_marker": "completed marker",
    }
    runtime.copy_to_container = lambda *args: None

    def run(command, *, timeout=120, workdir=None):
        if command == "bash /tests/test.sh":
            return "tests failed\ncompleted marker", 1
        if "reward.txt" in command:
            return "0\n", 0
        return "", 0

    runtime.run = run

    assert runtime.compute_reward() == (
        0.0,
        "tests failed\ncompleted marker",
    )


@pytest.mark.parametrize("reward_format", ["json-object", "json-scalar", "text"])
@pytest.mark.parametrize("reward", [-1, -0.25, 0, 1])
def test_compute_reward_rejects_negative_sentinels(reward_format, reward) -> None:
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.task = {
        "repo_path": "/testbed",
        "tests_dir": "/host/tests",
        "verifier_timeout_sec": 10,
    }
    runtime.copy_to_container = lambda *args: None
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command == "bash /tests/test.sh":
            return "verifier output", 0 if reward > 0 else 1
        if "reward.json" in command:
            if reward_format == "json-object":
                return json.dumps({"reward": reward}), 0
            if reward_format == "json-scalar":
                return json.dumps(reward), 0
        if "reward.txt" in command:
            return str(reward if reward_format == "text" else 1), 0
        return "", 0

    runtime.run = run

    if reward < 0:
        with pytest.raises(RuntimeError, match="negative reward"):
            runtime.compute_reward()
    else:
        assert runtime.compute_reward() == (float(reward), "verifier output")
    if reward_format != "text":
        assert not any("reward.txt" in command for command in commands)


def test_compute_reward_stages_new_files_before_verifier(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qm",
            "baseline",
            "--allow-empty",
        ],
        check=True,
    )
    new_file = tmp_path / "new.py"
    new_file.write_text("value = 1\n", encoding="utf-8")
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.task = {
        "repo_path": str(tmp_path),
        "tests_dir": "/host/tests",
        "verifier_timeout_sec": 10,
    }
    runtime.copy_to_container = lambda *args: None

    def run(command, **kwargs):
        if command in {"git add -A", "bash /tests/test.sh"}:
            actual = "git clean -fd" if command.startswith("bash") else command
            subprocess.run(["bash", "-c", actual], cwd=tmp_path, check=True)
        if "reward.txt" in command:
            return "1\n", 0
        return "", 0

    runtime.run = run

    assert runtime.compute_reward()[0] == 1
    assert new_file.read_text(encoding="utf-8") == "value = 1\n"


@pytest.mark.parametrize("exit_code,expected_reward", [(0, 1.0), (1, 0.0)])
def test_compute_patch_eval_verified_reward_uses_fresh_image(
    exit_code, expected_reward
) -> None:
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.task = {
        "repo_path": "/workspace/project",
        "verifier_protocol": "patch_eval_verified",
        "verifier_timeout_sec": 600,
    }
    runtime._verifier_mode = False
    recreated = []
    copied = []
    commands = []
    runtime.get_patch = lambda: "diff --git a/file b/file\n"
    runtime.recreate = lambda *, verifier=False: recreated.append(verifier)
    runtime.copy_to_container = lambda source, destination: copied.append(
        (source.read_text(encoding="utf-8"), destination)
    )

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return "validation output", exit_code

    runtime.run = run

    reward, output = runtime.compute_reward()

    assert reward == expected_reward
    assert output == "validation output"
    assert recreated == [True]
    assert copied == [("diff --git a/file b/file\n", "/workspace/fix.patch")]
    assert commands == [("bash fix-run.sh", {"timeout": 600, "workdir": "/workspace"})]


def test_detect_patch_eval_verified_repo_path() -> None:
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.task = {
        "repo_name": "Project",
        "repo_path": "/workspace",
    }

    def run(command, **kwargs):
        return "", 0 if "/workspace/project/.git" in command else 1

    runtime.run = run
    runtime._detect_repo_path()

    assert runtime.task["repo_path"] == "/workspace/project"


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("directory", [False, True])
def test_copy_to_container_verifies_transfer(tmp_path, corrupt, directory) -> None:
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime._core = type(
        "Core",
        (),
        {"connect_get_namespaced_pod_exec": object()},
    )()
    runtime.pod_name = "pod"
    runtime.namespace = "default"
    responses = []
    destination = tmp_path / "destination"
    destination.write_text("old gold patch", encoding="utf-8")

    class Response:
        def __init__(self, command):
            self.command = command
            self.written = []
            self.closed = False

        def write_stdin(self, value):
            self.written.append(value)

        def run_forever(self, timeout):
            payload = "".join(self.written)
            assert f"head -c {len(payload)}" in self.command[-1]
            if corrupt:
                archive = bytearray(base64.b64decode(payload))
                archive[-1] ^= 1
                payload = base64.b64encode(archive).decode()
            self.result = subprocess.run(
                self.command,
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        def is_open(self):
            return False

        def read_all(self):
            return self.result.stdout + self.result.stderr

        @property
        def returncode(self):
            return self.result.returncode

        def close(self):
            self.closed = True

    def stream(*args, **kwargs):
        response = Response(kwargs["command"])
        responses.append(response)
        return response

    runtime._stream = stream
    source = tmp_path / "source"
    if directory:
        source.mkdir()
        (source / "file.py").write_text("agent patch", encoding="utf-8")
    else:
        source.write_text("agent patch", encoding="utf-8")

    if corrupt:
        with pytest.raises(RuntimeError, match="failed to copy into pod"):
            runtime.copy_to_container(source, str(destination))
        assert destination.read_text(encoding="utf-8") == "old gold patch"
    else:
        runtime.copy_to_container(source, str(destination))
        result = destination / "file.py" if directory else destination
        assert result.read_text(encoding="utf-8") == "agent patch"

    assert responses[0].closed


@pytest.mark.parametrize("open_stream,code", [(True, None), (False, None), (False, 1)])
def test_copy_to_container_requires_successful_completion(tmp_path, open_stream, code):
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime._core = SimpleNamespace(connect_get_namespaced_pod_exec=object())
    runtime.pod_name = "pod"
    runtime.namespace = "default"
    runtime._pod_status_detail = lambda: "phase=Running"
    closed = []
    response = SimpleNamespace(
        write_stdin=lambda data: None,
        run_forever=lambda **kwargs: None,
        is_open=lambda: open_stream,
        read_all=lambda: "error",
        returncode=code,
        close=lambda: closed.append(True),
    )
    runtime._stream = lambda *args, **kwargs: response
    source = tmp_path / "source"
    source.write_text("agent patch", encoding="utf-8")

    with pytest.raises((RuntimeError, TimeoutError)):
        runtime.copy_to_container(source, "/workspace/fix.patch")

    assert closed == [True]


@pytest.mark.parametrize("keep_pods", [False, True])
@pytest.mark.parametrize("verifier", [False, True])
def test_recreate_replaces_pod_with_new_name(keep_pods, verifier) -> None:
    class ApiException(Exception):
        def __init__(self, status):
            self.status = status

    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.task = {"instance_id": "task"}
    runtime.namespace = "default"
    runtime.pod_name = "old-pod"
    runtime._pod_prefix = "sweb-v"
    runtime._run_id = "run"
    runtime._recovery_count = 0
    runtime._verifier_mode = True
    runtime.logger = logging.getLogger(__name__)
    runtime.config = SimpleNamespace(keep_pods=keep_pods)
    runtime._client_module = SimpleNamespace(
        exceptions=SimpleNamespace(ApiException=ApiException)
    )
    runtime._core = SimpleNamespace(
        read_namespaced_pod=lambda *args, **kwargs: (_ for _ in ()).throw(
            ApiException(404)
        )
    )
    deleted = []
    created = []
    runtime._delete_resources = lambda **kwargs: deleted.append(
        (runtime.pod_name, kwargs)
    )
    runtime._create_pod = lambda: created.append(runtime.pod_name)

    runtime.recreate(verifier=verifier)

    assert deleted == [("old-pod", {"grace_period_seconds": 1})]
    assert created == [runtime.pod_name]
    assert runtime.pod_name != "old-pod"
    assert runtime._network_policy_name == f"{runtime.pod_name}-deny-egress"
    assert runtime._verifier_mode is verifier


def test_recreate_cannot_replay_while_old_pod_still_exists(monkeypatch):
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.config = SimpleNamespace(keep_pods=False)
    runtime.pod_name = "old-pod"
    runtime.namespace = "default"
    runtime._client_module = SimpleNamespace(
        exceptions=SimpleNamespace(ApiException=RuntimeError)
    )
    events = []
    runtime._delete_resources = lambda **kwargs: events.append("delete")
    runtime._create_pod = lambda: events.append("create")
    runtime._core = SimpleNamespace(
        read_namespaced_pod=lambda *args, **kwargs: events.append("read")
    )
    times = iter([0, 0, 61])
    monkeypatch.setattr(
        "frognano.runtimes.kubernetes.time.monotonic", lambda: next(times)
    )
    monkeypatch.setattr("frognano.runtimes.kubernetes.time.sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="was not deleted for recovery"):
        runtime.recreate()
    assert events == ["delete", "read"]
    assert runtime.pod_name == "old-pod"
