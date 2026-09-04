from types import SimpleNamespace

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


def test_compute_patch_eval_reward_uses_fresh_image() -> None:
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.task = {
        "repo_path": "/workspace/project",
        "verifier_protocol": "patch_eval",
        "verifier_timeout_sec": 600,
    }
    runtime._verifier_mode = False
    recreated = []
    copied = []
    runtime.get_patch = lambda: "diff --git a/file b/file\n"
    runtime.recreate = lambda: recreated.append(runtime._verifier_mode)
    runtime.copy_to_container = lambda source, destination: copied.append(
        (source.read_text(encoding="utf-8"), destination)
    )
    runtime.run = lambda command, **kwargs: ("validation passed", 0)

    reward, output = runtime.compute_reward()

    assert reward == 1.0
    assert output == "validation passed"
    assert recreated == [True]
    assert copied == [("diff --git a/file b/file\n", "/workspace/fix.patch")]


def test_detect_patch_eval_repo_path() -> None:
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


def test_copy_to_container_supports_wsclient_without_close_stdin(
    tmp_path,
) -> None:
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime._core = type(
        "Core",
        (),
        {"connect_get_namespaced_pod_exec": object()},
    )()
    runtime.pod_name = "pod"
    runtime.namespace = "default"
    written = []

    class Response:
        def write_stdin(self, value):
            written.append(value)

        def close(self):
            written.append("closed")

    runtime._stream = lambda *args, **kwargs: Response()
    runtime.run = lambda *args, **kwargs: ("", 0)
    source = tmp_path / "runner.py"
    source.write_text("print('ok')", encoding="utf-8")

    runtime.copy_to_container(source, "/tmp/runner.py")

    assert written[-1] == "closed"
    assert len(written) >= 2


def test_recreate_replaces_pod_with_new_name(monkeypatch) -> None:
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
    runtime.config = SimpleNamespace(keep_pods=False)
    runtime._client_module = SimpleNamespace(
        exceptions=SimpleNamespace(ApiException=ApiException)
    )
    runtime._core = SimpleNamespace(
        read_namespaced_pod=lambda *args: (_ for _ in ()).throw(ApiException(404))
    )
    deleted = []
    created = []
    runtime._delete_resources = lambda: deleted.append(runtime.pod_name)
    runtime._create_pod = lambda: created.append(runtime.pod_name)

    runtime.recreate()

    assert deleted == ["old-pod"]
    assert created == [runtime.pod_name]
    assert runtime.pod_name != "old-pod"
    assert runtime._network_policy_name == f"{runtime.pod_name}-deny-egress"
