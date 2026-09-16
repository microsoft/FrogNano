import base64
import io
import json
import logging
import shlex
import shutil
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException
from kubernetes.stream.ws_client import ERROR_CHANNEL, WSClient

from frognano.runtimes.errors import CommandTimeoutError, PodExecutionError
from frognano.runtimes.kubernetes import (
    _CHECK_COREUTILS,
    _INSTALL_COREUTILS,
    KubernetesTaskRuntime,
)


@pytest.fixture
def runtime(tmp_path):
    runtime = object.__new__(KubernetesTaskRuntime)
    runtime.task = {"repo_path": "/testbed"}
    runtime.pod_name = "pod"
    runtime.namespace = "default"
    runtime.logger = logging.getLogger(__name__)
    runtime._core = SimpleNamespace(connect_get_namespaced_pod_exec=object())
    runtime._command_output_root = str(tmp_path / "outputs")
    runtime._pod_status_detail = lambda: "phase=Running"
    runtime.transfers = []

    def copy(source, destination):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        runtime.transfers.append((destination, path.read_bytes()))

    runtime.copy_to_container = copy
    return runtime


class FileProtocol:
    def __init__(self, output=b"", code=0, *, timed_out=False):
        self.output = output
        self.result = {
            "state": "completed",
            "returncode": code,
            "timed_out": timed_out,
            "output_size": len(output),
            "sha256": sha256(output).hexdigest(),
        }
        self.calls = []
        self.requests = {}

    def execute(self, command, *, timeout, operation, input_text=None):
        assert input_text is None
        assert command[:2] == ["/bin/bash", "-c"]
        action = _operation(command)
        directory = Path(command[4])
        command_id = directory.name
        arguments = command[5:]
        self.calls.append((action, command_id, arguments))
        request_file = directory / "command.sh"
        if action == "execute":
            request = {
                "command": request_file.read_bytes().decode("utf-8"),
                "interpreter": command[6:-1],
                "workdir": shlex.split(command[-1])[2],
                "timeout": float(command[5]),
            }
            if command_id in self.requests:
                assert self.requests[command_id] == request
            self.requests[command_id] = request
            return "", 0
        if action == "status":
            result = (
                self.result if command_id in self.requests else {"state": "missing"}
            )
            if result["state"] == "completed":
                return (
                    f"completed {result['returncode']} "
                    f"{int(result['timed_out'])} {result['output_size']} "
                    f"{result['sha256']}\n",
                    0,
                )
            return f"{result['state']} {result.get('error', '')}\n", 0
        if action == "read":
            offset, length = map(int, arguments)
            chunk = self.output[offset : offset + length]
            return base64.b64encode(chunk).decode("ascii"), 0
        assert action == "cleanup"
        if directory.exists():
            shutil.rmtree(directory)
        return "", 0

    def count(self, action):
        return sum(call[0] == action for call in self.calls)


def _operation(command):
    return command[3].removeprefix("frognano-")


@pytest.mark.parametrize("code", [0, 1, 2, 124, 127, 137, 143, 255])
def test_file_backed_command_returns_complete_output_and_exit_code(runtime, code):
    output = b"stdout\r\nstderr\n\n__FROGNANO_RC_not_a_protocol__0\n"
    protocol = FileProtocol(output, code)
    runtime._exec = protocol.execute

    assert runtime.run("command") == (output.decode(), code)
    assert protocol.count("execute") == protocol.count("cleanup") == 1


@pytest.mark.parametrize("interpreter", [None, ["/bin/bash", "-lc"]])
def test_file_backed_command_preserves_text_interpreter_and_cwd(runtime, interpreter):
    runtime.task["command_interpreter"] = interpreter
    protocol = FileProtocol()
    runtime._exec = protocol.execute
    command = "  printf '%s\\n' \"$0\"; printf 'end' # trailing comment\n\n"

    assert runtime.run(command, workdir="/a path", timeout=17) == ("", 0)
    assert list(protocol.requests.values()) == [
        {
            "command": command,
            "interpreter": interpreter or ["/bin/bash", "-c"],
            "workdir": "/a path",
            "timeout": 17,
        }
    ]


@pytest.mark.parametrize("interpreter", [[], "", "bash -c", ["bash", ""], [1]])
def test_file_backed_command_rejects_invalid_interpreters(runtime, interpreter):
    runtime.task["command_interpreter"] = interpreter
    with pytest.raises(ValueError, match="argument list"):
        runtime.run("true")


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan")])
def test_file_backed_command_rejects_invalid_deadlines(runtime, timeout):
    with pytest.raises(ValueError, match="positive and finite"):
        runtime.run("true", timeout=timeout)


@pytest.mark.parametrize("command", [None, 1, True, [], "printf x\0"])
def test_invalid_command_does_not_transfer_input(runtime, command):
    with pytest.raises(ValueError, match="command must be a string"):
        runtime.run(command)
    assert runtime.transfers == []


@pytest.mark.parametrize("size", [0, 1, 65535, 65536, 65537, 300000])
def test_every_request_size_is_transferred_as_a_file(runtime, size):
    protocol = FileProtocol()
    runtime._exec = protocol.execute
    command = "printf '%s' \"$HOME\"; #" + "a" * size + "\r\n\n"

    assert runtime.run(command) == ("", 0)
    assert len(runtime.transfers) == 1
    destination, content = runtime.transfers[0]
    assert destination.startswith(f"{runtime._command_output_root}/")
    assert destination.endswith("/command.sh")
    assert content.decode("utf-8") == command
    request = next(iter(protocol.requests.values()))
    assert request["command"] == command
    assert not Path(destination).exists()


@pytest.mark.parametrize("failure", ["lost", "timeout", "nonzero", "truncated"])
def test_ambiguous_execution_reads_state_without_resending_execution(
    runtime, monkeypatch, failure
):
    protocol = FileProtocol(b"completed once")
    monkeypatch.setattr("frognano.runtimes.kubernetes.time.sleep", lambda _: None)

    def execute(command, **kwargs):
        response = protocol.execute(command, **kwargs)
        if _operation(command) == "execute":
            if failure == "lost":
                raise PodExecutionError("execution response disconnected")
            if failure == "timeout":
                raise CommandTimeoutError("exec response timed out")
            if failure == "nonzero":
                return "wrapper exit response lost", 137
            return "{", 0
        return response

    runtime._exec = execute
    assert runtime.run("mutate") == ("completed once", 0)
    assert len(protocol.requests) == 1
    assert protocol.count("execute") == 1


def test_unconfirmed_execution_is_not_repeated(runtime):
    protocol = FileProtocol()

    def execute(command, **kwargs):
        if _operation(command) == "execute":
            raise PodExecutionError("disconnected before acknowledgement")
        return protocol.execute(command, **kwargs)

    runtime._exec = execute
    with pytest.raises(PodExecutionError, match="no completion record"):
        runtime.run("mutate")
    assert protocol.requests == {}
    assert protocol.count("cleanup") == 0


def test_failed_input_transfer_does_not_execute(runtime):
    protocol = FileProtocol()
    runtime._exec = protocol.execute

    def copy(*args):
        raise PodExecutionError("file transfer interrupted")

    runtime.copy_to_container = copy
    with pytest.raises(PodExecutionError, match="file transfer interrupted"):
        runtime.run("mutate")
    assert protocol.calls == []


@pytest.mark.parametrize("damage", ["transport", "truncated", "checksum"])
def test_large_output_retries_reads_without_reexecuting(runtime, monkeypatch, damage):
    output = b"line\r\n\x00\xff" * 17000 + "end\N{SNOWMAN}".encode()
    protocol = FileProtocol(output)
    monkeypatch.setattr("frognano.runtimes.kubernetes.time.sleep", lambda _: None)
    damaged = False

    def execute(command, **kwargs):
        nonlocal damaged
        result, code = protocol.execute(command, **kwargs)
        if _operation(command) == "read" and not damaged:
            damaged = True
            if damage == "transport":
                raise PodExecutionError("read disconnected")
            chunk = base64.b64decode(result)
            chunk = chunk[:-1] if damage == "truncated" else b"?" + chunk[1:]
            result = base64.b64encode(chunk).decode()
        return result, code

    runtime._exec = execute
    assert runtime.run("emit") == (output.decode("utf-8", errors="replace"), 0)
    assert protocol.count("execute") == 1
    assert protocol.count("read") > 3
    assert all(int(call[2][1]) <= 49152 for call in protocol.calls if call[0] == "read")


def test_utf8_is_decoded_after_reassembling_chunk_boundaries(runtime):
    output = "a" * 49151 + "\N{SNOWMAN}\r\n"
    protocol = FileProtocol(output.encode("utf-8"))
    runtime._exec = protocol.execute

    assert runtime.run("emit") == (output, 0)
    assert protocol.count("read") == 2


def test_corrupt_output_is_never_returned_or_cleaned_up(runtime):
    protocol = FileProtocol(b"right size")
    protocol.result["sha256"] = "0" * 64
    runtime._exec = protocol.execute

    with pytest.raises(PodExecutionError, match="SHA-256 verification"):
        runtime.run("emit")
    assert protocol.count("execute") == 1
    assert protocol.count("read") == 3
    assert protocol.count("cleanup") == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("returncode", None),
        ("returncode", True),
        ("returncode", -9),
        ("returncode", 256),
        ("timed_out", 2),
        ("output_size", -1),
        ("output_size", True),
        ("sha256", "incomplete"),
        ("state", "unknown"),
    ],
)
def test_malformed_completion_is_not_success(runtime, monkeypatch, field, value):
    protocol = FileProtocol()
    protocol.result[field] = value
    runtime._exec = protocol.execute
    monkeypatch.setattr("frognano.runtimes.kubernetes.time.sleep", lambda _: None)

    with pytest.raises(PodExecutionError, match="control requests"):
        runtime.run("mutate")
    assert protocol.count("execute") == 1
    assert protocol.count("cleanup") == 0


@pytest.mark.parametrize("state", ["missing", "error"])
def test_missing_completion_is_not_permission_to_repeat_command(runtime, state):
    protocol = FileProtocol()
    protocol.result = {
        "state": state,
        "error": "shell stopped" if state == "error" else "",
    }
    runtime._exec = protocol.execute

    with pytest.raises(PodExecutionError, match="no completion record"):
        runtime.run("mutate")
    assert protocol.count("execute") == 1
    assert protocol.count("cleanup") == 0


def test_timeout_preserves_full_partial_output(runtime):
    output = b"partial\r\n" * 1000
    protocol = FileProtocol(output, 124, timed_out=True)
    runtime._exec = protocol.execute

    with pytest.raises(CommandTimeoutError, match="timed out after 2s") as error:
        runtime.run("hang", timeout=2)
    assert error.value.output == output.decode()
    assert protocol.count("execute") == protocol.count("cleanup") == 1


def test_missing_deadline_record_requires_pod_recreation(runtime, monkeypatch):
    protocol = FileProtocol()
    protocol.result = {"state": "running"}
    clock = [0.0]
    monkeypatch.setattr("frognano.runtimes.kubernetes.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "frognano.runtimes.kubernetes.time.sleep",
        lambda _: clock.__setitem__(0, clock[0] + 100),
    )

    runtime._exec = protocol.execute
    with pytest.raises(PodExecutionError, match="pod must be recreated before replay"):
        runtime.run("hang", timeout=1)
    assert protocol.count("execute") == 1
    assert protocol.count("cleanup") == 0
    assert protocol.count("cancel") == 0


def test_cleanup_failure_does_not_discard_verified_output(runtime, monkeypatch, caplog):
    protocol = FileProtocol(b"complete", 7)
    monkeypatch.setattr("frognano.runtimes.kubernetes.time.sleep", lambda _: None)

    def execute(command, **kwargs):
        result = protocol.execute(command, **kwargs)
        if _operation(command) == "cleanup":
            raise PodExecutionError("pod disappeared after delivery")
        return result

    runtime._exec = execute
    assert runtime.run("command") == ("complete", 7)
    assert "Could not remove verified command output" in caplog.text


def test_bootstrap_uses_shell_tools_and_never_copies_a_python_supervisor(runtime):
    commands = []
    copies = []

    def execute(command, **kwargs):
        commands.append((command, kwargs))
        if kwargs["operation"] == "shell executor bootstrap":
            return "timeout (GNU coreutils) 9.4\n", 0
        return "/usr/bin/python3\n", 0

    runtime._exec = execute
    runtime.copy_to_container = lambda *args: copies.append(args)

    runtime._bootstrap_executor()

    assert len(commands) == 2
    assert commands[0][0] == ["/bin/bash", "-c", _CHECK_COREUTILS]
    assert "sys.version_info < (3, 6)" in commands[1][0][-1]
    assert copies == []
    assert not hasattr(runtime, "_python_bin")


@pytest.mark.parametrize(
    "output,code",
    [("timeout: command not found", 127), ("BusyBox timeout", 0)],
)
def test_missing_shell_utilities_are_reported_before_initialization(
    runtime, output, code
):
    runtime._exec = lambda *args, **kwargs: (output, code)
    with pytest.raises(RuntimeError, match="requires Bash and GNU coreutils"):
        runtime._bootstrap_executor()
    assert runtime.transfers == []


@pytest.mark.parametrize(
    "initial", [("timeout: command not found", 127), ("BusyBox timeout", 0)]
)
@pytest.mark.parametrize("python_missing", [False, True])
def test_public_bootstrap_installs_and_rechecks_coreutils(
    runtime, initial, python_missing
):
    runtime.task["agent_network_mode"] = "public"
    responses = [
        initial,
        ("coreutils installed", 0),
        ("GNU coreutils verified\n", 0),
    ]
    if python_missing:
        responses.extend([("Python missing", 1), ("Python installed", 0)])
    responses.append(("/usr/bin/python3\n", 0))
    outputs = iter(responses)
    commands = []

    def execute(command, **kwargs):
        commands.append((command, kwargs))
        return next(outputs)

    runtime._exec = execute
    runtime._bootstrap_executor()

    assert commands[0] == commands[2]
    assert commands[1][0] == ["/bin/sh", "-c", _INSTALL_COREUTILS]
    assert [call[1]["timeout"] for call in commands] == (
        [30, 300, 30, 30, 300, 30] if python_missing else [30, 300, 30, 30]
    )
    assert len(commands) == len(responses)
    assert runtime.transfers == []


@pytest.mark.parametrize(
    "responses,error",
    [
        ([("unsupported package manager", 1)], "could not install GNU coreutils"),
        ([("package installation denied", 1)], "could not install GNU coreutils"),
        (
            [("installed", 0), ("BusyBox dd", 0)],
            "GNU coreutils unavailable after installation",
        ),
        (
            [("installed", 0), ("missing sha256sum", 127)],
            "GNU coreutils unavailable after installation",
        ),
    ],
)
def test_coreutils_bootstrap_failures_remain_explicit(runtime, responses, error):
    runtime.task["agent_network_mode"] = "public"
    outputs = iter([("BusyBox timeout", 1), *responses])
    runtime._exec = lambda *args, **kwargs: next(outputs)
    with pytest.raises(RuntimeError, match=error):
        runtime._bootstrap_executor()
    assert runtime.transfers == []


@pytest.mark.parametrize("network_mode", [None, "no-network"])
def test_coreutils_installation_requires_public_network(runtime, network_mode):
    runtime.task["agent_network_mode"] = network_mode
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        return "BusyBox timeout", 1

    runtime._exec = execute
    with pytest.raises(RuntimeError, match="without public agent network access"):
        runtime._bootstrap_executor()
    assert len(commands) == 1


@pytest.mark.parametrize(
    "tool",
    ["timeout", "cat", "dd", "head", "base64", "sha256sum", "wc", "mkdir", "mv", "rm"],
)
@pytest.mark.parametrize("failure", ["missing", "busybox"])
def test_coreutils_probe_checks_each_resolved_executable(tmp_path, tool, failure):
    for name in (
        "timeout",
        "cat",
        "dd",
        "head",
        "base64",
        "sha256sum",
        "wc",
        "mkdir",
        "mv",
        "rm",
    ):
        path = tmp_path / name
        if name != tool:
            path.symlink_to(shutil.which(name))
        elif failure == "busybox":
            path.write_text("#!/bin/sh\nprintf 'BusyBox utility\\n'\n")
            path.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", "-c", _CHECK_COREUTILS],
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert tool in result.stderr


def test_coreutils_probe_accepts_real_gnu_tools():
    result = subprocess.run(
        ["/bin/bash", "-c", _CHECK_COREUTILS],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "GNU coreutils verified\n"


@pytest.mark.parametrize("tool", ["dd", "timeout"])
def test_coreutils_probe_rejects_unsupported_options(tmp_path, tool):
    for name in (
        "timeout",
        "cat",
        "dd",
        "head",
        "base64",
        "sha256sum",
        "wc",
        "mkdir",
        "mv",
        "rm",
    ):
        path = tmp_path / name
        if name == tool:
            path.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = --version ]; then\n'
                f"    printf '{tool} (GNU coreutils) 9.4\\n'\n"
                "else exit 1; fi\n"
            )
            path.chmod(0o755)
        else:
            path.symlink_to(shutil.which(name))
    result = subprocess.run(
        ["/bin/bash", "-c", _CHECK_COREUTILS],
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert "probe failed" in result.stderr


@pytest.mark.parametrize("manager", ["apk", "apt-get", None])
@pytest.mark.parametrize("exit_code", [0, 1])
def test_coreutils_installation_uses_supported_manager(tmp_path, manager, exit_code):
    if manager:
        executable = tmp_path / manager
        executable.write_text(
            "#!/bin/sh\n" f"printf '{manager} %s\\n' \"$*\"\n" f"exit {exit_code}\n"
        )
        executable.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", "-c", _INSTALL_COREUTILS],
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if manager is None:
        assert result.returncode == 1
        assert "No supported GNU coreutils package manager" in result.stderr
    else:
        assert result.returncode == exit_code
        if manager == "apk":
            assert result.stdout == "apk add --no-cache coreutils\n"
        elif exit_code:
            assert result.stdout == "apt-get update -qq\n"
        else:
            assert result.stdout == (
                "apt-get update -qq\n"
                "apt-get install -y -qq --no-install-recommends coreutils\n"
            )


def test_coreutils_apt_install_failure_is_not_ignored(tmp_path):
    executable = tmp_path / "apt-get"
    executable.write_text(
        '#!/bin/sh\nif [ "$1" = update ]; then exit 0; fi\n'
        "printf 'coreutils installation failed\\n' >&2\nexit 42\n"
    )
    executable.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", "-c", _INSTALL_COREUTILS],
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 42
    assert "coreutils installation failed" in result.stderr


@pytest.mark.parametrize(
    "network_mode,responses,error",
    [
        ("public", [("", 1), ("installed", 0), ("/usr/bin/python3\n", 0)], None),
        ("no-network", [("", 1)], "without public agent network access"),
        (None, [("", 1)], "without public agent network access"),
        (
            "public",
            [("", 1), ("package installation denied", 1)],
            "could not install Python 3",
        ),
        (
            "public",
            [("", 1), ("installed", 0), ("unsupported interpreter", 1)],
            "unavailable after installation",
        ),
    ],
)
def test_execution_provisions_missing_python(runtime, network_mode, responses, error):
    runtime.task["agent_network_mode"] = network_mode
    outputs = iter(responses)
    commands = []
    copies = []

    def execute(command, **kwargs):
        commands.append((command, kwargs))
        if kwargs["operation"] == "shell executor bootstrap":
            return "timeout (GNU coreutils) 9.4\n", 0
        return next(outputs)

    runtime._exec = execute
    runtime.copy_to_container = lambda *args: copies.append(args)
    if error:
        with pytest.raises(RuntimeError, match=error):
            runtime._bootstrap_executor()
        assert copies == []
    else:
        runtime._bootstrap_executor()
        assert copies == []
        assert [call[1]["timeout"] for call in commands] == [30, 30, 300, 30]
        assert "apt-get install" in commands[2][0][-1]
        assert "apk add --no-cache python3" in commands[2][0][-1]
        assert commands[1] == commands[3]
    assert len(commands) == len(responses) + 1


def _sdk_response(output, status):
    response = object.__new__(WSClient)
    response._connected = False
    response._returncode = None
    response._closed_channels = set()
    response._channels = {ERROR_CHANNEL: status}
    response._all = io.StringIO(output)
    response.binary = False
    response.sock = None
    return response


@pytest.mark.parametrize("code", [0, 7])
def test_raw_exec_reads_sdk_status_before_read_all_clears_it(runtime, code):
    status = (
        {"status": "Success"}
        if code == 0
        else {"status": "Failure", "details": {"causes": [{"message": str(code)}]}}
    )
    response = _sdk_response("output\r\n", json.dumps(status))
    runtime._stream = lambda *args, **kwargs: response

    assert runtime._exec(["true"], timeout=1, operation="test") == ("output\r\n", code)
    assert response._channels == {}


@pytest.mark.parametrize(
    "status",
    [
        "",
        '{"status": "Failure"}',
        '{"status": "Failure", "details": {"causes": []}}',
        '{"status": "Failure", "details": {"causes": [{"message": "bad"}]}}',
        "{",
    ],
)
def test_raw_exec_rejects_missing_and_malformed_sdk_status(runtime, status):
    response = _sdk_response("partial", status)
    closed = []
    response.close = lambda: closed.append(True)
    runtime._stream = lambda *args, **kwargs: response

    with pytest.raises(PodExecutionError, match="invalid Kubernetes exit status"):
        runtime._exec(["true"], timeout=1, operation="test")
    assert closed == [True]


def test_raw_exec_distinguishes_stream_timeout_and_closes_it(runtime):
    closed = []
    response = SimpleNamespace(
        run_forever=lambda **kwargs: None,
        is_open=lambda: True,
        read_all=lambda: "partial",
        close=lambda: closed.append(True),
    )
    runtime._stream = lambda *args, **kwargs: response

    with pytest.raises(CommandTimeoutError) as error:
        runtime._exec(["true"], timeout=1, operation="bootstrap")
    assert error.value.output == "partial"
    assert closed == [True]


def test_raw_exec_preserves_pod_not_found_hidden_by_sdk_error(runtime):
    def stream(*args, **kwargs):
        try:
            raise ApiException(status=404, reason="pod not found")
        except ApiException:
            raise AttributeError("'NoneType' object has no attribute 'decode'")

    runtime._stream = stream
    with pytest.raises(PodExecutionError, match="404") as error:
        runtime._exec(["true"], timeout=1, operation="test")
    assert isinstance(error.value.__cause__, AttributeError)
    assert "NoneType" in str(error.value)


def test_raw_exec_does_not_disguise_programming_errors(runtime):
    runtime._stream = lambda *args, **kwargs: (_ for _ in ()).throw(
        AttributeError("unrelated bug")
    )
    with pytest.raises(AttributeError, match="unrelated bug"):
        runtime._exec(["true"], timeout=1, operation="test")


def test_pod_diagnostics_include_container_and_init_failures(runtime):
    runtime._core.read_namespaced_pod = lambda *args, **kwargs: SimpleNamespace(
        status=SimpleNamespace(
            phase="Failed",
            reason="DeadlineExceeded",
            message="pod deadline elapsed",
            init_container_statuses=[
                SimpleNamespace(
                    name="init",
                    state=SimpleNamespace(
                        waiting=SimpleNamespace(
                            reason="ImagePullBackOff", message="cannot pull image"
                        ),
                        terminated=None,
                    ),
                    last_state=None,
                )
            ],
            container_statuses=[
                SimpleNamespace(
                    name="task",
                    state=SimpleNamespace(
                        waiting=None,
                        terminated=SimpleNamespace(
                            reason="OOMKilled", exit_code=137, message=None
                        ),
                    ),
                    last_state=None,
                )
            ],
            conditions=[
                SimpleNamespace(
                    type="Ready",
                    status="False",
                    reason="ContainersNotReady",
                    message="",
                )
            ],
        )
    )

    detail = KubernetesTaskRuntime._pod_status_detail(runtime)
    assert "DeadlineExceeded" in detail
    assert "init state.waiting: reason=ImagePullBackOff" in detail
    assert "task state.terminated: reason=OOMKilled, exit_code=137" in detail


class LocalExecResponse:
    def __init__(self, command):
        self.command = command
        self.input = []
        self.closed = False
        self.drop_status = False
        self.truncate_output = False

    def write_stdin(self, value):
        self.input.append(value)

    def run_forever(self, timeout):
        self.result = subprocess.run(
            self.command,
            input="".join(self.input).encode("ascii"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )

    def is_open(self):
        return False

    @property
    def returncode(self):
        return None if self.drop_status else self.result.returncode

    def read_all(self):
        output = (self.result.stdout + self.result.stderr).decode("utf-8")
        return output[:-2] if self.truncate_output else output

    def close(self):
        self.closed = True


@pytest.mark.parametrize("interpreter", [["/bin/bash", "-c"], ["/bin/bash", "-lc"]])
@pytest.mark.parametrize("fault", ["launch", "read", "truncated"])
def test_runtime_round_trip_through_real_local_shell(
    runtime, tmp_path, monkeypatch, interpreter, fault
):
    runtime.task["command_interpreter"] = interpreter
    runtime._command_output_root = str(tmp_path / "outputs")
    responses = []
    lost = False

    def stream(*args, **kwargs):
        nonlocal lost
        response = LocalExecResponse(kwargs["command"])
        responses.append(response)
        assert kwargs["command"][:2] == ["/bin/bash", "-c"]
        action = _operation(kwargs["command"])
        if action == "execute" and fault == "launch" and not lost:
            lost = True
            response.drop_status = True
        if action == "read" and fault in {"read", "truncated"} and not lost:
            lost = True
            if fault == "read":
                raise PodExecutionError("simulated lost read")
            response.truncate_output = True
        return response

    runtime._stream = stream
    code = (
        "from pathlib import Path; import sys; "
        "p = Path('executions'); p.write_text(p.read_text() + 'x' if p.exists() else 'x'); "
        "sys.stdout.buffer.write(b'line\\r\\n' * 21000)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    output, exit_code = runtime.run(command, timeout=10, workdir=str(tmp_path))

    assert output == "line\r\n" * 21000
    assert exit_code == 0
    assert (tmp_path / "executions").read_text() == "x"
    command_ids = {
        Path(response.command[4]).name
        for response in responses
        if _operation(response.command) == "execute"
    }
    assert len(command_ids) == 1
    assert all(
        not (tmp_path / "outputs" / command_id).exists() for command_id in command_ids
    )
    assert all(response.closed for response in responses if hasattr(response, "result"))


@pytest.mark.parametrize("size", [12, 70000, 300000])
def test_real_exec_handles_small_and_large_transferred_commands(
    runtime, tmp_path, monkeypatch, size
):
    runtime._stream = lambda *args, **kwargs: LocalExecResponse(kwargs["command"])
    command = "printf 'complete\\r\\n'; #" + "a" * size

    assert runtime.run(command, timeout=10, workdir=str(tmp_path)) == (
        "complete\r\n",
        0,
    )


def test_one_shot_wrapper_exits_after_its_command(runtime, tmp_path, monkeypatch):
    runtime._stream = lambda *args, **kwargs: LocalExecResponse(kwargs["command"])

    output, code = runtime.run('printf "%s" "$PPID"', workdir=str(tmp_path))

    assert code == 0
    assert not Path(f"/proc/{int(output)}").exists()


def test_disconnected_exec_recovers_an_already_running_command(
    runtime, tmp_path, monkeypatch
):
    runtime._stream = lambda *args, **kwargs: LocalExecResponse(kwargs["command"])
    actual_exec = runtime._exec
    processes = []

    def execute(command, **kwargs):
        if _operation(command) != "execute":
            return actual_exec(command, **kwargs)
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        processes.append(process)
        metadata = Path(command[4]) / "claimed"
        until = time.monotonic() + 5
        while not metadata.exists():
            if process.poll() is not None:
                pytest.fail(
                    f"Wrapper exited before claiming: {process.communicate()[1]!r}"
                )
            if time.monotonic() >= until:
                pytest.fail("Wrapper did not claim the command")
            time.sleep(0.01)
        raise PodExecutionError("exec connection lost while wrapper is running")

    runtime._exec = execute
    try:
        assert runtime.run(
            "sleep 0.3; printf 'recovered\\r\\n'", timeout=3, workdir=str(tmp_path)
        ) == ("recovered\r\n", 0)
        assert len(processes) == 1
        assert processes[0].wait(timeout=5) == 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)


def test_large_leaf_request_is_copied_and_executed_without_inline_payload(
    runtime, tmp_path, monkeypatch
):
    from frognano.harness.leaf.environment import LeafEnvironment

    monkeypatch.setattr(
        "frognano.harness.leaf.environment._RUNNER_PATH",
        str(tmp_path / "tool_runner.py"),
    )
    monkeypatch.setattr(
        "frognano.harness.leaf.environment._REQUEST_PREFIX",
        str(tmp_path / "leaf-request-"),
    )
    runtime.task["repo_path"] = str(tmp_path)
    runtime._stream = lambda *args, **kwargs: LocalExecResponse(kwargs["command"])
    environment = LeafEnvironment(runtime)
    content = "a" * 300000 + "\N{SNOWMAN}\r\n\n"

    environment.execute("Write", {"file_path": "large.txt", "content": content})

    assert (tmp_path / "large.txt").read_bytes() == content.encode()
    assert list(tmp_path.glob("leaf-request-*.json")) == []
    observation = environment.execute(
        "Bash", {"command": "printf '%s' \"$0\"; printf '\\r\\n'", "timeout": 2}
    )
    assert observation.startswith("Exit code: 0\n")
    assert "/bin/bash" in observation


@pytest.mark.parametrize(
    "command,code",
    [
        ("exit 7", 7),
        ("exit 124", 124),
        ("exit 137", 137),
        ("kill -TERM $$", 143),
        ("if then", 2),
        ("", 0),
    ],
)
def test_real_shell_preserves_nonzero_signal_and_syntax_results(
    runtime, tmp_path, command, code
):
    runtime._stream = lambda *args, **kwargs: LocalExecResponse(kwargs["command"])
    _, actual_code = runtime.run(command, timeout=3, workdir=str(tmp_path))
    assert actual_code == code


def test_real_shell_timeout_keeps_partial_output(runtime, tmp_path):
    runtime._stream = lambda *args, **kwargs: LocalExecResponse(kwargs["command"])
    with pytest.raises(CommandTimeoutError) as error:
        runtime.run(
            "printf 'partial\\r\\n'; sleep 10", timeout=0.1, workdir=str(tmp_path)
        )
    assert error.value.output == "partial\r\n"


def test_shell_execution_does_not_require_a_python_process(
    runtime, tmp_path, monkeypatch
):
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    for tool in (
        "timeout",
        "cat",
        "dd",
        "head",
        "base64",
        "sha256sum",
        "wc",
        "mkdir",
        "mv",
        "rm",
    ):
        executable = shutil.which(tool)
        assert executable is not None
        (bin_path / tool).symlink_to(executable)
    monkeypatch.setenv("PATH", str(bin_path))
    runtime._stream = lambda *args, **kwargs: LocalExecResponse(kwargs["command"])

    assert runtime.run("printf 'no Python wrapper'", workdir=str(tmp_path)) == (
        "no Python wrapper",
        0,
    )


@pytest.mark.parametrize("cleaned_up", [False, True])
def test_delayed_execute_cannot_repeat_a_finished_mutation(
    runtime, tmp_path, monkeypatch, cleaned_up
):
    runtime._stream = lambda *args, **kwargs: LocalExecResponse(kwargs["command"])
    actual_exec = runtime._exec
    launches = []
    monkeypatch.setattr("frognano.runtimes.kubernetes.time.sleep", lambda _: None)

    def execute(command, **kwargs):
        if _operation(command) == "execute":
            launches.append(command)
        if _operation(command) == "cleanup" and not cleaned_up:
            raise PodExecutionError("keep result to exercise a delayed duplicate")
        return actual_exec(command, **kwargs)

    runtime._exec = execute
    assert runtime.run("printf x >> mutations", workdir=str(tmp_path)) == ("", 0)
    assert len(launches) == 1
    duplicate = subprocess.run(launches[0], capture_output=True, timeout=3)
    assert duplicate.returncode != 0
    assert (tmp_path / "mutations").read_text() == "x"


@pytest.mark.parametrize("command_id", ["", ".", "..", "/tmp", "a" * 31, "g" * 32])
def test_control_requests_reject_unsafe_command_paths(runtime, command_id):
    with pytest.raises(ValueError, match="invalid command ID"):
        runtime._command_request("cleanup", command_id)


@pytest.mark.parametrize("interpreter", [["/bin/bash", "-c"], ["/bin/bash", "-lc"]])
def test_sourced_commands_preserve_shell_zero_arguments_and_login_mode(
    runtime, tmp_path, interpreter
):
    runtime.task["command_interpreter"] = interpreter
    runtime._stream = lambda *args, **kwargs: LocalExecResponse(kwargs["command"])
    command = 'printf "%s %s\\n" "$0" "$#"; shopt -q login_shell'
    expected = subprocess.run(
        [*interpreter, command],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert runtime.run(command, workdir=str(tmp_path)) == (
        expected.stdout,
        expected.returncode,
    )


def test_source_script_cannot_replace_capture_shell_exit_trap(runtime, tmp_path):
    runtime._stream = lambda *args, **kwargs: LocalExecResponse(kwargs["command"])
    assert runtime.run(
        "trap 'printf goodbye' EXIT; printf hello; exit 23", workdir=str(tmp_path)
    ) == ("hellogoodbye", 23)


def test_output_capture_waits_for_inherited_writers(runtime, tmp_path):
    runtime._stream = lambda *args, **kwargs: LocalExecResponse(kwargs["command"])
    assert runtime.run(
        "(sleep 0.1; printf after) & printf before", workdir=str(tmp_path)
    ) == ("beforeafter", 0)


def test_killed_capture_shell_is_unknown_not_a_success_or_rerun(runtime, tmp_path):
    runtime._stream = lambda *args, **kwargs: LocalExecResponse(kwargs["command"])
    with pytest.raises(PodExecutionError, match="no completion record"):
        runtime.run(
            'printf x >> mutations; kill -KILL "$PPID"',
            timeout=3,
            workdir=str(tmp_path),
        )
    assert (tmp_path / "mutations").read_text() == "x"


def test_unavailable_workdir_is_a_nonzero_shell_result(runtime, tmp_path):
    runtime._stream = lambda *args, **kwargs: LocalExecResponse(kwargs["command"])
    output, code = runtime.run(
        "printf should-not-run", workdir=str(tmp_path / "does-not-exist")
    )
    assert code != 0
    assert "should-not-run" not in output
