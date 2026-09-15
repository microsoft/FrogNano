import ast
import json
import logging
import math
import shlex
import subprocess
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from frognano.harness.leaf.agent import (
    LeafAgent,
    LeafConfig,
    OpenAIChatClient,
    _count_message_tokens,
)
from frognano.harness.leaf.environment import LeafEnvironment
from frognano.harness.leaf.tool_runner import main, run_tool
from frognano.harness.leaf.tools import OPENAI_TOOLS
from frognano.runtimes.errors import CommandTimeoutError, PodExecutionError


class FakeEnvironment:
    def __init__(self) -> None:
        self.calls = []

    def instruction(self) -> str:
        return "Fix the failing test."

    def execute(self, name, arguments) -> str:
        self.calls.append((name, arguments))
        return "file.py\n1. value = 1"

    def patch(self) -> str:
        return "diff --git a/file.py b/file.py"


def _response(
    *,
    content=None,
    tool_calls=(),
    tokens=10,
    finish_reason=None,
    prompt_tokens=None,
    completion_tokens=None,
):
    message = SimpleNamespace(content=content, tool_calls=list(tool_calls))
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        total_tokens=tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _tool_call(name: str, arguments: dict):
    function = SimpleNamespace(name=name, arguments=json.dumps(arguments))
    return SimpleNamespace(id="call-1", function=function)


class FakeClient:
    def __init__(self, responses) -> None:
        self.responses = iter(responses)

    def complete(self, messages, *, tools):
        assert tools
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _config(**overrides) -> LeafConfig:
    values = {
        "model": "model",
        "base_url": "http://model/v1",
        "api_key": "test",
        "max_steps": 4,
    }
    values.update(overrides)
    return LeafConfig(**values)


def test_leaf_does_not_swallow_patch_capture_failure() -> None:
    class Environment(FakeEnvironment):
        def patch(self) -> str:
            raise RuntimeError("patch capture failed")

    agent = LeafAgent(_config(), client=FakeClient([_response(content="Done.")]))
    with pytest.raises(RuntimeError, match="patch capture failed"):
        agent.run(Environment(), instance_id="task", seed=0)


def test_leaf_executes_tools_and_returns_patch() -> None:
    environment = FakeEnvironment()
    client = FakeClient(
        [
            _response(tool_calls=[_tool_call("Read", {"file_path": "file.py"})]),
            _response(content="Done."),
        ]
    )

    trajectory = LeafAgent(_config(), client=client).run(
        environment,
        instance_id="task-1",
        seed=0,
    )

    assert trajectory["exit_reason"] == "agent"
    assert trajectory["final_message"] == "Done."
    assert trajectory["output_patch"].startswith("diff --git")
    assert environment.calls == [("Read", {"file_path": "file.py"})]
    assert trajectory["messages"][-1]["content"] == "Done."
    assert trajectory["n_steps"] == 2
    assert len(trajectory["steps"]) == 2
    assert trajectory["steps"][-1]["observations"] == []


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("model unavailable"),
        RuntimeError("No available workers"),
        ValueError("invalid context length parameter"),
        RuntimeError("rate limit exceeded"),
    ],
)
def test_leaf_records_query_failures(error) -> None:
    trajectory = LeafAgent(_config(), client=FakeClient([error])).run(
        FakeEnvironment(),
        instance_id="task-1",
        seed=1,
    )

    assert trajectory["exit_reason"] == "llm_query_error"
    assert str(error) in trajectory["error"]


@pytest.mark.parametrize(
    "message",
    [
        "CONTEXT_LENGTH_EXCEEDED",
        "Please reduce the length of the messages",
        "This model's maximum context length is 131072 tokens",
        "Input context length exceeds the model limit",
    ],
)
def test_leaf_preserves_work_on_context_rejection(monkeypatch, caplog, message) -> None:
    monkeypatch.setattr(
        "frognano.harness.leaf.agent._count_message_tokens", lambda *args: 1
    )
    environment = FakeEnvironment()
    checkpoints = []
    client = FakeClient(
        [
            _response(
                tool_calls=[
                    _tool_call("Write", {"file_path": "file.py", "content": "fix"})
                ]
            ),
            ValueError(message),
        ]
    )

    trajectory = LeafAgent(_config(), client=client).run(
        environment,
        instance_id="task",
        seed=0,
        checkpoint_callback=checkpoints.append,
    )

    assert trajectory["exit_reason"] == "max_context_len"
    assert trajectory["error"] is None
    assert trajectory["output_patch"] == environment.patch()
    assert trajectory["n_steps"] == 2
    assert environment.calls == [("Write", {"file_path": "file.py", "content": "fix"})]
    assert trajectory["messages"][-1]["role"] == "tool"
    assert checkpoints[-1]["exit_reason"] == "max_context_len"
    assert checkpoints[-1]["error"] is None
    assert "model context limit" in caplog.text


def test_leaf_recognizes_openai_context_rejection() -> None:
    import httpx
    from openai import BadRequestError

    error = BadRequestError(
        "context_length_exceeded",
        response=httpx.Response(
            400,
            request=httpx.Request("POST", "http://model.test/v1/chat/completions"),
        ),
        body={"code": "context_length_exceeded"},
    )

    trajectory = LeafAgent(_config(), client=FakeClient([error])).run(
        FakeEnvironment(), instance_id="task", seed=0
    )

    assert trajectory["exit_reason"] == "max_context_len"
    assert trajectory["error"] is None


def test_leaf_treats_empty_tool_free_response_as_agent_finish() -> None:
    client = FakeClient([_response(content="")])

    trajectory = LeafAgent(_config(), client=client).run(
        FakeEnvironment(),
        instance_id="task-1",
        seed=0,
    )

    assert trajectory["exit_reason"] == "agent"
    assert trajectory["final_message"] == ""
    assert trajectory["messages"][-1] == {
        "role": "assistant",
        "content": "",
    }


def test_leaf_whitelists_assistant_message_fields() -> None:
    message = SimpleNamespace(
        content="Done.",
        tool_calls=[],
        reasoning_content="private reasoning",
        matched_stop="stop",
    )

    trajectory = LeafAgent(
        _config(),
        client=FakeClient(
            [
                SimpleNamespace(
                    choices=[SimpleNamespace(message=message)],
                    usage=SimpleNamespace(total_tokens=1),
                )
            ]
        ),
    ).run(FakeEnvironment(), instance_id="task", seed=0)

    assert trajectory["messages"][-1] == {
        "role": "assistant",
        "content": "Done.",
        "reasoning_content": "private reasoning",
    }


def test_leaf_classifies_truncated_turn_without_executing_tools() -> None:
    client = FakeClient(
        [
            _response(
                content="partial",
                tool_calls=[_tool_call("Read", {"file_path": "file.py"})],
                finish_reason="length",
                prompt_tokens=10,
                completion_tokens=20,
            )
        ]
    )
    environment = FakeEnvironment()

    trajectory = LeafAgent(
        _config(max_tokens_per_turn=20),
        client=client,
    ).run(environment, instance_id="task", seed=0)

    assert trajectory["exit_reason"] == "max_tokens_per_turn"
    assert environment.calls == []


def test_leaf_classifies_context_clamped_truncated_turn() -> None:
    trajectory = LeafAgent(
        _config(max_tokens_per_turn=20, max_context_tokens=30),
        client=FakeClient(
            [
                _response(
                    content="partial",
                    finish_reason="length",
                    prompt_tokens=10,
                    completion_tokens=20,
                )
            ]
        ),
    ).run(FakeEnvironment(), instance_id="task", seed=0)

    assert trajectory["exit_reason"] == "max_context_len"


def test_leaf_turns_malformed_tool_arguments_into_tool_observation() -> None:
    malformed = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="Read", arguments="{"),
    )
    environment = FakeEnvironment()
    trajectory = LeafAgent(
        _config(),
        client=FakeClient(
            [
                _response(tool_calls=[malformed]),
                _response(content="Done."),
            ]
        ),
    ).run(environment, instance_id="task", seed=0)

    assert trajectory["exit_reason"] == "agent"
    assert environment.calls[0] == ("Read", {})


def test_leaf_context_accounting_includes_tool_observations() -> None:
    environment = FakeEnvironment()
    trajectory = LeafAgent(
        _config(max_context_tokens=11),
        client=FakeClient(
            [
                _response(
                    tool_calls=[_tool_call("Read", {"file_path": "file.py"})],
                    tokens=10,
                )
            ]
        ),
    ).run(environment, instance_id="task", seed=0)

    assert trajectory["exit_reason"] == "max_context_len"
    assert trajectory["context_tokens"] > 10


def test_leaf_context_prefers_huggingface_tokenizer(monkeypatch) -> None:
    messages = [
        {"role": "user", "content": "The quick brown fox jumps over the lazy dog."}
    ]
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2, 3])
    loaded = []
    monkeypatch.setenv("FROGNANO_TOKENIZER", "Qwen/Qwen3.5-32B")
    monkeypatch.setattr(
        "frognano.harness.leaf.agent._load_huggingface_tokenizer",
        lambda tokenizer_id: loaded.append(tokenizer_id) or tokenizer,
    )
    monkeypatch.setattr(
        "frognano.harness.leaf.agent._load_tiktoken_encoding",
        lambda tokenizer_id: (_ for _ in ()).throw(AssertionError),
    )

    assert _count_message_tokens(messages, "Qwen/Qwen3.5") == 3
    assert loaded == ["Qwen/Qwen3.5-32B"]


def test_leaf_context_falls_through_to_tiktoken(monkeypatch) -> None:
    messages = [{"role": "user", "content": "hello"}]
    encoding = SimpleNamespace(encode=lambda text: [1, 2])
    loaded = []
    monkeypatch.setenv("FROGNANO_TOKENIZER", "o200k_base")
    monkeypatch.setattr(
        "frognano.harness.leaf.agent._load_huggingface_tokenizer",
        lambda tokenizer_id: loaded.append(tokenizer_id) or None,
    )
    monkeypatch.setattr(
        "frognano.harness.leaf.agent._load_tiktoken_encoding",
        lambda tokenizer_id: encoding,
    )

    assert _count_message_tokens(messages, "Qwen/Qwen3.5") == 2
    assert loaded == ["o200k_base"]


def test_leaf_context_falls_back_to_local_estimate(monkeypatch) -> None:
    messages = [{"role": "user", "content": "hello"}]
    serialized = json.dumps(messages, separators=(",", ":"), ensure_ascii=False)
    monkeypatch.delenv("FROGNANO_TOKENIZER", raising=False)
    monkeypatch.setattr(
        "frognano.harness.leaf.agent._load_huggingface_tokenizer",
        lambda tokenizer_id: None,
    )
    monkeypatch.setattr(
        "frognano.harness.leaf.agent._load_tiktoken_encoding",
        lambda tokenizer_id: None,
    )

    count = _count_message_tokens(messages, "unknown/model")

    assert count == math.ceil(len(serialized.encode("utf-8")) / 4)


def test_leaf_emits_checkpoints_and_honors_cancellation() -> None:
    stop_event = threading.Event()
    stop_event.set()
    checkpoints = []

    trajectory = LeafAgent(
        _config(),
        client=FakeClient([]),
    ).run(
        FakeEnvironment(),
        instance_id="task",
        seed=0,
        checkpoint_callback=checkpoints.append,
        stop_event=stop_event,
    )

    assert trajectory["exit_reason"] == "cancelled"
    assert trajectory["n_steps"] == 0
    assert checkpoints[0]["partial"] is True
    assert checkpoints[-1]["exit_reason"] == "cancelled"
    assert checkpoints[-1]["partial"] is True


def test_leaf_returns_cancelled_trajectory_when_patch_capture_fails() -> None:
    class FailingPatchEnvironment(FakeEnvironment):
        def patch(self) -> str:
            raise RuntimeError("pod is terminating")

    stop_event = threading.Event()
    stop_event.set()

    trajectory = LeafAgent(
        _config(),
        client=FakeClient([]),
    ).run(
        FailingPatchEnvironment(),
        instance_id="task",
        seed=0,
        stop_event=stop_event,
    )

    assert trajectory["exit_reason"] == "cancelled"
    assert trajectory["output_patch"] == ""


def test_leaf_stops_at_context_limit() -> None:
    client = FakeClient(
        [
            _response(
                tool_calls=[_tool_call("Glob", {"pattern": "*.py"})],
                tokens=100,
            )
        ]
    )

    trajectory = LeafAgent(
        _config(max_context_tokens=50),
        client=client,
    ).run(FakeEnvironment(), instance_id="task", seed=0)

    assert trajectory["exit_reason"] == "max_context_len"


def test_tool_runner_reads_edits_writes_and_globs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert "Wrote example.py" in run_tool(
        "Write", {"file_path": "example.py", "content": "value = 1\n"}
    )
    assert "1. value = 1" in run_tool("Read", {"file_path": "example.py"})
    assert "Edited example.py (1 replacement)." == run_tool(
        "Edit",
        {
            "file_path": "example.py",
            "old_string": "value = 1",
            "new_string": "value = 2",
        },
    )
    assert run_tool("Glob", {"pattern": "*.py"}) == "example.py"
    output = run_tool(
        "Bash",
        {
            "command": f"{shlex.quote(sys.executable)} -c 'print(42)'",
            "timeout": 5,
        },
    )
    assert "Exit code: 0" in output
    assert "42" in output


def test_tool_schemas_match_full_leaf_surface() -> None:
    tools = {tool["function"]["name"]: tool["function"] for tool in OPENAI_TOOLS}

    assert tools["Write"]["description"] == (
        "Write a complete file, creating parent directories when needed."
    )
    assert tools["Glob"]["parameters"]["properties"]["pattern"]["description"] == (
        "Glob pattern, e.g. **/*.py."
    )
    assert tools["Bash"]["parameters"]["properties"]["description"] == {
        "type": "string",
        "description": "Short description of the command.",
    }


def test_tool_runner_reports_errors_and_timeouts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert run_tool("Missing", {}) == "Unknown tool: Missing"
    assert "file_path is required" in run_tool("Read", {})

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd="sleep",
            timeout=1,
            output=b"partial",
            stderr=b"late",
        )

    monkeypatch.setattr(subprocess, "run", timeout)
    result = run_tool("Bash", {"command": "sleep 10", "timeout": 1})
    assert "Timed out after 1.0s" in result
    assert "partial" in result


def test_tool_runner_main_reads_standard_input(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["tool_runner.py"])
    monkeypatch.setattr(
        sys,
        "stdin",
        SimpleNamespace(
            read=lambda: json.dumps(
                {
                    "workdir": str(tmp_path),
                    "tool": "Write",
                    "args": {"file_path": "created.txt", "content": "ok"},
                }
            ),
        ),
    )

    assert main() == 0
    assert capsys.readouterr().out == "Wrote created.txt (2 bytes)."


def test_tool_runner_uses_python_36_compatible_annotations() -> None:
    path = Path("frognano/harness/leaf/tool_runner.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 6))
    unsupported = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"dict", "list", "set", "tuple", "type"}
    ]

    assert unsupported == []


class FakeRuntime:
    def __init__(self, tmp_path) -> None:
        self.task = {"repo_path": "/testbed"}
        self.logger = logging.getLogger(__name__)
        self.copies = []
        self.commands = []
        self.request_files = {}
        self.recreations = 0
        self.tmp_path = tmp_path

    def copy_to_container(self, source, destination) -> None:
        self.copies.append((source, destination))
        if destination.endswith(".json"):
            self.request_files[destination] = Path(source).read_bytes()

    def get_task_instruction(self) -> str:
        return "Task"

    def run(self, command, *, timeout, workdir):
        self.commands.append((command, timeout, workdir))
        return "observation", 0

    def get_patch(self) -> str:
        return "patch"

    def compute_reward(self):
        return 0.0, "valid unresolved"

    def recreate(self) -> None:
        self.recreations += 1


def test_leaf_environment_transports_tool_calls(tmp_path) -> None:
    runtime = FakeRuntime(tmp_path)
    environment = LeafEnvironment(runtime)

    assert environment.instruction() == "Task"
    assert environment.execute("Bash", {"command": "true", "timeout": 2}) == (
        "observation"
    )
    assert environment.patch() == "patch"
    assert environment.compute_reward() == (0.0, "valid unresolved")
    assert runtime.copies[0][1] == "/tmp/frognano_leaf_tool_runner.py"
    assert len(runtime.commands) == 1
    assert runtime.commands[0][1] == 62
    assert "/usr/bin/python3" in runtime.commands[0][0]
    [request_path] = runtime.request_files
    assert json.loads(runtime.request_files[request_path]) == {
        "tool": "Bash",
        "args": {"command": "true", "timeout": 2},
        "workdir": "/testbed",
    }
    assert request_path in runtime.commands[0][0]
    assert "base64" not in runtime.commands[0][0]


@pytest.mark.parametrize("tool", ["Read", "Write", "Edit", "Glob", "Bash"])
@pytest.mark.parametrize("size", [0, 32, 65536, 300000])
def test_all_leaf_tools_transfer_request_files_at_every_size(tmp_path, tool, size):
    runtime = FakeRuntime(tmp_path)
    environment = LeafEnvironment(runtime)
    value = "request-marker-'\"$HOME\r\n" + "x" * size + "\n\n"
    arguments = {
        "Read": {"file_path": value},
        "Write": {"file_path": "file", "content": value},
        "Edit": {"file_path": "file", "old_string": "old", "new_string": value},
        "Glob": {"pattern": value},
        "Bash": {"command": value},
    }[tool]

    assert environment.execute(tool, arguments) == "observation"
    [path] = runtime.request_files
    assert json.loads(runtime.request_files[path]) == {
        "tool": tool,
        "args": arguments,
        "workdir": "/testbed",
    }
    assert value not in runtime.commands[0][0]
    assert path in runtime.commands[0][0]


def test_leaf_request_copy_failure_does_not_execute(tmp_path):
    runtime = FakeRuntime(tmp_path)
    environment = LeafEnvironment(runtime)

    def copy(*args):
        raise RuntimeError("copy failed")

    runtime.copy_to_container = copy
    with pytest.raises(RuntimeError, match="copy failed"):
        environment.execute("Read", {"file_path": "file"})
    assert runtime.commands == []


def test_leaf_environment_replays_mutations_after_pod_recreation(tmp_path) -> None:
    runtime = FakeRuntime(tmp_path)
    environment = LeafEnvironment(runtime)
    environment.execute("Write", {"file_path": "first.py", "content": "one"})
    original_run = runtime.run
    failed = False

    def fail_once(command, *, timeout, workdir):
        nonlocal failed
        if not failed:
            failed = True
            raise PodExecutionError("connection refused")
        return original_run(command, timeout=timeout, workdir=workdir)

    runtime.run = fail_once
    result = environment.execute(
        "Edit",
        {
            "file_path": "first.py",
            "old_string": "one",
            "new_string": "two",
        },
    )

    assert result == "observation"
    assert runtime.recreations == 1
    assert (
        sum(
            destination == "/tmp/frognano_leaf_tool_runner.py"
            for _, destination in runtime.copies
        )
        == 2
    )
    assert len(runtime.request_files) == 4
    assert len(runtime.commands) == 3


@pytest.mark.parametrize(
    "error", [RuntimeError("invalid command"), CommandTimeoutError("deadline")]
)
def test_leaf_environment_does_not_recreate_for_non_pod_errors(tmp_path, error) -> None:
    runtime = FakeRuntime(tmp_path)
    environment = LeafEnvironment(runtime)
    runtime.run = lambda *args, **kwargs: (_ for _ in ()).throw(error)

    with pytest.raises(type(error), match=str(error)):
        environment.execute("Read", {"file_path": "missing.py"})
    assert runtime.recreations == 0


@pytest.mark.parametrize("code", [1, 2, 127, 137])
def test_leaf_runner_nonzero_exit_is_an_observation(tmp_path, code):
    runtime = FakeRuntime(tmp_path)
    environment = LeafEnvironment(runtime)
    runtime.run = lambda *args, **kwargs: ("runner output", code)

    assert environment.execute("Bash", {"command": "false"}) == (
        f"Tool runner exited with code {code}.\nrunner output"
    )
    assert runtime.recreations == 0


@pytest.mark.parametrize("operation", ["patch", "compute_reward"])
def test_leaf_recovers_patch_and_grading_by_replaying_mutations(tmp_path, operation):
    runtime = FakeRuntime(tmp_path)
    environment = LeafEnvironment(runtime)
    environment.execute("Write", {"file_path": "first.py", "content": "one"})
    environment.execute("Read", {"file_path": "first.py"})
    original = runtime.get_patch if operation == "patch" else runtime.compute_reward
    calls = []

    def once():
        calls.append(True)
        if len(calls) == 1:
            raise PodExecutionError("missing pod")
        assert len(runtime.commands) == 3
        return original()

    setattr(runtime, "get_patch" if operation == "patch" else operation, once)
    assert getattr(environment, operation)() == original()
    assert len(calls) == 2
    assert runtime.recreations == 1
    assert (
        sum(
            destination == "/tmp/frognano_leaf_tool_runner.py"
            for _, destination in runtime.copies
        )
        == 2
    )


def test_leaf_recovery_is_bounded_across_operations(tmp_path):
    runtime = FakeRuntime(tmp_path)
    environment = LeafEnvironment(runtime)
    runtime.get_patch = lambda: (_ for _ in ()).throw(PodExecutionError("missing pod"))

    with pytest.raises(PodExecutionError, match="missing pod"):
        environment.patch()
    assert runtime.recreations == 2
    with pytest.raises(PodExecutionError, match="missing pod"):
        environment.patch()
    assert runtime.recreations == 2


def test_leaf_recreation_failure_preserves_both_errors(tmp_path):
    runtime = FakeRuntime(tmp_path)
    environment = LeafEnvironment(runtime)
    runtime.get_patch = lambda: (_ for _ in ()).throw(PodExecutionError("original"))
    runtime.recreate = lambda: (_ for _ in ()).throw(TimeoutError("pod not deleted"))

    with pytest.raises(PodExecutionError, match="original") as error:
        environment.patch()
    assert "pod not deleted" in str(error.value)
    assert isinstance(error.value.__cause__, TimeoutError)


@pytest.mark.parametrize(
    "replay",
    [
        PodExecutionError("replay lost pod"),
        ("runner was killed", 137),
        ("PermissionError: write denied", 0),
    ],
)
def test_leaf_replay_failure_aborts_before_grading(tmp_path, replay):
    runtime = FakeRuntime(tmp_path)
    environment = LeafEnvironment(runtime)
    environment.execute("Write", {"file_path": "first.py", "content": "one"})
    calls = []

    def grade():
        calls.append(True)
        raise PodExecutionError("original")

    def run(*args, **kwargs):
        if isinstance(replay, Exception):
            raise replay
        return replay

    runtime.compute_reward = grade
    runtime.run = run
    with pytest.raises(PodExecutionError, match="replay of Write failed"):
        environment.compute_reward()
    assert len(calls) == 1
    assert runtime.recreations == 1


@pytest.mark.parametrize(
    "original_status,replayed_status",
    [
        ("Exit code: 0", "Exit code: 0"),
        ("Exit code: 2", "Exit code: 2"),
        ("Exit code: 0", "Exit code: 127"),
        ("Timed out after 2.0s", "Timed out after 2.0s"),
    ],
)
def test_bash_replay_compares_outcome_not_nondeterministic_output(
    tmp_path, original_status, replayed_status
):
    runtime = FakeRuntime(tmp_path)
    environment = LeafEnvironment(runtime)
    runtime.run = lambda *args, **kwargs: (f"{original_status}\nSTDOUT:\nold", 0)
    environment.execute("Bash", {"command": "action"})
    runtime.run = lambda *args, **kwargs: (f"{replayed_status}\nSTDOUT:\nnew", 0)
    attempts = iter([PodExecutionError("gone"), "patch"])

    def patch():
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    runtime.get_patch = patch
    if original_status == replayed_status:
        assert environment.patch() == "patch"
    else:
        with pytest.raises(PodExecutionError, match="different outcome"):
            environment.patch()


@pytest.mark.parametrize("command,code", [("(", 2), ("frognano_no_such_command", 127)])
def test_invalid_bash_command_remains_normal_leaf_observation(command, code):
    output = run_tool("Bash", {"command": command, "timeout": 2})
    assert output.startswith(f"Exit code: {code}\nSTDOUT:\n")
    assert "\nSTDERR:\n" in output


@pytest.mark.parametrize("token_limit", ["default", None, 100])
def test_openai_client_builds_tool_call_request(monkeypatch, token_limit) -> None:
    requests = []

    class Completions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return "response"

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=Completions())

    module = types.ModuleType("openai")
    module.OpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", module)
    overrides = {} if token_limit == "default" else {"max_tokens_per_turn": token_limit}
    config = _config(
        **overrides,
        extra_body={"top_p": 0.9},
        parallel_tool_calls=False,
    )

    client = OpenAIChatClient(config)
    response = client.complete([{"role": "user", "content": "x"}], tools=[])

    assert response == "response"
    expected = 8192 if token_limit == "default" else token_limit
    if expected is None:
        assert "max_completion_tokens" not in requests[0]
    else:
        assert requests[0]["max_completion_tokens"] == expected
    assert requests[0]["extra_body"] == {"top_p": 0.9}
    assert requests[0]["parallel_tool_calls"] is False
