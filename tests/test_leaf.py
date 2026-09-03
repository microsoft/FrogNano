import ast
import json
import math
import shlex
import subprocess
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

from frognano.leaf.agent import (
    LeafAgent,
    LeafConfig,
    OpenAIChatClient,
    _count_message_tokens,
)
from frognano.leaf.environment import LeafEnvironment
from frognano.leaf.tool_runner import main, run_tool
from frognano.leaf.tools import OPENAI_TOOLS


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
        return next(self.responses)


def _config(**overrides) -> LeafConfig:
    values = {
        "model": "model",
        "base_url": "http://model/v1",
        "api_key": "test",
        "max_steps": 4,
    }
    values.update(overrides)
    return LeafConfig(**values)


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


def test_leaf_records_query_failures() -> None:
    class FailingClient:
        def complete(self, messages, *, tools):
            raise TimeoutError("model unavailable")

    trajectory = LeafAgent(_config(), client=FailingClient()).run(
        FakeEnvironment(),
        instance_id="task-1",
        seed=1,
    )

    assert trajectory["exit_reason"] == "llm_query_error"
    assert "model unavailable" in trajectory["error"]


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
        "frognano.leaf.agent._load_huggingface_tokenizer",
        lambda tokenizer_id: loaded.append(tokenizer_id) or tokenizer,
    )
    monkeypatch.setattr(
        "frognano.leaf.agent._load_tiktoken_encoding",
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
        "frognano.leaf.agent._load_huggingface_tokenizer",
        lambda tokenizer_id: loaded.append(tokenizer_id) or None,
    )
    monkeypatch.setattr(
        "frognano.leaf.agent._load_tiktoken_encoding",
        lambda tokenizer_id: encoding,
    )

    assert _count_message_tokens(messages, "Qwen/Qwen3.5") == 2
    assert loaded == ["o200k_base"]


def test_leaf_context_falls_back_to_local_estimate(monkeypatch) -> None:
    messages = [{"role": "user", "content": "hello"}]
    serialized = json.dumps(messages, separators=(",", ":"), ensure_ascii=False)
    monkeypatch.delenv("FROGNANO_TOKENIZER", raising=False)
    monkeypatch.setattr(
        "frognano.leaf.agent._load_huggingface_tokenizer",
        lambda tokenizer_id: None,
    )
    monkeypatch.setattr(
        "frognano.leaf.agent._load_tiktoken_encoding",
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
    path = Path("frognano/leaf/tool_runner.py")
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
        self.copies = []
        self.commands = []
        self.recreations = 0
        self.tmp_path = tmp_path

    def copy_to_container(self, source, destination) -> None:
        self.copies.append((source, destination))

    def get_task_instruction(self) -> str:
        return "Task"

    def run(self, command, *, timeout, workdir):
        self.commands.append((command, timeout, workdir))
        return "observation", 0

    def get_patch(self) -> str:
        return "patch"

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
    assert runtime.copies[0][1] == "/tmp/frognano_leaf_tool_runner.py"
    assert runtime.commands[0][1] == 62
    assert "/usr/bin/python3" in runtime.commands[0][0]


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
            raise RuntimeError("connection refused")
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
    assert len(runtime.copies) == 2
    assert len(runtime.commands) == 3


def test_leaf_environment_does_not_recreate_for_non_pod_errors(tmp_path) -> None:
    runtime = FakeRuntime(tmp_path)
    environment = LeafEnvironment(runtime)
    runtime.run = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("invalid command")
    )

    try:
        environment.execute("Read", {"file_path": "missing.py"})
    except RuntimeError as exc:
        assert str(exc) == "invalid command"
    else:
        raise AssertionError("non-pod error should propagate")

    assert runtime.recreations == 0


def test_openai_client_builds_tool_call_request(monkeypatch) -> None:
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
    config = _config(
        max_tokens_per_turn=100,
        extra_body={"top_p": 0.9},
        parallel_tool_calls=False,
    )

    client = OpenAIChatClient(config)
    response = client.complete([{"role": "user", "content": "x"}], tools=[])

    assert response == "response"
    assert requests[0]["max_completion_tokens"] == 100
    assert requests[0]["extra_body"] == {"top_p": 0.9}
    assert requests[0]["parallel_tool_calls"] is False
