"""Leaf model and tool orchestration."""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Protocol

import tiktoken

from frognano.config import DEFAULT_MAX_TOKENS_PER_TURN
from frognano.harness.leaf.environment import LeafEnvironment
from frognano.harness.leaf.tools import OPENAI_TOOLS, SYSTEM_PROMPT

_LENGTH_FINISH_REASONS = frozenset({"length", "max_tokens", "max_new_tokens"})
logger = logging.getLogger(__name__)


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> Any: ...


@dataclass(frozen=True)
class LeafConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.6
    max_tokens_per_turn: int | None = DEFAULT_MAX_TOKENS_PER_TURN
    timeout_sec: int = 7200
    max_retries: int = 5
    parallel_tool_calls: bool = True
    max_steps: int = 100
    max_context_tokens: int = 65536
    max_total_time_sec: int | None = None
    extra_body: dict[str, Any] | None = None


class OpenAIChatClient:
    def __init__(self, config: LeafConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is required; install FrogNano") from exc
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_sec,
            max_retries=config.max_retries,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.config.temperature,
            "parallel_tool_calls": self.config.parallel_tool_calls,
        }
        if self.config.max_tokens_per_turn is not None:
            kwargs["max_completion_tokens"] = self.config.max_tokens_per_turn
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body
        return self.client.chat.completions.create(**kwargs)


class LeafAgent:
    def __init__(
        self,
        config: LeafConfig,
        *,
        client: ChatClient | None = None,
    ) -> None:
        self.config = config
        self.client = client or OpenAIChatClient(config)

    def run(
        self,
        environment: LeafEnvironment,
        *,
        instance_id: str,
        seed: int,
        checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
        stop_event: Any | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": environment.instruction()},
        ]
        steps: list[dict[str, Any]] = []
        started = time.monotonic()
        exit_reason = "unknown"
        error: str | None = None
        context_tokens = 0
        final_message = ""
        step_count = 0

        def trajectory(
            *,
            reason: str,
            partial: bool,
            output_patch: str = "",
        ) -> dict[str, Any]:
            return {
                "trajectory_format": "leaf-function-calling",
                "trajectory_id": trajectory_id,
                "instance_id": instance_id,
                "seed": seed,
                "messages": messages,
                "tools": OPENAI_TOOLS,
                "steps": steps,
                "n_steps": step_count,
                "exit_reason": reason,
                "error": error,
                "final_message": final_message,
                "context_tokens": context_tokens,
                "elapsed_sec": time.monotonic() - started,
                "output_patch": output_patch,
                "partial": partial,
            }

        def emit_checkpoint(reason: str = "running") -> None:
            if checkpoint_callback is not None:
                try:
                    checkpoint_callback(trajectory(reason=reason, partial=True))
                except Exception:
                    logger.warning("Leaf checkpoint failed", exc_info=True)

        trajectory_id = uuid.uuid4().hex
        emit_checkpoint()
        for step in range(self.config.max_steps):
            if stop_event is not None and stop_event.is_set():
                exit_reason = "cancelled"
                break
            if (
                self.config.max_total_time_sec is not None
                and time.monotonic() - started >= self.config.max_total_time_sec
            ):
                exit_reason = "max_time"
                break
            step_count += 1
            try:
                response = self.client.complete(messages, tools=OPENAI_TOOLS)
            except Exception as exc:
                if _is_context_overflow(exc):
                    exit_reason = "max_context_len"
                    logger.warning("Leaf reached the model context limit: %s", exc)
                else:
                    exit_reason = "llm_query_error"
                    error = f"{type(exc).__name__}: {exc}"
                break
            choice = response.choices[0]
            message = choice.message
            assistant = _message_dict(message)
            messages.append(assistant)
            usage = getattr(response, "usage", None)
            context_tokens = _usage_total(usage) or _count_message_tokens(
                messages,
                self.config.model,
            )
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            truncated = _is_length_finish(choice)
            if not tool_calls or truncated:
                final_message = str(getattr(message, "content", None) or "")
                steps.append(
                    {
                        "step": step,
                        "assistant": assistant,
                        "observations": [],
                    }
                )
                exit_reason = (
                    self._length_exit_reason(response) if truncated else "agent"
                )
                emit_checkpoint(exit_reason)
                break
            observations: list[dict[str, str]] = []
            tool_messages: list[dict[str, Any]] = []
            try:
                for call in tool_calls:
                    arguments = _parse_arguments(call.function.arguments)
                    observation = environment.execute(
                        call.function.name,
                        arguments,
                    )
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": observation,
                    }
                    messages.append(tool_message)
                    tool_messages.append(tool_message)
                    observations.append(
                        {
                            "tool": call.function.name,
                            "observation": observation,
                        }
                    )
            except Exception as exc:
                exit_reason = "tool_error"
                error = f"{type(exc).__name__}: {exc}"
                break
            steps.append(
                {
                    "step": step,
                    "assistant": assistant,
                    "observations": observations,
                }
            )
            context_tokens += _count_message_tokens(
                tool_messages,
                self.config.model,
            )
            emit_checkpoint()
            if (
                self.config.max_context_tokens > 0
                and context_tokens >= self.config.max_context_tokens
            ):
                exit_reason = "max_context_len"
                break
        else:
            exit_reason = "max_turns"

        emit_checkpoint(exit_reason)
        output_patch = "" if exit_reason == "cancelled" else environment.patch()
        result = trajectory(
            reason=exit_reason,
            partial=False,
            output_patch=output_patch,
        )
        return result

    def _length_exit_reason(self, response: Any) -> str:
        cap = int(self.config.max_tokens_per_turn or 0)
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            if (
                self.config.max_context_tokens > 0
                and prompt_tokens + completion_tokens >= self.config.max_context_tokens
            ):
                return "max_context_len"
            if cap and completion_tokens >= cap:
                return "max_tokens_per_turn"
            return "max_context_len"
        return "max_tokens_per_turn" if cap else "max_context_len"


def _message_dict(message: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": "assistant",
        "content": getattr(message, "content", None),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        result["reasoning_content"] = reasoning
    return result


def _is_length_finish(choice: Any) -> bool:
    return (
        str(getattr(choice, "finish_reason", None) or "").lower()
        in _LENGTH_FINISH_REASONS
    )


def _is_context_overflow(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "context_length_exceeded",
            "reduce the length",
            "maximum context length",
        )
    ) or ("context length" in message and "exceed" in message)


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _usage_total(usage: Any) -> int:
    total = getattr(usage, "total_tokens", None)
    if isinstance(total, int):
        return total
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if isinstance(prompt, int) and isinstance(completion, int):
        return prompt + completion
    return 0


def _count_message_tokens(
    messages: list[dict[str, Any]],
    model: str,
) -> int:
    if not messages:
        return 0
    serialized = json.dumps(
        messages,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    tokenizer_id = os.environ.get("FROGNANO_TOKENIZER", "").strip() or model
    tokenizer = _load_huggingface_tokenizer(tokenizer_id)
    if tokenizer is not None:
        try:
            return len(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                )
            )
        except Exception:
            try:
                return len(
                    tokenizer.encode(
                        serialized,
                        add_special_tokens=False,
                    )
                )
            except Exception:
                logger.debug(
                    "Hugging Face tokenizer %s could not count messages",
                    tokenizer_id,
                    exc_info=True,
                )

    encoding = _load_tiktoken_encoding(tokenizer_id)
    if encoding is not None:
        return len(encoding.encode(serialized))

    return math.ceil(len(serialized.encode("utf-8")) / 4)


@lru_cache(maxsize=None)
def _load_huggingface_tokenizer(tokenizer_id: str) -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception:
        logger.debug(
            "Could not load Hugging Face tokenizer %s",
            tokenizer_id,
            exc_info=True,
        )
        return None


@lru_cache(maxsize=None)
def _load_tiktoken_encoding(tokenizer_id: str) -> tiktoken.Encoding | None:
    try:
        return tiktoken.encoding_for_model(tokenizer_id)
    except KeyError:
        try:
            return tiktoken.get_encoding(tokenizer_id)
        except ValueError:
            return None
