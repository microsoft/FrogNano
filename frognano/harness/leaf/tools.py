"""Leaf system prompt and tool schemas."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a software-engineering agent working in a checked-out repository.\n"
    "Use the available tools to inspect files, make focused edits, and run commands.\n"
    "Solve the task, validate when practical, then give a brief final answer."
)

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a text file. Returns numbered lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Optional 1-based starting line.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum number of lines.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": (
                "Write a complete file, creating parent directories when needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file contents.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Replace an exact string in a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to replace.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": (
                            "Replace every occurrence instead of requiring one match."
                        ),
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Find files by glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. **/*.py.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional directory to search from.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command in the repository workdir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to run.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of the command.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Optional timeout in seconds.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]
