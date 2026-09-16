"""Failures at the task execution boundary."""


class CommandTimeoutError(TimeoutError):
    """A command did not finish within its execution deadline."""

    def __init__(self, message: str, *, output: str = "") -> None:
        super().__init__(message)
        self.output = output


class PodExecutionError(RuntimeError):
    """The pod could not provide a complete, verified execution result."""
