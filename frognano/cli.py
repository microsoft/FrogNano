from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from .config import load_config
from .datasets import get_dataset
from .runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frognano-eval",
        description="Run FrogNano evaluations with Harbor tasks and Leaf.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run an evaluation config")
    run.add_argument("--config", required=True, help="Path to YAML config")
    inspect = subparsers.add_parser("dataset", help="Show a registered dataset source")
    inspect.add_argument("name")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "dataset":
        source, _ = get_dataset(args.name)
        print(json.dumps(source.__dict__, indent=2))
        return 0
    summary = run_evaluation(load_config(args.config))
    print(json.dumps(summary, indent=2))
    return 0 if summary["jobs_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
