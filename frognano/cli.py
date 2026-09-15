from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import replace

from frognano.config import load_config
from frognano.datasets import get_dataset
from frognano.runner import run_evaluation


def _parse_image_registry(value: str) -> str:
    registry = value.strip().rstrip("/")
    if not registry:
        raise argparse.ArgumentTypeError("image registry must be non-empty")
    return registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frognano-eval",
        description="Run FrogNano evaluations with Harbor tasks and Leaf.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run an evaluation config")
    run.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config or bundled config name",
    )
    run.add_argument(
        "--image-registry",
        type=_parse_image_registry,
        help=(
            "Registry prefix for all task images, replacing any source registry "
            "(overrides kubernetes.image_registry)"
        ),
    )
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
    config = load_config(args.config)
    if args.image_registry is not None:
        config = replace(
            config,
            kubernetes=replace(config.kubernetes, image_registry=args.image_registry),
        )
    summary = run_evaluation(config)
    print(json.dumps(summary, indent=2))
    return 0 if summary["jobs_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
