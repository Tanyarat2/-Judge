#!/usr/bin/env python3
"""One-folder entry point for the mtRAG answer-quality judge.

All active judging code and prompts are in judge_run. Run artifacts are kept
under judge_run/runs/<name>; conversation.json remains the read-only source.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import List, Optional

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
STAGES = ("prepare", "preview", "mtrag", "annotate", "serve")


def run_dir_for(name: str, runs_dir: pathlib.Path) -> pathlib.Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("run name must use only letters, digits, underscores, and hyphens")
    return runs_dir / name


def require_file(path: pathlib.Path, instruction: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit("Missing %s. %s" % (path, instruction))


def require_new(path: pathlib.Path) -> None:
    if path.exists():
        raise SystemExit("Already exists: %s. Use a new --name to preserve this result." % path)


def api_arguments() -> List[str]:
    url = os.environ.get("JUDGE_BASE_URL")
    model = os.environ.get("JUDGE_MODEL")
    key_name = os.environ.get("JUDGE_API_KEY_ENV")
    if not url or not model or not key_name or not os.environ.get(key_name):
        raise SystemExit(
            "Set JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_API_KEY_ENV, and its API key "
            "in this PowerShell session before calling a judge."
        )
    return ["--base-url", url, "--model", model, "--api-key-env", key_name]


def run_paper(stage: str, input_path: pathlib.Path, output_path: pathlib.Path,
              idk_max_tokens: Optional[int] = None, no_max_tokens: bool = False,
              max_tokens: Optional[int] = None,
              reasoning_max_tokens: Optional[int] = None) -> None:
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if any(record.get("contexts") is None for record in records):
        raise SystemExit("mtRAG needs recorded selected contexts for every record.")
    if any(any(context is None for context in record["contexts"])
           for record in records if record.get("contexts") is not None):
        raise SystemExit("mtRAG cannot read null context entries; selected passages must be recorded.")
    try:
        subprocess.run(
            [sys.executable, str(HERE / "judge_paper.py"),
             "--input", str(input_path), "--output", str(output_path), "--mode", "mtrag",
             *(["--idk-max-tokens", str(idk_max_tokens)] if idk_max_tokens is not None else []),
             *(["--no-max-tokens"] if no_max_tokens else []),
             *(["--max-tokens", str(max_tokens)] if max_tokens is not None else []),
             *(["--reasoning-max-tokens", str(reasoning_max_tokens)]
               if reasoning_max_tokens is not None else []),
             *api_arguments()],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        # The paper judge already printed the failed record and its cause.
        raise SystemExit(exc.returncode) from None


def main(argv: Optional[List[str]] = None, runs_dir: Optional[pathlib.Path] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--name", default="pilot3", help="run folder name under judge_run/runs")
    parser.add_argument("--limit", type=int, help="first N conversations; prepare stage only")
    parser.add_argument("--idk-max-tokens", type=int,
                        help="initial IDK completion-token budget for mtrag; retries double it")
    parser.add_argument("--no-max-tokens", action="store_true",
                        help="omit request max_tokens for mtrag; provider/model limits still apply")
    parser.add_argument("--max-tokens", type=int,
                        help="completion-token ceiling per mtRAG request when reasoning is budgeted")
    parser.add_argument("--reasoning-max-tokens", type=int,
                        help="OpenRouter thinking-token budget per mtRAG request (requires --max-tokens)")
    args = parser.parse_args(argv)
    if args.limit is not None and (args.stage != "prepare" or args.limit < 1):
        parser.error("--limit must be a positive number and is used only with prepare")
    if args.idk_max_tokens is not None and (args.stage != "mtrag" or args.idk_max_tokens < 64):
        parser.error("--idk-max-tokens must be >= 64 and is used only with mtrag")
    if args.no_max_tokens and args.stage != "mtrag":
        parser.error("--no-max-tokens is used only with mtrag")
    if args.no_max_tokens and args.idk_max_tokens is not None:
        parser.error("--no-max-tokens cannot be combined with --idk-max-tokens")
    if args.max_tokens is not None and (args.stage != "mtrag" or args.max_tokens < 64):
        parser.error("--max-tokens must be >= 64 and is used only with mtrag")
    if args.reasoning_max_tokens is not None and (args.stage != "mtrag" or args.reasoning_max_tokens < 1):
        parser.error("--reasoning-max-tokens must be positive and is used only with mtrag")
    if args.no_max_tokens and args.max_tokens is not None:
        parser.error("--no-max-tokens cannot be combined with --max-tokens")
    if args.max_tokens is not None and args.reasoning_max_tokens is None:
        parser.error("--max-tokens here requires --reasoning-max-tokens")
    if args.reasoning_max_tokens is not None and (
            args.max_tokens is None or args.no_max_tokens or
            args.max_tokens <= args.reasoning_max_tokens):
        parser.error("--reasoning-max-tokens requires --max-tokens larger than the reasoning budget")
    if args.reasoning_max_tokens is not None and args.idk_max_tokens is not None:
        parser.error("--reasoning-max-tokens cannot be combined with --idk-max-tokens")
    run_dir = run_dir_for(args.name, runs_dir if runs_dir is not None else HERE / "runs")
    input_path = run_dir / "input.json"

    if args.stage == "prepare":
        require_new(run_dir)
        command = [sys.executable, str(HERE / "prepare_input.py"),
                   "--src", str(ROOT / "conversation.json"), "--out", str(run_dir)]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        subprocess.run(command, check=True)
        return

    if args.stage == "serve":
        subprocess.run([sys.executable, str(HERE / "annotation_server.py"),
                        "--runs-dir", str(run_dir.parent)], check=True)
        return

    require_file(input_path, "Run: python run.py prepare --name %s --limit 3" % args.name)

    if args.stage == "preview":
        output_path = run_dir / "preview.json"
        require_new(output_path)
        model = os.environ.get("JUDGE_MODEL")
        if not model:
            raise SystemExit("Set JUDGE_MODEL before preview so it matches the judge run.")
        command = [sys.executable, str(HERE / "judge_paper.py"),
                   "--input", str(input_path), "--dry-run", "--mode", "mtrag", "--model", model]
        result = subprocess.run(command, capture_output=True)
        if result.returncode:
            raise SystemExit(result.stderr.decode("utf-8", errors="replace"))
        with output_path.open("xb") as stream:
            stream.write(result.stdout)
        print("Preview saved: %s (no API call)" % output_path)
        return

    if args.stage == "annotate":
        require_new(run_dir / "annotate.html")
        subprocess.run([sys.executable, str(HERE / "make_annotation.py"),
                        "--work-dir", str(run_dir)], check=True)
        return

    output_path = run_dir / "out.mtrag.json"
    require_new(output_path)
    run_paper(args.stage, input_path, output_path, args.idk_max_tokens, args.no_max_tokens,
              args.max_tokens, args.reasoning_max_tokens)


if __name__ == "__main__":
    main()
