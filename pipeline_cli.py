#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT_SOLVER_PATH = str((REPO_ROOT / "solve_module.py").resolve())

# slug -> default prompt file
DEFAULT_PROMPT_BY_COMPETITION: Dict[str, str] = {
    "cayleypy-rapapport-m2": str((REPO_ROOT / "prompts" / "rapaport_m2.txt").resolve()),
}


def run_cmd(cmd, *, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, text=True)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p


def cmd_generate_solver(args: argparse.Namespace) -> None:
    out_path = os.path.abspath(args.out or DEFAULT_ROOT_SOLVER_PATH)

    # prompt: either explicit --prompt-file, or inferred from competition
    prompt_file = args.prompt_file
    if not prompt_file:
        if args.competition and args.competition in DEFAULT_PROMPT_BY_COMPETITION:
            prompt_file = DEFAULT_PROMPT_BY_COMPETITION[args.competition]
        else:
            raise SystemExit("Provide --prompt-file or --competition with known default prompt mapping.")

    cmd = [
        sys.executable,
        str((REPO_ROOT / "AgentLaboratory" / "perm_pipeline" / "run_perm_pipeline.py").resolve()),
        "--competition", args.competition,
        "--user-prompt-file", prompt_file,
        "--models", args.models,
        "--out", out_path,
        "--max-iters", str(args.max_iters),
        "--validator", str((REPO_ROOT / "validate_solve_output.py").resolve()),
    ]
    if args.custom_prompts:
        cmd += ["--custom-prompts", args.custom_prompts]
    if args.no_llm:
        cmd += ["--no-llm"]
    if args.strict:
        cmd += ["--strict"]

    run_cmd(cmd, check=True)
    print(json.dumps({"ok": True, "solver_written": out_path}, ensure_ascii=False))


def cmd_validate_solver(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str((REPO_ROOT / "validate_solve_output.py").resolve()),
        "--competition", args.competition,
        "--solver", os.path.abspath(args.solver),
        "--vector", args.vector,
    ]
    run_cmd(cmd, check=False)


def main():
    ap = argparse.ArgumentParser(description="Unified AgentLaboratory + g4f pipeline CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_g = sub.add_parser("generate-solver", help="Generate solve_module.py via agents")
    ap_g.add_argument("--competition", required=True, help="Competition slug, e.g. cayleypy-rapapport-m2")
    ap_g.add_argument("--prompt-file", default=None, help="User prompt file (optional if known slug mapping exists)")
    ap_g.add_argument("--models", default="gpt-4o-mini,command-r,aria", help="Comma-separated g4f model names")
    ap_g.add_argument("--custom-prompts", default=None, help="JSON override for planner/coder/fixer prompts")
    ap_g.add_argument("--max-iters", type=int, default=4, help="Max repair iterations")
    ap_g.add_argument("--out", default=DEFAULT_ROOT_SOLVER_PATH, help="Where to write solver (default: ./solve_module.py)")
    ap_g.add_argument("--no-llm", action="store_true", help="Do not call LLM; write baseline")
    ap_g.add_argument("--strict", action="store_true", help="Fail if solver doesn't validate; no baseline fallback")
    ap_g.set_defaults(func=cmd_generate_solver)

    ap_v = sub.add_parser("validate-solver", help="Validate solver on one vector")
    ap_v.add_argument("--competition", required=True, help="Competition slug, e.g. cayleypy-rapapport-m2")
    ap_v.add_argument("--solver", default=DEFAULT_ROOT_SOLVER_PATH, help="Path to solve_module.py")
    ap_v.add_argument("--vector", default="[3,1,2,5,4]", help="Vector JSON list")
    ap_v.set_defaults(func=cmd_validate_solver)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()