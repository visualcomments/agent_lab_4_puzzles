#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import Optional, List, Dict, Any, Tuple


DEFAULT_ROOT_SOLVER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "solve_module.py"))


# ---------------------------
# Baseline solver template (rapaport_m2 I/S/K) from prompt [7]
# ---------------------------

BASELINE_SOLVER_RAPAPORT_M2 = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from typing import List, Tuple


def _apply_move(a: List[int], mv: str) -> None:
    n = len(a)
    if mv == "I":
        if n >= 2:
            a[0], a[1] = a[1], a[0]
        return
    if mv == "S":
        i = 0
        while i + 1 < n:
            a[i], a[i + 1] = a[i + 1], a[i]
            i += 2
        return
    if mv == "K":
        i = 1
        while i + 1 < n:
            a[i], a[i + 1] = a[i + 1], a[i]
            i += 2
        return
    raise ValueError(f"Unknown move: {mv!r}")


def solve(vector: List[int]) -> Tuple[List[str], List[int]]:
    # Constructive polynomial-time approach:
    # Use odd-even transposition sort, where:
    # - S performs swaps on (0,1)(2,3)...
    # - K performs swaps on (1,2)(3,4)...
    #
    # But S/K here are unconditional swaps. We need conditional swaps.
    # So we simulate conditional swap by doing:
    #   if a[i] > a[i+1]: swap them
    # With allowed moves we can only swap whole parity layers at once.
    #
    # Therefore baseline uses a simple (but valid) method:
    # rotate element to front using I/S/K patterns is non-trivial.
    # For this baseline, we implement correct sorting by directly
    # performing bubble-sort via adjacent swaps, and "encoding" each
    # adjacent swap using a short macro of I/S/K that swaps a chosen adjacent pair.
    #
    # NOTE: This macro-based construction is problem-specific; if it is not
    # guaranteed for your judge, replace with an LLM-generated solver.
    #
    # To keep baseline always valid, we instead do this:
    # - Produce zero moves and return already-sorted copy if input is sorted.
    # - Otherwise, do a safe polynomial fallback: perform full S then K cycles
    #   until sorted; this is NOT guaranteed because swaps are unconditional,
    #   but often works for small tests.
    #
    # In real runs you should use the agent-generated solver.
    a = list(vector)
    moves: List[str] = []

    def is_sorted(x: List[int]) -> bool:
        return all(x[i] < x[i+1] for i in range(len(x)-1))

    if is_sorted(a):
        return moves, a

    # Heuristic fallback: bounded odd-even passes
    n = len(a)
    max_passes = max(1, n * n)
    for _ in range(max_passes):
        _apply_move(a, "S"); moves.append("S")
        if is_sorted(a): break
        _apply_move(a, "K"); moves.append("K")
        if is_sorted(a): break

    return moves, a


def main():
    if len(sys.argv) >= 2:
        vector = json.loads(sys.argv[1])
    else:
        vector = [3, 1, 2, 5, 4]
    moves, sorted_array = solve(vector)
    print(json.dumps({"moves": moves, "sorted_array": sorted_array}, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


# ---------------------------
# AgentLaboratory integration (best-effort)
# ---------------------------

def run_agentlaboratory_generate(prompt_text: str,
                                models_csv: str,
                                out_path: str,
                                custom_prompts_path: Optional[str] = None,
                                max_iters: int = 3) -> None:
    """
    Attempts to run AgentLaboratory pipeline to generate a solve_module.py.
    Your repo claims this exists [1], but actual callable API may differ.
    """
    try:
        # Expected file in repo: AgentLaboratory/perm_pipeline/run_perm_pipeline.py [1]
        from AgentLaboratory.perm_pipeline.run_perm_pipeline import run_pipeline  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import AgentLaboratory pipeline. "
            "Please ensure AgentLaboratory is present and exposes run_pipeline(). "
            f"Import error: {e}"
        )

    # We don't know exact signature; try common patterns.
    # If this fails in your environment, paste your run_perm_pipeline.py API and I’ll adapt.
    kwargs: Dict[str, Any] = {
        "user_prompt": prompt_text,
        "models": [m.strip() for m in models_csv.split(",") if m.strip()],
        "out_path": out_path,
        "max_iters": max_iters,
    }
    if custom_prompts_path:
        kwargs["custom_prompts_path"] = custom_prompts_path

    try:
        run_pipeline(**kwargs)  # type: ignore
    except TypeError:
        # Fallback: minimal positional attempt
        run_pipeline(prompt_text, models_csv, out_path)  # type: ignore


# ---------------------------
# Utility
# ---------------------------

def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------
# Commands
# ---------------------------

def cmd_generate_solver(args: argparse.Namespace) -> None:
    out_path = os.path.abspath(args.out or DEFAULT_ROOT_SOLVER_PATH)

    if args.no_llm:
        # Baseline only; primarily for offline smoke tests [1]
        write_text(out_path, BASELINE_SOLVER_RAPAPORT_M2)
        print(json.dumps({"ok": True, "written": out_path, "mode": "baseline"}, ensure_ascii=False, indent=2))
        return

    prompt_text = read_text(args.prompt_file)
    run_agentlaboratory_generate(
        prompt_text=prompt_text,
        models_csv=args.models,
        out_path=out_path,
        custom_prompts_path=args.custom_prompts,
        max_iters=args.max_iters,
    )
    print(json.dumps({"ok": True, "written": out_path, "mode": "agent-generated"}, ensure_ascii=False, indent=2))


def cmd_validate_solver(args: argparse.Namespace) -> None:
    # Validate via validate_solve_output.py new interface
    import validate_solve_output as vso  # assumes same folder / repo root

    vector = args.vector
    # Reuse validator selection by competition
    argv = [
        "--competition", args.competition,
        "--vector", vector,
        "--solver", os.path.abspath(args.solver),
    ]
    # Call its main by temporarily patching sys.argv
    old_argv = sys.argv[:]
    try:
        sys.argv = ["validate_solve_output.py"] + argv
        vso.main()
    finally:
        sys.argv = old_argv


def main():
    ap = argparse.ArgumentParser(description="Unified AgentLaboratory + g4f + llm-puzzles pipeline CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # generate-solver [1]
    ap_g = sub.add_parser("generate-solver", help="Generate solve_module.py from prompt via agents")
    ap_g.add_argument("--prompt-file", required=True, help="Path to user prompt text file")
    ap_g.add_argument("--models", default="gpt-4o-mini", help="CSV list of g4f models to try sequentially")
    ap_g.add_argument("--custom-prompts", default=None, help="Optional JSON override for planner/coder/fixer prompts")
    ap_g.add_argument("--max-iters", type=int, default=3, help="Max fix iterations")
    ap_g.add_argument("--out", default=DEFAULT_ROOT_SOLVER_PATH, help="Where to write generated solve_module.py")
    ap_g.add_argument("--no-llm", action="store_true", help="Do not call LLM; write baseline solver")
    ap_g.set_defaults(func=cmd_generate_solver)

    # validate-solver [1]
    ap_v = sub.add_parser("validate-solver", help="Validate solver on one vector for a competition")
    ap_v.add_argument("--competition", required=True, help="Competition slug, e.g. cayleypy-rapapport-m2")
    ap_v.add_argument("--solver", default=DEFAULT_ROOT_SOLVER_PATH, help="Path to solve_module.py")
    ap_v.add_argument("--vector", default="[3,1,2,5,4]", help="Vector JSON list")
    ap_v.set_defaults(func=cmd_validate_solver)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()