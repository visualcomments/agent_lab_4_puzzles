#!/usr/bin/env python3
"""Pipeline CLI

This repo started as a single-competition template. We extend it with:
- per-competition baselines
- per-competition validators
- per-competition prompt bundles
- a CLI switch (competition slug) that selects the right pieces

Key idea: `--competition` selects a pipeline from pipeline_registry.

Examples
--------

# List supported pipelines
python pipeline_cli.py list-pipelines

# Run RapaportM2 pipeline (no LLM, baseline solver) on a local test.csv
python pipeline_cli.py run \
  --competition cayleypy-rapapport-m2 \
  --puzzles /path/to/test.csv \
  --output submission.csv \
  --no-llm

# Run Pancake pipeline (no LLM, baseline solver)
python pipeline_cli.py run \
  --competition CayleyPy-pancake \
  --puzzles /path/to/test.csv \
  --output submission.csv \
  --no-llm

# Generate a new solver with AgentLaboratory for RapaportM2
python pipeline_cli.py generate-solver \
  --competition cayleypy-rapapport-m2 \
  --out generated/rapapport_solve_module.py

"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pipeline_registry import PipelineSpec, get_pipeline, list_pipelines


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compile_all() -> None:
    """Compile all python files to catch syntax errors early."""
    print("[compile] Running python -m compileall ...")
    subprocess.check_call([PYTHON, "-m", "compileall", str(ROOT)])
    print("[compile] OK")


def _load_solve_fn(solver_path: Path) -> Callable[[Sequence[int]], Tuple[Any, Any]]:
    """Dynamically import `solve` from an arbitrary solve_module.py."""
    if not solver_path.exists():
        raise FileNotFoundError(solver_path)

    module_name = f"solve_module_dyn_{abs(hash(str(solver_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, solver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {solver_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    solve = getattr(module, "solve", None)
    if solve is None or not callable(solve):
        raise AttributeError(f"No callable solve(vec) in {solver_path}")

    return solve


def _parse_int_list(s: str) -> List[int]:
    """Parse a list of ints from either JSON '[1,2,3]' or CSV '1,2,3'."""
    s = s.strip()
    if not s:
        return []

    if s[0] in "[(":
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [int(x) for x in obj]
        except Exception:
            # fall back to CSV parsing
            pass

    # CSV style: "3,0,1,4,2"
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _extract_state(row: Dict[str, str], spec: PipelineSpec, vector_col_override: Optional[str] = None) -> List[int]:
    """Extract puzzle state vector from a CSV row based on pipeline spec."""
    # Explicit override wins
    if vector_col_override:
        if vector_col_override not in row:
            raise KeyError(f"vector column {vector_col_override!r} not found in CSV")
        return _parse_int_list(row[vector_col_override])

    # Try pipeline candidates
    if spec.state_columns:
        for col in spec.state_columns:
            if col in row and row[col].strip() != "":
                return _parse_int_list(row[col])

    # Fallback: if row has exactly one non-id column, take it
    non_empty = {k: v for k, v in row.items() if v.strip() != ""}
    for candidate in ("vector", "permutation", "initial_state", "state"):
        if candidate in non_empty:
            return _parse_int_list(non_empty[candidate])

    raise KeyError(
        "Could not infer state column. Please pass --vector-col explicitly or update pipeline_registry.py"
    )


def _validate_solver(solver_path: Path, validator_path: Path, smoke_vector: Sequence[int]) -> None:
    print(f"[validate] {validator_path.name} ...")
    subprocess.check_call(
        [
            PYTHON,
            str(validator_path),
            "--solver",
            str(solver_path),
            "--vector",
            json.dumps(list(smoke_vector)),
        ]
    )


def _ensure_llm_puzzles_on_path() -> None:
    lp_dir = ROOT / "llm-puzzles"
    if str(lp_dir) not in sys.path:
        sys.path.insert(0, str(lp_dir))


def _build_submission(
    *,
    puzzles_csv: Path,
    out_csv: Path,
    competition_format_slug: str,
    solver_path: Path,
    spec: PipelineSpec,
    vector_col_override: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> None:
    """Build a Kaggle submission CSV using llm-puzzles universal_adapter."""
    _ensure_llm_puzzles_on_path()

    from src.universal_adapter import build_submission as lp_build_submission  # type: ignore

    solve_fn = _load_solve_fn(solver_path)

    def row_solver(row: Dict[str, str], cfg: Any) -> Union[List[str], str]:
        vec = _extract_state(row, spec, vector_col_override)
        out = solve_fn(vec)

        # Expected: (moves, sorted_array)
        if isinstance(out, tuple) and len(out) == 2:
            moves = out[0]
            return moves

        # Allow solve() to directly return moves list / string.
        return out  # type: ignore[return-value]

    print(f"[submit] Building submission for format={competition_format_slug}")
    print(f"         puzzles={puzzles_csv}")
    print(f"         output={out_csv}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    lp_build_submission(
        puzzles_csv=str(puzzles_csv),
        output_csv=str(out_csv),
        competition=competition_format_slug,
        solver=row_solver,
        max_rows=max_rows,
    )


# ---------------------------------------------------------------------------
# AgentLaboratory generation
# ---------------------------------------------------------------------------


def _run_agent_laboratory(
    *,
    prompt_file: Path,
    out_path: Path,
    validator: Path,
    baseline: Path,
    custom_prompts: Optional[Path] = None,
    llm: str = "gpt-4o-mini",
    max_iters: int = 8,
    no_llm: bool = False,
    allow_baseline: bool = True,
) -> None:
    """Run AgentLaboratory perm_pipeline to generate/repair a solver."""

    pipeline_script = ROOT / "AgentLaboratory" / "perm_pipeline" / "run_perm_pipeline.py"
    if not pipeline_script.exists():
        raise FileNotFoundError(pipeline_script)

    cmd = [
        PYTHON,
        str(pipeline_script),
        "--user-prompt-file",
        str(prompt_file),
        "--out",
        str(out_path),
        "--validator",
        str(validator),
        "--baseline",
        str(baseline),
        "--max-iters",
        str(max_iters),
        "--models",
        llm,
    ]

    # run_perm_pipeline.py already falls back to baseline unless --strict is used.
    if no_llm:
        cmd.append("--no-llm")
    if custom_prompts:
        cmd.extend(["--custom-prompts", str(custom_prompts)])

    print("[agentlab] " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_list_pipelines(_: argparse.Namespace) -> None:
    print("Available pipelines (competition slugs):")
    for spec in list_pipelines():
        print(f"- {spec.key:45s}  (competition='{spec.competition}', format='{spec.format_slug}')")


def cmd_generate_solver(args: argparse.Namespace) -> None:
    spec = get_pipeline(args.competition) if args.competition else None
    if spec is None:
        raise SystemExit(
            "Unknown --competition. Run `python pipeline_cli.py list-pipelines` to see supported pipelines."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Select prompt bundle (can be overridden)
    prompt_file = Path(args.prompt_file) if args.prompt_file else spec.prompt_file
    if prompt_file is None:
        raise SystemExit(f"No prompt_file configured for {spec.key}; pass --prompt-file.")

    custom_prompts = Path(args.custom_prompts) if args.custom_prompts else spec.custom_prompts_file

    if args.no_llm:
        # Copy baseline
        shutil.copyfile(spec.baseline_solver, out_path)
        print(f"[generate-solver] --no-llm: copied baseline -> {out_path}")
        _validate_solver(out_path, spec.validator, spec.smoke_vector or [0, 1])
        return

    _run_agent_laboratory(
        prompt_file=prompt_file,
        out_path=out_path,
        validator=spec.validator,
        baseline=spec.baseline_solver,
        custom_prompts=custom_prompts,
        llm=args.models,
        max_iters=args.max_iters,
        no_llm=False,
        allow_baseline=args.allow_baseline,
    )

    _validate_solver(out_path, spec.validator, spec.smoke_vector or [0, 1])


def cmd_build_submission(args: argparse.Namespace) -> None:
    spec = get_pipeline(args.competition) if args.competition else None
    if spec is None:
        raise SystemExit(
            "Unknown --competition. Run `python pipeline_cli.py list-pipelines` to see supported pipelines."
        )

    solver_path = Path(args.solver)
    puzzles_csv = Path(args.puzzles)
    out_csv = Path(args.output)

    _build_submission(
        puzzles_csv=puzzles_csv,
        out_csv=out_csv,
        competition_format_slug=args.format or spec.format_slug,
        solver_path=solver_path,
        spec=spec,
        vector_col_override=args.vector_col,
        max_rows=args.max_rows,
    )


def cmd_validate_solver(args: argparse.Namespace) -> None:
    spec = get_pipeline(args.competition) if args.competition else None
    if spec is None:
        raise SystemExit(
            "Unknown --competition. Run `python pipeline_cli.py list-pipelines` to see supported pipelines."
        )

    solver_path = Path(args.solver)
    vec = json.loads(args.vector) if args.vector is not None else list(spec.smoke_vector or [0, 1])

    _validate_solver(solver_path, spec.validator, vec)


def cmd_run(args: argparse.Namespace) -> None:
    # Pipeline selection by competition slug
    spec = get_pipeline(args.competition)
    if spec is None:
        raise SystemExit(
            f"Unknown competition/pipeline '{args.competition}'. Run `python pipeline_cli.py list-pipelines`."
        )

    generated_dir = ROOT / "generated"
    generated_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    solver_path = generated_dir / f"solve_{spec.key}_{ts}.py"

    if args.no_llm:
        shutil.copyfile(spec.baseline_solver, solver_path)
        print(f"[run] --no-llm: copied baseline solver -> {solver_path}")
    else:
        prompt_file = Path(args.prompt_file) if args.prompt_file else spec.prompt_file
        if prompt_file is None:
            raise SystemExit(f"No prompt_file configured for {spec.key}; pass --prompt-file.")

        custom_prompts = Path(args.custom_prompts) if args.custom_prompts else spec.custom_prompts_file

        _run_agent_laboratory(
            prompt_file=prompt_file,
            out_path=solver_path,
            validator=spec.validator,
            baseline=spec.baseline_solver,
            custom_prompts=custom_prompts,
            llm=args.models,
            max_iters=args.max_iters,
            no_llm=False,
            allow_baseline=args.allow_baseline,
        )

    # Smoke validate
    _validate_solver(solver_path, spec.validator, spec.smoke_vector or [0, 1])

    # Build submission
    puzzles_csv = Path(args.puzzles)
    out_csv = Path(args.output)

    _build_submission(
        puzzles_csv=puzzles_csv,
        out_csv=out_csv,
        competition_format_slug=args.format or spec.format_slug,
        solver_path=solver_path,
        spec=spec,
        vector_col_override=args.vector_col,
        max_rows=args.max_rows,
    )

    # Optionally submit to Kaggle
    if args.submit:
        if not args.message:
            raise SystemExit("--submit requires --message")

        cmd = ["kaggle", "competitions", "submit", "-c", spec.competition, "-f", str(out_csv), "-m", args.message]
        print("[kaggle] " + " ".join(cmd))
        subprocess.check_call(cmd)

    print("[run] Done.")


def cmd_selftest(_: argparse.Namespace) -> None:
    """Offline smoke tests (no Kaggle, no LLM)."""
    compile_all()

    tmp = ROOT / "_selftest"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    # Test a few pipelines end-to-end
    to_test = [
        "lrx-sort",
        "cayleypy-rapapport-m2",
        "CayleyPy-pancake",
    ]

    for comp in to_test:
        spec = get_pipeline(comp)
        assert spec is not None
        print(f"\n[selftest] pipeline={spec.key}")

        # Copy baseline solver
        solver_path = tmp / f"{spec.key}_baseline.py"
        shutil.copyfile(spec.baseline_solver, solver_path)

        # Validate on smoke vector
        _validate_solver(solver_path, spec.validator, spec.smoke_vector or [0, 1])

        # Build a tiny puzzles.csv depending on pipeline
        puzzles_csv = tmp / f"{spec.key}_puzzles.csv"
        out_csv = tmp / f"{spec.key}_submission.csv"

        if spec.key == "cayleypy-rapapport-m2":
            # id,n,permutation
            with puzzles_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "n", "permutation"])
                w.writeheader()
                w.writerow({"id": "0", "n": "5", "permutation": "3,0,1,4,2"})
        elif spec.key == "cayleypy-pancake":
            # initial_state_id,initial_state,state_size
            with puzzles_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["initial_state_id", "initial_state", "state_size"])
                w.writeheader()
                w.writerow({"initial_state_id": "0", "initial_state": "3,1,2,0", "state_size": "4"})
        else:
            # generic vector column
            with puzzles_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "vector"])
                w.writeheader()
                w.writerow({"id": "0", "vector": json.dumps(spec.smoke_vector or [3, 1, 2])})

        _build_submission(
            puzzles_csv=puzzles_csv,
            out_csv=out_csv,
            competition_format_slug=spec.format_slug,
            solver_path=solver_path,
            spec=spec,
            vector_col_override=None,
            max_rows=None,
        )

        print(f"[selftest] wrote {out_csv}")

    print("\n[selftest] All OK")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-competition pipeline CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list-pipelines", help="List built-in pipeline configs")
    sp.set_defaults(func=cmd_list_pipelines)

    sp = sub.add_parser("generate-solver", help="Generate/repair a solver with AgentLaboratory")
    sp.add_argument("--competition", required=True, help="Competition slug / pipeline key")
    sp.add_argument("--out", required=True, help="Output path for generated solver")
    sp.add_argument("--prompt-file", default=None, help="Override user prompt file")
    sp.add_argument("--custom-prompts", default=None, help="Override AgentLaboratory custom prompts JSON")
    sp.add_argument("--models", dest="models", default="gpt-4o-mini", help="Comma-separated g4f model list (passed to AgentLaboratory --models)")
    sp.add_argument("--llm", dest="models", default=None, help=argparse.SUPPRESS)
    sp.add_argument("--max-iters", type=int, default=8)
    sp.add_argument("--allow-baseline", action="store_true")
    sp.add_argument("--no-llm", action="store_true", help="Skip LLM: just copy baseline")
    sp.set_defaults(func=cmd_generate_solver)

    sp = sub.add_parser("build-submission", help="Build a submission CSV from a solver")
    sp.add_argument("--competition", required=True, help="Competition slug / pipeline key")
    sp.add_argument("--puzzles", required=True, help="Input puzzles/test CSV")
    sp.add_argument("--solver", required=True, help="Path to solve_module.py")
    sp.add_argument("--output", required=True, help="Submission CSV output")
    sp.add_argument("--format", default=None, help="Override llm-puzzles format slug")
    sp.add_argument("--vector-col", default=None, help="Override state column")
    sp.add_argument("--max-rows", type=int, default=None)
    sp.set_defaults(func=cmd_build_submission)


    sp = sub.add_parser("validate-solver", help="Validate a solver with the competition-specific validator")
    sp.add_argument("--competition", required=True, help="Competition slug / pipeline key")
    sp.add_argument("--solver", required=True, help="Path to solve_module.py")
    sp.add_argument("--vector", default=None, help="Optional JSON list. If omitted, uses the pipeline smoke vector.")
    sp.set_defaults(func=cmd_validate_solver)

    sp = sub.add_parser("run", help="End-to-end: (generate solver) -> validate -> build submission -> optional Kaggle submit")
    sp.add_argument("--competition", required=True, help="Competition slug / pipeline key")
    sp.add_argument("--puzzles", required=True, help="Input puzzles/test CSV")
    sp.add_argument("--output", required=True, help="Submission CSV output")
    sp.add_argument("--prompt-file", default=None, help="Override user prompt file")
    sp.add_argument("--custom-prompts", default=None, help="Override custom prompts JSON")
    sp.add_argument("--models", dest="models", default="gpt-4o-mini", help="Comma-separated g4f model list (passed to AgentLaboratory --models)")
    sp.add_argument("--llm", dest="models", default=None, help=argparse.SUPPRESS)
    sp.add_argument("--max-iters", type=int, default=8)
    sp.add_argument("--allow-baseline", action="store_true")
    sp.add_argument("--no-llm", action="store_true")
    sp.add_argument("--format", default=None, help="Override llm-puzzles format slug")
    sp.add_argument("--vector-col", default=None, help="Override state column")
    sp.add_argument("--max-rows", type=int, default=None)
    sp.add_argument("--submit", action="store_true")
    sp.add_argument("--message", default=None, help="Kaggle submission message")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("selftest", help="Offline smoke tests")
    sp.set_defaults(func=cmd_selftest)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
