#!/usr/bin/env python3
"""pipeline_cli.py

Unified CLI for:
- AgentLaboratory (multi-agent solver generation, g4f-backed)
- Local validation of the generated solver
- llm-puzzles submission building
- Optional Kaggle submit + score retrieval

Design goals:
- Works offline for everything except (g4f calls, Kaggle submit/score)
- No hard dependency on kaggle unless you ask to submit/score
- Uses bundled gpt4free checkout if present (./gpt4free) OR pip-installed g4f

Run `python pipeline_cli.py -h` for full help.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
AGENTLAB = ROOT / "AgentLaboratory"
LLMPUZZLES = ROOT / "llm-puzzles"
G4F_VENDOR = ROOT / "gpt4free"  # bundled checkout (optional)


def _add_repo_paths() -> None:
    """Ensure imports work even if nothing is pip-installed."""
    for p in [str(LLMPUZZLES), str(AGENTLAB)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    # Allow `import g4f` from bundled checkout if present
    if G4F_VENDOR.exists() and str(G4F_VENDOR) not in sys.path:
        sys.path.insert(0, str(G4F_VENDOR))


def _read_text(path: Optional[str]) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8")


def _run(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)


def generate_solver(
    prompt_text: str,
    models_csv: str,
    out_path: Path,
    custom_prompts: Optional[str] = None,
    max_iters: int = 4,
    no_llm: bool = False,
) -> None:
    """Generate (or repair) a solver via AgentLaboratory perm_pipeline."""
    runner = AGENTLAB / "perm_pipeline" / "run_perm_pipeline.py"
    if not runner.exists():
        raise FileNotFoundError(f"AgentLaboratory runner not found: {runner}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write prompt to a file to avoid shell quoting issues
    gen_dir = out_path.parent
    prompt_file = gen_dir / "prompt.txt"
    prompt_file.write_text(prompt_text, encoding="utf-8")

    cmd = [
        sys.executable,
        str(runner),
        "--user-prompt-file",
        str(prompt_file),
        "--models",
        models_csv,
        "--out",
        str(out_path),
        "--validator",
        str(ROOT / "validate_solve_output.py"),
        "--max-iters",
        str(max_iters),
    ]
    if custom_prompts:
        cmd += ["--custom-prompts", custom_prompts]
    if no_llm:
        cmd += ["--no-llm"]

    res = _run(cmd, cwd=ROOT)
    sys.stdout.write(res.stdout)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"Solver generation failed with exit={res.returncode}")


def validate_solver(solver_path: Path, vector: List[int]) -> None:
    validator = ROOT / "validate_solve_output.py"
    if not validator.exists():
        raise FileNotFoundError(f"validator not found: {validator}")
    cmd = [sys.executable, str(validator), "--solver", str(solver_path), "--vector", json.dumps(vector)]
    res = _run(cmd, cwd=ROOT)
    sys.stdout.write(res.stdout)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError("Solver failed validation")



def build_submission(
    puzzles_csv: Path,
    out_csv: Path,
    competition_slug_for_format: str,
    solver_path: Path,
    vector_col: Optional[str] = None,
    add_config: Optional[str] = None,
) -> None:
    """Build submission.csv using llm-puzzles universal_adapter."""
    _add_repo_paths()

    if add_config:
        # Use the real module path inside llm-puzzles
        from src import comp_registry  # type: ignore
        data = json.loads(Path(add_config).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "slug" in data:
            items = [data]
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError("--add-config expects a JSON object or list of objects")
        for obj in items:
            cfg = comp_registry.CompConfig(**obj)
            comp_registry.REGISTRY[cfg.slug] = cfg

    # Copy solver into examples/agentlab_sort so the adapter can import it
    ex_dir = LLMPUZZLES / "examples" / "agentlab_sort"
    ex_dir.mkdir(parents=True, exist_ok=True)
    (ex_dir / "solve_module.py").write_text(solver_path.read_text(encoding="utf-8"), encoding="utf-8")

    if vector_col:
        os.environ["VECTOR_COL"] = vector_col

    from src.universal_adapter import build_submission as _build  # type: ignore
    from examples.agentlab_sort.solver import solve_row  # type: ignore

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _build(str(puzzles_csv), str(out_csv), competition_slug_for_format, solve_row)


def kaggle_submit(competition: str, file: Path, message: str) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except Exception as e:
        raise RuntimeError("kaggle package not installed. Run: pip install kaggle") from e

    api = KaggleApi()
    api.authenticate()
    api.competition_submit(file_name=str(file), message=message, competition=competition)


def kaggle_latest_score(competition: str) -> Dict[str, Any]:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except Exception as e:
        raise RuntimeError("kaggle package not installed. Run: pip install kaggle") from e

    api = KaggleApi()
    api.authenticate()
    subs = api.competition_submissions(competition) or []
    # Best-effort: return the first one with a score
    for s in subs:
        d = getattr(s, "__dict__", {})
        ps = d.get("publicScore") or d.get("public_score")
        prs = d.get("privateScore") or d.get("private_score")
        if ps not in (None, "", "None") or prs not in (None, "", "None"):
            return {"status": d.get("status"), "public_score": ps, "private_score": prs}
    return {"status": "no_scored_submissions"}


def selftest(seed: int = 0) -> None:
    """Offline self-test: compile, validate solver on random cases, build a dummy submission."""
    random.seed(seed)

    # 1) Compile everything
    print("[selftest] compileall ...")
    res = _run([sys.executable, "-m", "compileall", str(ROOT)], cwd=ROOT)
    if res.returncode != 0:
        sys.stdout.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError("compileall failed")

    # 2) Validate baseline solver
    baseline = ROOT / "solve_module.py"
    if not baseline.exists():
        raise FileNotFoundError("baseline solve_module.py not found at repo root")

    for n in [1, 2, 3, 5, 7]:
        vec = random.sample(range(-50, 50), n)
        print(f"[selftest] validate baseline n={n} vec={vec}")
        validate_solver(baseline, vec)

    # 3) Dummy puzzles.csv -> submission
    print("[selftest] build dummy submission ...")
    dummy = ROOT / "_selftest" / "puzzles.csv"
    dummy.parent.mkdir(parents=True, exist_ok=True)
    dummy.write_text("id,vector\n1,\"[3,1,2]\"\n2,\"[4,2,1,3]\"\n", encoding="utf-8")

    out_csv = ROOT / "_selftest" / "submission.csv"
    build_submission(dummy, out_csv, "format/moves-dot", baseline, vector_col="vector")
    if not out_csv.exists() or out_csv.stat().st_size < 10:
        raise RuntimeError("submission.csv not created")
    print(f"[selftest] OK: {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified AgentLaboratory + llm-puzzles + Kaggle CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # selftest
    sp = sub.add_parser("selftest", help="Run offline self-test (no LLM, no Kaggle)")
    sp.add_argument("--seed", type=int, default=0)

    # generate-solver
    sp = sub.add_parser("generate-solver", help="Generate or repair solve_module.py using AgentLaboratory")
    sp.add_argument("--prompt", default="")
    sp.add_argument("--prompt-file", default=None)
    sp.add_argument("--models", default=os.getenv("G4F_MODELS", "gpt-4o-mini,command-r,aria"))
    sp.add_argument("--custom-prompts", default=None)
    sp.add_argument("--out", default=str(ROOT / "generated" / "solve_module.py"))
    sp.add_argument("--max-iters", type=int, default=4)
    sp.add_argument("--no-llm", action="store_true")

    # validate-solver
    sp = sub.add_parser("validate-solver", help="Validate a solver on a single vector")
    sp.add_argument("--solver", required=True)
    sp.add_argument("--vector", required=True, help="JSON list, e.g. '[3,1,2]' ")

    # build-submission
    sp = sub.add_parser("build-submission", help="Build submission.csv from puzzles.csv")
    sp.add_argument("--puzzles", required=True)
    sp.add_argument("--out", default="submission.csv")
    sp.add_argument("--format", default="format/moves-dot", help="llm-puzzles output format slug")
    sp.add_argument("--solver", required=True)
    sp.add_argument("--vector-col", default=None)
    sp.add_argument("--add-config", default=None)

    # submit
    sp = sub.add_parser("submit", help="Submit a prepared CSV to Kaggle")
    sp.add_argument("--competition", required=True)
    sp.add_argument("--file", required=True)
    sp.add_argument("--message", default="auto-submit")

    # score
    sp = sub.add_parser("score", help="Fetch latest scored submission info")
    sp.add_argument("--competition", required=True)

    # run (end-to-end)
    sp = sub.add_parser("run", help="End-to-end: generate -> validate -> build -> (optional submit/score)")
    sp.add_argument("--competition", required=True, help="Kaggle competition slug (used for submit/score)")
    sp.add_argument("--puzzles", required=True, help="Path to puzzles.csv downloaded from Kaggle")
    sp.add_argument("--out", default="submission.csv")
    sp.add_argument("--format", default="format/moves-dot", help="llm-puzzles output format slug")
    sp.add_argument("--prompt", default="")
    sp.add_argument("--prompt-file", default=None)
    sp.add_argument("--models", default=os.getenv("G4F_MODELS", "gpt-4o-mini,command-r,aria"))
    sp.add_argument("--custom-prompts", default=None)
    sp.add_argument("--max-iters", type=int, default=4)
    sp.add_argument("--no-llm", action="store_true", help="Skip LLM calls; use baseline solver")
    sp.add_argument("--solver-out", default=str(ROOT / "generated" / "solve_module.py"))
    sp.add_argument("--vector-col", default=None)
    sp.add_argument("--add-config", default=None)
    sp.add_argument("--submit", action="store_true")
    sp.add_argument("--message", default="agentlab auto-submit")
    sp.add_argument("--print-score", action="store_true")

    args = ap.parse_args()

    if args.cmd == "selftest":
        selftest(seed=args.seed)
        return

    if args.cmd == "generate-solver":
        prompt = (args.prompt or "").strip() or _read_text(args.prompt_file).strip()
        if not prompt:
            raise SystemExit("Empty prompt. Provide --prompt or --prompt-file")
        generate_solver(prompt, args.models, Path(args.out), custom_prompts=args.custom_prompts, max_iters=args.max_iters, no_llm=args.no_llm)
        return

    if args.cmd == "validate-solver":
        validate_solver(Path(args.solver), json.loads(args.vector))
        return

    if args.cmd == "build-submission":
        build_submission(Path(args.puzzles), Path(args.out), args.format, Path(args.solver), vector_col=args.vector_col, add_config=args.add_config)
        print(f"[+] Saved: {args.out}")
        return

    if args.cmd == "submit":
        kaggle_submit(args.competition, Path(args.file), args.message)
        print("[+] Submitted")
        return

    if args.cmd == "score":
        print(json.dumps(kaggle_latest_score(args.competition), ensure_ascii=False))
        return

    if args.cmd == "run":
        prompt = (args.prompt or "").strip() or _read_text(args.prompt_file).strip()
        if not prompt and not args.no_llm:
            raise SystemExit("Empty prompt. Provide --prompt/--prompt-file or set --no-llm")

        solver_out = Path(args.solver_out)
        if args.no_llm:
            # Use baseline solver at repo root
            solver_out.parent.mkdir(parents=True, exist_ok=True)
            solver_out.write_text((ROOT / "solve_module.py").read_text(encoding="utf-8"), encoding="utf-8")
        else:
            generate_solver(prompt, args.models, solver_out, custom_prompts=args.custom_prompts, max_iters=args.max_iters, no_llm=False)

        # Validate on a smoke test
        validate_solver(solver_out, [3, 1, 2, 5, 4])

        # Build submission
        build_submission(Path(args.puzzles), Path(args.out), args.format, solver_out, vector_col=args.vector_col, add_config=args.add_config)
        print(f"[+] Saved: {args.out}")

        if args.submit:
            kaggle_submit(args.competition, Path(args.out), args.message)
            print("[+] Submitted")
            if args.print_score:
                print(json.dumps(kaggle_latest_score(args.competition), ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
