#!/usr/bin/env python3
"""
run_agentlab_comp.py

End-to-end pipeline:
1) (Optional) Generate / repair a constructive solver using AgentLaboratory (g4f-backed agents).
2) Validate the solver locally using validate_solve_output.py.
3) Use llm-puzzles universal_adapter to build a Kaggle submission CSV.
4) (Optional) Submit via Kaggle API and fetch latest scored submission.

Typical usage:
  python run_agentlab_comp.py \
      --competition <kaggle-slug> \
      --puzzles /path/to/puzzles.csv \
      --prompt-file ./prompt.txt \
      --submit --message "agentlab auto"
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from src.universal_adapter import build_submission


def _kaggle():
    """Lazy import Kaggle utilities so running without kaggle installed still works."""
    try:
        from src.kaggle_utils import ensure_auth, submit_file, latest_scored_submission
        return ensure_auth, submit_file, latest_scored_submission
    except Exception as e:
        raise RuntimeError(
            "Kaggle submission requested, but kaggle package/auth is not available. "
            "Install extras (pip install kaggle) and configure kaggle.json / env vars."
        ) from e


def _read_text(p: Optional[str]) -> str:
    if not p:
        return ""
    return Path(p).read_text(encoding="utf-8")

def main():
    p = argparse.ArgumentParser(description="AgentLaboratory + llm-puzzles unified Kaggle pipeline")
    p.add_argument("--competition", required=True, help="Kaggle competition slug")
    p.add_argument("--puzzles", required=True, help="Path to puzzles.csv (downloaded from Kaggle)")
    p.add_argument("--out", default="submission.csv", help="Output submission CSV path")
    p.add_argument("--prompt", default="", help="User prompt (inline)")
    p.add_argument("--prompt-file", default=None, help="Path to text file with user prompt")
    p.add_argument("--models", default=os.getenv("G4F_MODELS","").strip() or "gpt-4o-mini,command-r,aria",
                   help="Comma-separated g4f model names (fallback order)")
    p.add_argument("--custom-prompts", default=None, help="Path to JSON overriding AgentLab system prompts")
    p.add_argument("--no-regen", action="store_true", help="Skip regeneration; use existing examples/agentlab_sort/solve_module.py")
    p.add_argument("--no-llm", action="store_true", help="Generate solver using baseline (skip LLM calls)")
    p.add_argument("--submit", action="store_true", help="Submit to Kaggle after building submission.csv")
    p.add_argument("--message", default="agentlab auto-submit", help="Kaggle submission message")
    p.add_argument("--print-score", action="store_true", help="After submit, try to print latest scored submission")
    p.add_argument("--agentlab-path", default=os.getenv("AGENTLAB_PATH",""),
                   help="Path to AgentLaboratory repo (if not sibling of llm-puzzles)")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent
    agentlab_path = Path(args.agentlab_path).resolve() if args.agentlab_path else (repo_root.parent / "AgentLaboratory")
    agentlab_runner = agentlab_path / "perm_pipeline" / "run_perm_pipeline.py"
    if not agentlab_runner.exists() and not args.no_regen:
        raise SystemExit(f"AgentLaboratory runner not found: {agentlab_runner}. Use --agentlab-path or --no-regen.")

    prompt_text = args.prompt.strip() or _read_text(args.prompt_file).strip()
    if not prompt_text and not args.no_regen:
        raise SystemExit("Provide --prompt or --prompt-file (or use --no-regen).")

    # Paths
    validator = repo_root / "validate_solve_output.py"
    gen_dir = repo_root / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    gen_solver = gen_dir / "solve_module.py"

    agentlab_solver_dst = repo_root / "examples" / "agentlab_sort" / "solve_module.py"
    agentlab_solver_dst.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_regen:
        # Write prompt to temp file (so we don't fight shell quoting)
        prompt_file = gen_dir / "prompt.txt"
        prompt_file.write_text(prompt_text, encoding="utf-8")

        cmd = [
            sys.executable, str(agentlab_runner),
            "--user-prompt-file", str(prompt_file),
            "--models", args.models,
            "--out", str(gen_solver),
            "--validator", str(validator),
        ]
        if args.no_llm:
            cmd += ["--no-llm"]
        if args.custom_prompts:
            cmd += ["--custom-prompts", args.custom_prompts]

        print(f"[+] Generating solver via AgentLaboratory: {agentlab_runner}")
        res = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr, file=sys.stderr)
            raise SystemExit(f"AgentLaboratory solver generation failed (exit {res.returncode}).")

        # Copy into examples/agentlab_sort for universal_adapter import
        agentlab_solver_dst.write_text(gen_solver.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        if not agentlab_solver_dst.exists():
            raise SystemExit(f"--no-regen set, but solver not found: {agentlab_solver_dst}")

    # Validate solver quickly on a default vector
    print("[+] Validating solver on a smoke test...")
    v = "[3,1,2,5,4]"
    vcmd = [sys.executable, str(validator), "--solver", str(agentlab_solver_dst), "--vector", v]
    vres = subprocess.run(vcmd, cwd=str(repo_root), text=True, capture_output=True)
    print(vres.stdout)
    if vres.returncode != 0:
        print(vres.stderr, file=sys.stderr)
        raise SystemExit("Solver failed validation; refusing to build submission.")

    # Build submission
    print(f"[+] Building submission for {args.competition} ...")
    from examples.agentlab_sort.solver import solve_row  # local import
    build_submission(args.puzzles, args.out, args.competition, solve_row)
    print(f"[+] Saved: {args.out}")

    if args.submit:
        print("[+] Submitting via Kaggle API...")
        ensure_auth, submit_file, latest_scored_submission = _kaggle()
        api = ensure_auth()
        submit_file(api, args.competition, args.out, message=args.message)
        print("[+] Submitted.")

        if args.print_score:
            latest = latest_scored_submission(api, args.competition)
            if latest is None:
                print("[i] No scored submissions yet (Kaggle may still be evaluating).")
            else:
                ps = latest.get("public_score") or latest.get("publicScore")
                prs = latest.get("private_score") or latest.get("privateScore")
                st = latest.get("status")
                print(json.dumps({"status": st, "public_score": ps, "private_score": prs}, ensure_ascii=False))

if __name__ == "__main__":
    main()
