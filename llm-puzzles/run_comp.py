#!/usr/bin/env python3
import argparse
import importlib
import os
import subprocess

from src.universal_adapter import build_submission

def load_solver(dotted: str):
    if not dotted:
        dotted = "examples.llm_solver.solver:solve_row"
    parts = (dotted.split(":") + ["solve_row"])[:2]
    mod_name, func_name = parts[0], parts[1]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    return fn

def main():
    p = argparse.ArgumentParser(description="Universal Kaggle submission builder for CayleyPy & similar comps")
    p.add_argument("--competition", required=True, help="Kaggle competition slug")
    p.add_argument("--puzzles", required=True, help="Path to puzzles.csv (downloaded from Kaggle)")
    p.add_argument("--out", default="submission.csv", help="Output submission.csv path")
    p.add_argument("--solver", default="examples.llm_solver.solver:solve_row",
                   help="Dotted path to solver function (row, cfg) -> moves (str|list[str]|dict)")
    p.add_argument("--submit", action="store_true", help="Also submit via Kaggle API")
    p.add_argument("--message", default="perm-llm-starter auto-submit", help="Submission message")
    p.add_argument("--prefer-sample", action="store_true", help="Always take paths from sample_submission.csv")
    p.add_argument("--strict-llm", action="store_true", help="Only LLM solution (no sample fallback)")
    args = p.parse_args()

    os.environ["PUZZLES_CSV"] = args.puzzles
    if args.prefer_sample:
        os.environ["PREFER_SAMPLE"] = "1"
    if args.strict_llm:
        os.environ["STRICT_LLM"] = "1"

    print(f"[+] Building submission for {args.competition} using solver {args.solver}")
    solver_fn = load_solver(args.solver)
    build_submission(args.puzzles, args.out, args.competition, solver_fn)
    print(f"[+] Saved: {args.out}")

    if args.submit:
        print("[+] Submitting to Kaggle...")
        cmd = ["kaggle", "competitions", "submit", "-c", args.competition, "-f", args.out, "-m", args.message]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            print(res.stdout)
            if res.returncode != 0:
                print(res.stderr or "[!] Kaggle CLI returned non-zero exit code")
        except FileNotFoundError:
            print("[!] kaggle CLI not found. Please install it or submit manually.")

if __name__ == "__main__":
    main()
