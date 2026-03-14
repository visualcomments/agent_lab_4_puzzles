#!/usr/bin/env python3
import argparse, os
from importlib import import_module
from src.santa_adapter import solve_dataset
from src.kaggle_utils import ensure_auth, submit_file

def main():
    p = argparse.ArgumentParser(description="Santa 2023 submission builder & submitter")
    p.add_argument("--puzzles", required=True, help="Path to puzzles.csv from Kaggle competition")
    p.add_argument("--out", default="submission.csv", help="Output CSV path (submission file)")
    p.add_argument("--solver", default="examples.santa_solver.example_solver:solve_row",
                   help="Dotted path to function row->moves, e.g. pkg.module:func")
    p.add_argument("--submit", action="store_true", help="If set, submit via Kaggle API")
    p.add_argument("--message", default="perm-llm-starter auto-submit", help="Submission message")
    p.add_argument("--competition", default="santa-2023", help="Kaggle competition slug")
    args = p.parse_args()

    # import solver
    if ":" in args.solver:
        mod_name, func_name = args.solver.split(":", 1)
    else:
        mod_name, func_name = args.solver, "solve_row"
    mod = import_module(mod_name)
    solver_fn = getattr(mod, func_name)

    print(f"Building submission: {args.out} from {args.puzzles} using {args.solver}")
    solve_dataset(args.puzzles, args.out, solver_fn)
    print(f"Saved submission file to {args.out}")

    if args.submit:
        print("Authenticating Kaggle API...")
        api = ensure_auth()
        print(f"Submitting to {args.competition} with message: {args.message}")
        submit_file(api, args.competition, args.out, args.message)
        print("Submission requested. Check Kaggle submissions page for status.")

if __name__ == "__main__":
    main()
