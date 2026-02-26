#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import json
import os
import sys
from typing import List, Tuple, Dict, Any, Optional


# ---------------------------
# Competition validator registry
# ---------------------------

class ValidationError(Exception):
    pass


def is_sorted_strict_ascending(a: List[int]) -> bool:
    return all(a[i] < a[i + 1] for i in range(len(a) - 1))


# ---- rapaport_m2 validator (I/S/K) ----
# Moves definition from prompt [7]:
# I: swap 1st and 2nd (0 and 1)
# S: swap (0,1), (2,3), (4,5)...
# K: swap (1,2), (3,4), (5,6)...
def apply_move_rapaport_m2(a: List[int], mv: str) -> None:
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
    raise ValidationError(f"Unknown move: {mv!r}. Allowed moves: I, S, K")


def simulate_rapaport_m2(vector: List[int], moves: List[str]) -> List[int]:
    a = list(vector)
    for mv in moves:
        if not isinstance(mv, str):
            raise ValidationError(f"Move must be string, got: {type(mv)}")
        apply_move_rapaport_m2(a, mv)
    return a


def validate_rapaport_m2(vector: List[int], moves: List[str], sorted_array: List[int]) -> Dict[str, Any]:
    # distinct integers constraint mentioned in prompt [7] (we enforce lightly)
    if len(set(vector)) != len(vector):
        raise ValidationError("Input vector must contain distinct integers")

    simulated = simulate_rapaport_m2(vector, moves)

    if simulated != sorted_array:
        raise ValidationError(
            "sorted_array does not match simulation result.\n"
            f"simulated={simulated}\nreported ={sorted_array}"
        )
    if not is_sorted_strict_ascending(sorted_array):
        raise ValidationError(f"Result is not strictly ascending: {sorted_array}")

    return {
        "ok": True,
        "n_moves": len(moves),
        "final": sorted_array,
    }


VALIDATORS = {
    "cayleypy-rapapport-m2": validate_rapaport_m2,
}


def load_solver_module(path: str):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("solve_module_dynamic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import solver from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_vector(s: str) -> List[int]:
    try:
        v = json.loads(s)
    except Exception as e:
        raise ValidationError(f"Failed to parse vector JSON: {e}")
    if not isinstance(v, list):
        raise ValidationError("Vector must be a JSON list")
    if not all(isinstance(x, int) for x in v):
        raise ValidationError("Vector must contain integers only")
    return v


def parse_solution_json(s: str) -> Tuple[List[str], List[int]]:
    try:
        obj = json.loads(s)
    except Exception as e:
        raise ValidationError(f"Failed to parse solution JSON: {e}")
    if not isinstance(obj, dict):
        raise ValidationError("Solution must be a JSON object with keys moves, sorted_array")

    moves = obj.get("moves", None)
    sorted_array = obj.get("sorted_array", None)
    if not isinstance(moves, list):
        raise ValidationError("Solution field 'moves' must be a list")
    if not isinstance(sorted_array, list) or not all(isinstance(x, int) for x in sorted_array):
        raise ValidationError("Solution field 'sorted_array' must be a list[int]")
    return moves, sorted_array


def main():
    ap = argparse.ArgumentParser(description="Validate solver output for selected competition slug.")
    ap.add_argument("--competition", required=True, help="Kaggle competition slug, e.g. cayleypy-rapapport-m2")
    ap.add_argument("--vector", default="[3,1,2,5,4]", help="Input vector as JSON list")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--solver", help="Path to solve_module.py that provides solve(vector)->(moves, sorted_array)")
    g.add_argument("--solution-json", help="JSON object string with keys moves, sorted_array")
    args = ap.parse_args()

    if args.competition not in VALIDATORS:
        raise SystemExit(
            "Unknown competition slug. Supported:\n"
            + "\n".join(sorted(VALIDATORS.keys()))
        )
    validator_fn = VALIDATORS[args.competition]

    vector = parse_vector(args.vector)

    if args.solver:
        mod = load_solver_module(args.solver)
        if not hasattr(mod, "solve"):
            raise SystemExit("Solver module must define solve(vector)")

        moves, sorted_array = mod.solve(list(vector))
        # normalize / basic checks
        if not isinstance(moves, list) or not all(isinstance(x, str) for x in moves):
            raise SystemExit("solve() must return moves as list[str]")
        if not isinstance(sorted_array, list) or not all(isinstance(x, int) for x in sorted_array):
            raise SystemExit("solve() must return sorted_array as list[int]")
    else:
        moves, sorted_array = parse_solution_json(args.solution_json)

    try:
        report = validator_fn(vector, moves, sorted_array)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except ValidationError as e:
        err = {"ok": False, "error": str(e)}
        print(json.dumps(err, ensure_ascii=False, indent=2))
        sys.exit(2)


if __name__ == "__main__":
    main()