#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple


class ValidationError(Exception):
    pass


def load_solver_module(path: str):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Solver file not found: {path}")
    spec = importlib.util.spec_from_file_location("solve_module_dynamic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import solver module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "solve"):
        raise RuntimeError("Solver module must define solve(vector)")
    return module


def parse_vector(s: str) -> List[int]:
    try:
        v = json.loads(s)
    except Exception as e:
        raise ValidationError(f"Failed to parse --vector JSON: {e}")
    if not isinstance(v, list) or not all(isinstance(x, int) for x in v):
        raise ValidationError("--vector must be a JSON list of integers")
    return v


def is_strictly_ascending(a: List[int]) -> bool:
    return all(a[i] < a[i + 1] for i in range(len(a) - 1))


# ---------------------------
# Competition: cayleypy-rapapport-m2
# Moves I/S/K from rapaport_m2.txt [7]
# I: swap 1st and 2nd
# S: swap (1st,2nd), (3rd,4th), ...
# K: swap (2nd,3rd), (4th,5th), ...
# ---------------------------

def apply_move_isk(a: List[int], mv: str) -> None:
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
    raise ValidationError(f"Unknown move {mv!r}. Allowed moves: I, S, K")


def simulate_isk(vec: List[int], moves: List[str]) -> List[int]:
    a = list(vec)
    for mv in moves:
        if not isinstance(mv, str):
            raise ValidationError(f"Move must be string, got {type(mv)}")
        apply_move_isk(a, mv)
    return a


def validate_rapaport_m2(vec: List[int], moves: List[str], sorted_array: List[int]) -> Dict[str, Any]:
    # per prompt: distinct integers [7]
    if len(set(vec)) != len(vec):
        raise ValidationError("Input vector must contain distinct integers")

    sim = simulate_isk(vec, moves)
    if sim != sorted_array:
        raise ValidationError(
            "sorted_array does not match simulation.\n"
            f"simulated={sim}\nreported ={sorted_array}"
        )
    if not is_strictly_ascending(sorted_array):
        raise ValidationError(f"Result not strictly ascending: {sorted_array}")

    return {"ok": True, "n_moves": len(moves)}


VALIDATORS: Dict[str, Callable[[List[int], List[str], List[int]], Dict[str, Any]]] = {
    "cayleypy-rapapport-m2": validate_rapaport_m2,
}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Validate solver output by competition slug.")
    p.add_argument("--competition", required=True, help="Competition slug, e.g. cayleypy-rapapport-m2")
    p.add_argument("--solver", required=True, help="Path to solve_module.py")
    p.add_argument("--vector", required=True, help='Vector JSON, e.g. "[3,1,2,5,4]"')
    args = p.parse_args(argv)

    slug = args.competition.strip()
    if slug not in VALIDATORS:
        raise SystemExit(f"Unknown competition slug {slug!r}. Supported: {sorted(VALIDATORS.keys())}")

    vec = parse_vector(args.vector)
    mod = load_solver_module(args.solver)

    moves, sorted_array = mod.solve(list(vec))

    if not isinstance(moves, list) or not all(isinstance(m, str) for m in moves):
        raise SystemExit("solve() must return moves as list[str]")
    if not isinstance(sorted_array, list) or not all(isinstance(x, int) for x in sorted_array):
        raise SystemExit("solve() must return sorted_array as list[int]")

    try:
        report = VALIDATORS[slug](vec, moves, sorted_array)
        print(json.dumps(report, ensure_ascii=False))
    except ValidationError as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()