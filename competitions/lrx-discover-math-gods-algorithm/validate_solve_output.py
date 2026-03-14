#!/usr/bin/env python3
"""
validate_solve_output.py  (enhanced diagnostics + JSON/Markdown reporting + MOVESET simulation)

Validates JSON output produced by a solver script (default: solve_module.py) for a given input vector.

Checks performed:
0) Solver exit code is 0.
1) stdout parses as JSON object (dict).
2) Keys "moves" and "sorted_array" exist.
3) Both are lists.
4) moves contains ONLY allowed moves for one detected moveset:
    - LRX: {"L","R","X"}  (left rotate, right rotate, swap first two)
    - ISK: {"I","S","K"}  (swap first two, swap even pairs, swap odd pairs)  [legacy]
5) Applying the moves sequentially to a copy of the input vector yields exactly "sorted_array".
6) "sorted_array" is sorted non-decreasing.
7) Multiset of "sorted_array" equals multiset of the input.

Usage:
    python validate_solve_output.py --solver ./solve_module.py --vector "[3,1,2,5,4]" \
        --report-json ./report.json --report-md ./report.md
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone
import platform


MOVESETS: Dict[str, set[str]] = {
    "LRX": {"L", "R", "X"},
    "ISK": {"I", "S", "K"},
}


def run_solver(solver_path: str, input_vector: List[Any]) -> Tuple[int, str, str]:
    cmd = [sys.executable, solver_path, json.dumps(input_vector, separators=(',', ':'))]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def detect_moveset(moves: List[Any]) -> Optional[str]:
    ms = set(m for m in moves if isinstance(m, str))
    for name, allowed in MOVESETS.items():
        if ms.issubset(allowed):
            return name
    return None


def apply_move_L(a: List[Any]) -> None:
    n = len(a)
    if n <= 1:
        return
    first = a[0]
    for i in range(n - 1):
        a[i] = a[i + 1]
    a[n - 1] = first


def apply_move_R(a: List[Any]) -> None:
    n = len(a)
    if n <= 1:
        return
    last = a[n - 1]
    for i in range(n - 1, 0, -1):
        a[i] = a[i - 1]
    a[0] = last


def apply_move_X(a: List[Any]) -> None:
    if len(a) >= 2:
        a[0], a[1] = a[1], a[0]


def apply_move_I(a: List[Any]) -> None:
    # swap first two
    apply_move_X(a)


def apply_move_S(a: List[Any]) -> None:
    # swap pairs (0,1),(2,3)...
    for i in range(0, len(a) - 1, 2):
        a[i], a[i + 1] = a[i + 1], a[i]


def apply_move_K(a: List[Any]) -> None:
    # swap pairs (1,2),(3,4)...
    for i in range(1, len(a) - 1, 2):
        a[i], a[i + 1] = a[i + 1], a[i]


def simulate(input_vector: List[Any], moves: List[str], moveset: str) -> List[Any]:
    a = list(input_vector)
    for m in moves:
        if moveset == "LRX":
            if m == "L":
                apply_move_L(a)
            elif m == "R":
                apply_move_R(a)
            elif m == "X":
                apply_move_X(a)
            else:
                raise ValueError(f"Unknown move {m} for moveset {moveset}")
        elif moveset == "ISK":
            if m == "I":
                apply_move_I(a)
            elif m == "S":
                apply_move_S(a)
            elif m == "K":
                apply_move_K(a)
            else:
                raise ValueError(f"Unknown move {m} for moveset {moveset}")
        else:
            raise ValueError(f"Unsupported moveset {moveset}")
    return a


def find_first_unsorted_index(a: List[Any]) -> int:
    for i in range(len(a) - 1):
        try:
            if a[i] > a[i + 1]:
                return i
        except TypeError:
            return i
    return -1


def validate_json_output(input_vector: List[Any], stdout_text: str) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    # 1) stdout parses as JSON object
    try:
        obj = json.loads(stdout_text.strip())
        is_dict = isinstance(obj, dict)
        checks.append({
            'ok': is_dict,
            'msg': '1) Output must be a JSON object (dict).',
            'details': {} if is_dict else {'actual_type': type(obj).__name__}
        })
        if not is_dict:
            return False, checks, {}
    except json.JSONDecodeError as e:
        checks.append({
            'ok': False,
            'msg': '1) Output is not valid JSON.',
            'details': {
                'parse_error': str(e),
                'raw_stdout_preview': stdout_text[:500]
            }
        })
        return False, checks, {}

    # 2) keys
    has_keys = ("moves" in obj) and ("sorted_array" in obj)
    checks.append({
        'ok': has_keys,
        'msg': '2) Object must contain keys "moves" and "sorted_array".',
        'details': {} if has_keys else {'present_keys': list(obj.keys())[:50]}
    })
    if not has_keys:
        return False, checks, obj

    moves = obj.get("moves")
    sorted_array = obj.get("sorted_array")

    # 3) both lists
    moves_is_list = isinstance(moves, list)
    sa_is_list = isinstance(sorted_array, list)
    checks.append({
        'ok': moves_is_list,
        'msg': '3) "moves" must be a list.',
        'details': {} if moves_is_list else {'actual_type': type(moves).__name__}
    })
    checks.append({
        'ok': sa_is_list,
        'msg': '3) "sorted_array" must be a list.',
        'details': {} if sa_is_list else {'actual_type': type(sorted_array).__name__}
    })
    if not (moves_is_list and sa_is_list):
        return False, checks, obj

    # 4) detect moveset + validate moves
    moveset = detect_moveset(moves)
    if moveset is None:
        checks.append({
            'ok': False,
            'msg': '4) "moves" must be only from a supported moveset (LRX or ISK).',
            'details': {'supported_movesets': {k: sorted(list(v)) for k, v in MOVESETS.items()}}
        })
        return False, checks, obj

    allowed = MOVESETS[moveset]
    bad = None
    for i, m in enumerate(moves):
        if not (isinstance(m, str) and m in allowed):
            bad = {'index': i, 'value': m}
            break
    checks.append({
        'ok': bad is None,
        'msg': f'4) "moves" may only contain moves in moveset {moveset}: {sorted(list(allowed))}.',
        'details': {} if bad is None else {'first_offending_move': bad, 'moveset': moveset}
    })
    if bad is not None:
        return False, checks, obj

    # 5) simulate and compare to sorted_array
    try:
        sim = simulate(input_vector, moves, moveset)
        ok_sim = (sim == sorted_array)
        details = {} if ok_sim else {
            'moveset': moveset,
            'simulated_head': sim[:20],
            'reported_head': sorted_array[:20],
            'note': 'The solver must append a move immediately after performing it; applying moves must reproduce sorted_array.'
        }
    except Exception as e:
        ok_sim = False
        details = {'moveset': moveset, 'simulation_error': str(e)}
    checks.append({
        'ok': ok_sim,
        'msg': '5) Applying "moves" to the input must yield exactly "sorted_array".',
        'details': details
    })

    # 6) sorted order
    first_unsorted = find_first_unsorted_index(sorted_array)
    ok_sorted = (first_unsorted == -1)
    checks.append({
        'ok': ok_sorted,
        'msg': '6) "sorted_array" must be sorted in non-decreasing order.',
        'details': {} if ok_sorted else {'first_unsorted_index': first_unsorted, 'pair': [sorted_array[first_unsorted], sorted_array[first_unsorted+1]]}
    })

    # 7) multiset equality
    try:
        ok_multiset = Counter(input_vector) == Counter(sorted_array)
        details = {} if ok_multiset else {'input_counts': dict(Counter(input_vector)), 'output_counts': dict(Counter(sorted_array))}
    except TypeError as e:
        ok_multiset = False
        details = {'type_error': str(e)}
    checks.append({
        'ok': ok_multiset,
        'msg': '7) Value counts (multiset) must match between input and "sorted_array".',
        'details': details
    })

    all_passed = all(c['ok'] for c in checks)
    obj['_detected_moveset'] = moveset
    return all_passed, checks, obj


def make_json_report(args, rc, out, err, input_vector, all_passed, checks, parsed) -> Dict[str, Any]:
    # Use timezone-aware UTC datetime (utcnow() is deprecated in recent Python).
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "meta": {
            "timestamp_utc": now,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "solver_path": args.solver,
        },
        "input": {"vector": input_vector},
        "run": {"exit_code": rc},
        "stdout_preview": (out or "")[: args.max_preview],
        "stderr_preview": (err or "")[: args.max_preview],
        "results": {
            "check_0": {"ok": rc == 0, "msg": "0) Solver exited with code 0.", "details": {"actual_exit_code": rc}},
            "checks": checks,
            "overall_pass": bool((rc == 0) and all_passed),
        },
        "parsed_json": parsed if isinstance(parsed, (dict, list)) else None
    }


def make_markdown_report(json_report: Dict[str, Any]) -> str:
    meta = json_report.get("meta", {})
    vec = json_report.get("input", {}).get("vector", [])
    rc = json_report.get("run", {}).get("exit_code", None)
    checks = json_report.get("results", {}).get("checks", [])
    overall = "PASS ✅" if json_report.get("results", {}).get("overall_pass") else "FAIL ❌"

    lines = [
        "# Validation Report",
        "## Meta",
        f"- Timestamp (UTC): `{meta.get('timestamp_utc','')}`",
        f"- Platform: `{meta.get('platform','')}`",
        f"- Python: `{meta.get('python_version','')}`",
        f"- Solver: `{meta.get('solver_path','')}`",
        "",
        "## Input",
        f"- Vector: `{vec}`",
        "",
        "## Summary",
        f"- Overall: **{overall}**",
        f"- Exit code: `{rc}`",
        "",
        "## Checks",
        "| # | Description | Status |",
        "|---|-------------|--------|",
    ]

    # check numbering: 0 is exit code, then checks list is 1..N
    lines.append(f"| 0 | Solver exited with code 0 | {'PASS' if rc == 0 else 'FAIL'} |")
    for i, c in enumerate(checks, start=1):
        status = "PASS" if c.get("ok") else "FAIL"
        lines.append(f"| {i} | {c.get('msg','')} | {status} |")

    # failed details
    failed = [c for c in checks if not c.get("ok")]
    lines.append("")
    lines.append("## Detailed Analytics")
    if rc != 0:
        lines.append(f"- Exit code != 0: `{rc}`")
    if not failed:
        lines.append("All checks passed.")
    else:
        for c in failed:
            lines.append("")
            lines.append(f"### {c.get('msg','')}")
            details = c.get("details") or {}
            if details:
                lines.append("```json")
                lines.append(json.dumps(details, ensure_ascii=False, indent=2))
                lines.append("```")
            else:
                lines.append("_No additional details._")

    # raw previews
    stdout_prev = json_report.get("stdout_preview","")
    stderr_prev = json_report.get("stderr_preview","")
    lines.append("")
    lines.append("## Raw Output Previews")
    if stdout_prev:
        lines += ["**STDOUT (preview):**", "```", stdout_prev, "```"]
    if stderr_prev:
        lines += ["**STDERR (preview):**", "```", stderr_prev, "```"]

    parsed_json = json_report.get("parsed_json", None)
    lines.append("")
    lines.append("## Parsed JSON Preview")
    if parsed_json is not None:
        lines.append("```json")
        lines.append(json.dumps(parsed_json, ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append("_None or not available._")

    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Validate JSON output of solve_module.py")
    p.add_argument("--solver", required=True, help="Path to solve_module.py")
    p.add_argument("--vector", default="[3,1,2,5,4]", help="JSON array to pass to the solver")
    p.add_argument("--show-stdout", action="store_true", help="Print solver stdout before validation")
    p.add_argument("--show-stderr", action="store_true", help="Print solver stderr if any")
    p.add_argument("--report-json", default=None, help="Path to write a JSON report")
    p.add_argument("--report-md", default=None, help="Path to write a Markdown report")
    p.add_argument("--max-preview", type=int, default=2000, help="Max chars in stdout/stderr previews")
    args = p.parse_args()

    try:
        input_vector = json.loads(args.vector)
        if not isinstance(input_vector, list):
            raise ValueError("vector must be JSON array")
    except Exception as e:
        print(f"Error: invalid --vector. {e}", file=sys.stderr)
        sys.exit(1)

    rc, out, err = run_solver(args.solver, input_vector)
    if args.show_stdout:
        print("=== Solver stdout ===")
        print(out.rstrip())
        print("=====================")
    if args.show_stderr and err:
        print("=== Solver stderr ===", file=sys.stderr)
        print(err.rstrip(), file=sys.stderr)
        print("=====================", file=sys.stderr)

    all_passed, checks, parsed = validate_json_output(input_vector, out)
    report = make_json_report(args, rc, out, err, input_vector, all_passed, checks, parsed)

    print("Validation results:")
    print(f" - [{'PASS' if rc == 0 else 'FAIL'}] 0) Solver exited with code 0 (actual: {rc}).")
    for c in checks:
        print(f" - [{'PASS' if c['ok'] else 'FAIL'}] {c['msg']}")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"JSON report written to: {args.report_json}")
    if args.report_md:
        with open(args.report_md, "w", encoding="utf-8") as f:
            f.write(make_markdown_report(report))
        print(f"Markdown report written to: {args.report_md}")

    sys.exit(0 if report["results"]["overall_pass"] else 1)


if __name__ == "__main__":
    main()
