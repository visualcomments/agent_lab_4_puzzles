#!/usr/bin/env python3
"""
AgentLaboratory/perm_pipeline/run_perm_pipeline.py

3-agent loop (planner -> coder -> fixer) for generating a constructive solver.

Default backend: g4f models (GPT4Free). You can provide multiple models and the pipeline will try them in order
using a fallback syntax: g4f:model1|g4f:model2|...

This script produces a solver python file (default: ./generated/solve_module.py) and validates it using
the repo-level validate_solve_output.py (supports LRX or legacy ISK movesets).

Example:
  python AgentLaboratory/perm_pipeline/run_perm_pipeline.py \
      --user-prompt-file ./prompt.txt \
      --models "gpt-4o-mini,command-r,aria" \
      --out ./generated/solve_module.py
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import AgentLaboratory inference (patched to support g4f:)
THIS_DIR = Path(__file__).resolve().parent
AGENTLAB_ROOT = THIS_DIR.parent
sys.path.insert(0, str(AGENTLAB_ROOT))
from inference import query_model, MissingLLMCredentials  # type: ignore

RE_PY_BLOCK = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)

def load_prompts(custom_path: Optional[str]) -> Dict[str, str]:
    prompts_path = THIS_DIR / "default_prompts.json"
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    if custom_path:
        override = json.loads(Path(custom_path).read_text(encoding="utf-8"))
        prompts.update({k: v for k, v in override.items() if isinstance(v, str)})
    return prompts

def read_user_prompt(args) -> str:
    if args.user_prompt_file:
        return Path(args.user_prompt_file).read_text(encoding="utf-8")
    return args.user_prompt

def make_model_fallback(models: List[str]) -> str:
    return "|".join([f"g4f:{m.strip()}" for m in models if m.strip()])

def extract_python(resp: str) -> Optional[str]:
    m = RE_PY_BLOCK.search(resp or "")
    if not m:
        return None
    code = m.group(1).strip()
    return code if code else None

def run_validator(validator_path: Path, solver_path: Path, vec: List[int]) -> Tuple[int, str, str]:
    cmd = [sys.executable, str(validator_path), "--solver", str(solver_path), "--vector", json.dumps(vec)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user-prompt", default="", help="User prompt (inline string).")
    p.add_argument("--user-prompt-file", default=None, help="Path to a text file with the user prompt.")
    p.add_argument("--models", default=os.getenv("G4F_MODELS","").strip() or "gpt-4o-mini,command-r,aria",
                   help="Comma-separated g4f model names. Tried in order.")
    p.add_argument("--custom-prompts", default=None, help="Path to JSON overriding default system prompts.")
    p.add_argument("--out", default=str(Path.cwd() / "generated" / "solve_module.py"), help="Where to write the final solver.")
    p.add_argument("--max-iters", type=int, default=4, help="Max repair iterations.")
    p.add_argument("--no-llm", action="store_true", help="Skip LLM, write baseline solver directly.")
    p.add_argument("--validator", default=str(Path.cwd() / "validate_solve_output.py"),
                   help="Path to validate_solve_output.py (supports LRX/ISK simulation).")
    args = p.parse_args()

    user_prompt = read_user_prompt(args).strip()
    if not user_prompt:
        print("[!] Empty user prompt. Provide --user-prompt or --user-prompt-file.", file=sys.stderr)
        sys.exit(2)

    prompts = load_prompts(args.custom_prompts)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    model_fallback = make_model_fallback(models)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validator_path = Path(args.validator).resolve()

    # Baseline: if a "solve_module.py" exists in CWD, use it as a known-good fallback.
    baseline_path = Path.cwd() / "solve_module.py"
    if baseline_path.exists():
        baseline_code = baseline_path.read_text(encoding="utf-8")
    else:
        baseline_code = "# baseline solve_module.py not found in current working directory\n"

    if args.no_llm:
        out_path.write_text(baseline_code, encoding="utf-8")
        print(f"[+] Wrote baseline solver to {out_path}")
        sys.exit(0)

    # --- Agent 1: Planner ---
    try:
        plan = query_model(model_fallback, user_prompt, prompts["planner"])
        if not plan:
            plan = "(planner failed; proceeding)"
    except MissingLLMCredentials as e:
        # Make the CLI usable out-of-the-box even when g4f providers require auth.
        print(f"[!] {e}")
        print("[!] Falling back to the offline baseline solver. To enable LLM generation, set a provider API key or cookies/HAR for g4f, or pass --no-llm explicitly.")
        out_path.write_text(baseline_code, encoding="utf-8")
        print(f"[+] Wrote baseline solver to {out_path}")
        sys.exit(0)

    # --- Agent 2: Coder ---
    coder_prompt = f"USER TASK:\n{user_prompt}\n\nPLANNER NOTES:\n{plan}\n\nNow write the solver file."
    resp = query_model(model_fallback, coder_prompt, prompts["coder"])
    code = extract_python(resp or "") if resp else None
    if not code:
        code = baseline_code

    out_path.write_text(code, encoding="utf-8")

    # Minimal deterministic test set (distinct ints)
    tests = [
        [3,1,2,5,4],
        [1,2,3,4],
        [4,3,2,1],
        [2,0,3,1],
        [10,-1,7,3,5],
    ]

    for it in range(args.max_iters + 1):
        ok = True
        last_report = ""
        for vec in tests:
            rc, out, err = run_validator(validator_path, out_path, vec)
            if rc != 0:
                ok = False
                last_report = f"=== ITER {it}, VECTOR: {vec} ===\nSTDOUT:\n{out}\nSTDERR:\n{err}\n"
                break
        if ok:
            print(f"[+] Solver validated on {len(tests)} tests. Saved to {out_path}")
            sys.exit(0)

        if it >= args.max_iters:
            break

        # --- Agent 3: Fixer ---
        fix_prompt = f"""USER TASK:\n{user_prompt}\n\nCURRENT CODE:\n```python\n{out_path.read_text(encoding='utf-8')}\n```\n\nVALIDATOR REPORT:\n{last_report}\n\nReturn a corrected full python file."""
        resp = query_model(model_fallback, fix_prompt, prompts["fixer"])
        new_code = extract_python(resp or "") if resp else None
        if not new_code:
            print("[!] Fixer failed to return code. Stopping.", file=sys.stderr)
            break
        out_path.write_text(new_code, encoding="utf-8")

    print("[!] Failed to validate solver within max iterations.", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
