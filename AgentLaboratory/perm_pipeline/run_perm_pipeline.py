#!/usr/bin/env python3
"""
AgentLaboratory/perm_pipeline/run_perm_pipeline.py

3-agent loop (planner -> coder -> fixer) for generating a constructive solver.

Default backend: g4f models (GPT4Free). You can provide multiple models and the
pipeline will probe/rank them for code-generation quality, then try them one by
one until a locally validated solver is produced.

Important safety/reliability behavior:
- The pipeline never returns unvalidated LLM code.
- If all model attempts fail, it falls back to the known-good offline baseline
  (unless --strict is used).
- Model probing checks for syntactically valid Python code blocks only; it does
  not execute arbitrary model-generated code.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is in requirements, this is just a safe fallback
    tqdm = None  # type: ignore

# Import AgentLaboratory inference (patched to support g4f:)
THIS_DIR = Path(__file__).resolve().parent
AGENTLAB_ROOT = THIS_DIR.parent
sys.path.insert(0, str(AGENTLAB_ROOT))
from inference import query_model, MissingLLMCredentials  # type: ignore

RE_PY_BLOCK = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
RE_ANY_BLOCK = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```", re.DOTALL)

DEFAULT_MODELS = os.getenv(
    "G4F_MODELS",
    "",
).strip() or "gpt-4o-mini,claude-3.5-sonnet,deepseek-chat,command-r-plus,command-r,aria"

MODEL_HINT_SCORES: Tuple[Tuple[str, int], ...] = (
    ("claude-3.7", 170),
    ("claude-3-7", 170),
    ("claude-sonnet-4", 168),
    ("claude-3.5-sonnet", 165),
    ("claude-3-5-sonnet", 165),
    ("gpt-4.1", 160),
    ("gpt-4o", 155),
    ("o3", 152),
    ("o1", 150),
    ("deepseek-r1", 148),
    ("deepseek-chat", 142),
    ("qwen2.5-coder", 140),
    ("qwen-2.5-coder", 140),
    ("qwq", 138),
    ("coder", 132),
    ("command-r-plus", 128),
    ("command-r+", 128),
    ("command-r", 120),
    ("qwen", 116),
    ("gemini", 110),
    ("llama", 100),
    ("aria", 70),
)


def load_prompts(custom_path: Optional[str]) -> Dict[str, str]:
    prompts_path = THIS_DIR / "default_prompts.json"
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    if custom_path:
        override = json.loads(Path(custom_path).read_text(encoding="utf-8"))
        prompts.update({k: v for k, v in override.items() if isinstance(v, str)})
    return prompts


def read_user_prompt(args: argparse.Namespace) -> str:
    if args.user_prompt_file:
        return Path(args.user_prompt_file).read_text(encoding="utf-8")
    return args.user_prompt


def normalize_model_name(model: str) -> str:
    s = (model or "").strip()
    if not s:
        return ""
    if ":" in s:
        return s
    return f"g4f:{s}"


def parse_models(raw: str) -> List[str]:
    items: List[str] = []
    seen = set()
    for part in (raw or "").split(","):
        m = normalize_model_name(part)
        if m and m not in seen:
            seen.add(m)
            items.append(m)
    return items


def model_quality_score(model: str) -> int:
    m = model.lower()
    score = 0
    for needle, value in MODEL_HINT_SCORES:
        if needle in m:
            score = max(score, value)
    if "mini" in m:
        score -= 6
    if "free" in m:
        score -= 2
    return score


def rank_models_for_codegen(models: Sequence[str]) -> List[str]:
    return sorted(models, key=lambda m: (-model_quality_score(m), m.lower()))


def extract_python(resp: str) -> Optional[str]:
    text = (resp or "").strip()
    if not text:
        return None

    m = RE_PY_BLOCK.search(text)
    if m:
        code = m.group(1).strip()
        return code or None

    m = RE_ANY_BLOCK.search(text)
    if m:
        code = m.group(1).strip()
        return code or None

    # Fallback: some models ignore the fence instruction and return raw Python.
    if any(token in text for token in ("def solve", "import ", "from __future__", "if __name__")):
        return text
    return None


def compile_python(code: str) -> Tuple[bool, str]:
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno}, offset {e.offset})"
    except Exception as e:  # pragma: no cover - defensive only
        return False, f"ParseError: {type(e).__name__}: {e}"
    return True, ""


def run_validator(validator_path: Path, solver_path: Path, vec: List[int]) -> Tuple[int, str, str]:
    cmd = [sys.executable, str(validator_path), "--solver", str(solver_path), "--vector", json.dumps(vec)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def validate_solver_suite(validator_path: Path, solver_path: Path, tests: Iterable[List[int]]) -> Tuple[bool, str]:
    for idx, vec in enumerate(tests):
        rc, out, err = run_validator(validator_path, solver_path, vec)
        if rc != 0:
            report = (
                f"=== TEST {idx} FAILED ===\n"
                f"VECTOR: {vec}\n"
                f"STDOUT:\n{out}\n"
                f"STDERR:\n{err}\n"
            )
            return False, report
    return True, ""


def probe_model_for_codegen(model: str) -> Tuple[bool, str]:
    prompt = (
        "Return only one ```python``` block that defines a function `solve(vec)` and returns the input unchanged. "
        "Do not add any explanation."
    )
    system = "You are checking whether you can follow strict code-only output requirements."
    try:
        resp = query_model(model, prompt, system, tries=1, timeout=12.0, print_cost=False)
    except MissingLLMCredentials:
        return False, "credentials required"
    except Exception as e:
        return False, str(e)

    code = extract_python(resp or "")
    if not code:
        return False, "no python block"
    ok, reason = compile_python(code)
    if not ok:
        return False, reason
    if "def solve" not in code:
        return False, "missing solve()"
    return True, "ok"


def order_models_for_codegen(models: Sequence[str]) -> List[str]:
    ranked = rank_models_for_codegen(models)
    if os.getenv("AGENTLAB_MODEL_PROBE", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return ranked

    try:
        probe_limit = int(os.getenv("AGENTLAB_MODEL_PROBE_TOP", "4") or "4")
    except Exception:
        probe_limit = 4

    if probe_limit <= 0:
        return ranked

    head = ranked[:probe_limit]
    tail = ranked[probe_limit:]
    good: List[str] = []
    bad: List[str] = []
    for model in head:
        ok, reason = probe_model_for_codegen(model)
        status = "OK" if ok else f"skip ({reason})"
        print(f"[model-probe] {model}: {status}")
        (good if ok else bad).append(model)
    return good + tail + bad


def ask_first_nonempty(models: Sequence[str], prompt: str, system_prompt: str) -> Tuple[str, Optional[str]]:
    last_error: Optional[Exception] = None
    for model in models:
        try:
            resp = query_model(model, prompt, system_prompt)
            if isinstance(resp, str) and resp.strip():
                return resp.strip(), model
        except MissingLLMCredentials as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue
    if last_error is not None:
        raise last_error
    return "", None


def make_baseline_stub() -> str:
    return """from __future__ import annotations
import json
import sys


def solve(vec):
    return \"UNSOLVED\", list(vec)


if __name__ == \"__main__\":
    vec = json.loads(sys.argv[1])
    moves, sorted_array = solve(vec)
    print(json.dumps({\"moves\": moves, \"sorted_array\": sorted_array}))
"""


def _make_iteration_progress(model: str, max_iters: int):
    if max_iters <= 0 or tqdm is None:
        return None
    return tqdm(
        total=max_iters,
        desc=f"fix {model}",
        unit="iter",
        dynamic_ncols=True,
        leave=True,
        file=sys.stderr,
    )


def try_generate_with_model(
    *,
    model: str,
    user_prompt: str,
    plan: str,
    prompts: Dict[str, str],
    out_path: Path,
    validator_path: Path,
    tests: Sequence[List[int]],
    max_iters: int,
) -> Tuple[bool, str]:
    coder_prompt = f"USER TASK:\n{user_prompt}\n\nPLANNER NOTES:\n{plan}\n\nNow write the solver file."

    try:
        resp = query_model(model, coder_prompt, prompts["coder"])
    except MissingLLMCredentials as e:
        return False, f"{model}: credentials required ({e})"
    except Exception as e:
        return False, f"{model}: coder failed ({e})"

    code = extract_python(resp or "")
    if not code:
        return False, f"{model}: coder did not return a python file"

    ok, compile_err = compile_python(code)
    if not ok:
        last_report = f"Initial compile check failed.\n{compile_err}\n"
    else:
        out_path.write_text(code, encoding="utf-8")
        valid, last_report = validate_solver_suite(validator_path, out_path, tests)
        if valid:
            return True, f"{model}: coder output validated immediately"

    current_code = code
    progress = _make_iteration_progress(model, max_iters)
    if progress is not None:
        progress.set_postfix_str(f"iter 0/{max_iters}")

    try:
        for it in range(1, max_iters + 1):
            if progress is not None:
                progress.set_postfix_str(f"iter {it}/{max_iters}")

            fix_prompt = (
                f"USER TASK:\n{user_prompt}\n\n"
                f"CURRENT CODE:\n```python\n{current_code}\n```\n\n"
                f"FAILURE REPORT:\n{last_report}\n\n"
                "Return a corrected full python file."
            )
            try:
                resp = query_model(model, fix_prompt, prompts["fixer"])
            except MissingLLMCredentials as e:
                return False, f"{model}: fixer credentials required ({e})"
            except Exception as e:
                return False, f"{model}: fixer failed ({e})"

            new_code = extract_python(resp or "")
            if not new_code:
                return False, f"{model}: fixer iteration {it} returned no python file"

            ok, compile_err = compile_python(new_code)
            current_code = new_code
            if progress is not None:
                progress.update(1)

            if not ok:
                last_report = f"Fix iteration {it} compile check failed.\n{compile_err}\n"
                continue

            out_path.write_text(current_code, encoding="utf-8")
            valid, last_report = validate_solver_suite(validator_path, out_path, tests)
            if valid:
                return True, f"{model}: validated after fixer iteration {it}"
    finally:
        if progress is not None:
            progress.close()

    return False, f"{model}: failed validation after {max_iters} fixer iterations\n{last_report}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user-prompt", default="", help="User prompt (inline string).")
    p.add_argument("--user-prompt-file", default=None, help="Path to a text file with the user prompt.")
    p.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=(
            "Comma-separated model list. Bare names use g4f backend (remote providers). "
            "You can also pass explicit backends like local:<hf_model_id> to run Transformers locally (CUDA-supported)."
        ),
    )
    p.add_argument("--custom-prompts", default=None, help="Path to JSON overriding default system prompts.")
    p.add_argument("--out", default=str(Path.cwd() / "generated" / "solve_module.py"), help="Where to write the final solver.")
    p.add_argument("--max-iters", type=int, default=4, help="Max repair iterations per model candidate.")
    p.add_argument("--no-llm", action="store_true", help="Skip LLM, write baseline solver directly.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail with non-zero exit code if LLM generation/repair does not validate. "
             "By default, the pipeline falls back to the offline baseline solver and exits 0.",
    )
    p.add_argument("--validator", default=str(Path.cwd() / "validate_solve_output.py"),
                   help="Path to validate_solve_output.py (supports LRX/ISK simulation).")
    p.add_argument("--baseline", default=None,
                   help="Path to baseline solve_module.py used for --no-llm and fallback. Default: ./solve_module.py in current working directory.")
    args = p.parse_args()

    user_prompt = read_user_prompt(args).strip()
    if not user_prompt:
        print("[!] Empty user prompt. Provide --user-prompt or --user-prompt-file.", file=sys.stderr)
        sys.exit(2)

    prompts = load_prompts(args.custom_prompts)
    models = parse_models(args.models)
    if not models and not args.no_llm:
        print("[!] No models configured. Pass --models or set G4F_MODELS.", file=sys.stderr)
        sys.exit(2)
    ordered_models = order_models_for_codegen(models)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validator_path = Path(args.validator).resolve()

    baseline_path = Path(args.baseline) if args.baseline else (Path.cwd() / "solve_module.py")
    if baseline_path.exists():
        baseline_code = baseline_path.read_text(encoding="utf-8")
    else:
        baseline_code = make_baseline_stub()

    if args.no_llm:
        out_path.write_text(baseline_code, encoding="utf-8")
        print(f"[+] Wrote baseline solver to {out_path}")
        sys.exit(0)

    def _fallback_to_baseline(reason: str) -> None:
        print(f"[!] {reason}", file=sys.stderr)
        if args.strict:
            sys.exit(1)
        out_path.write_text(baseline_code, encoding="utf-8")
        print("[!] Falling back to the offline baseline solver.", file=sys.stderr)
        print(f"[+] Wrote baseline solver to {out_path}")
        sys.exit(0)

    try:
        plan, planner_model = ask_first_nonempty(ordered_models, user_prompt, prompts["planner"])
        if not plan:
            plan = "(planner failed; proceeding without planner notes)"
        print(f"[planner] selected model: {planner_model or 'none'}")
    except MissingLLMCredentials as e:
        _fallback_to_baseline(
            "g4f provider requires credentials (api_key or .har). "
            "Set OPENROUTER_API_KEY / OPENAI_API_KEY (or other provider key), or place a .har/.json in ./har_and_cookies, "
            f"or run with --no-llm. Original error: {e}"
        )
    except Exception as e:
        _fallback_to_baseline(f"Planner failed (LLM error): {e}")

    tests: List[List[int]] = [
        [3, 1, 2, 5, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [2, 0, 3, 1],
        [10, -1, 7, 3, 5],
    ]

    for model in ordered_models:
        print(f"[coder] trying model: {model}")
        ok, report = try_generate_with_model(
            model=model,
            user_prompt=user_prompt,
            plan=plan,
            prompts=prompts,
            out_path=out_path,
            validator_path=validator_path,
            tests=tests,
            max_iters=args.max_iters,
        )
        if ok:
            print(f"[+] {report}. Saved to {out_path}")
            sys.exit(0)
        print(f"[coder] {report}")

    _fallback_to_baseline("Failed to generate a locally validated solver with the configured models.")


if __name__ == "__main__":
    main()
