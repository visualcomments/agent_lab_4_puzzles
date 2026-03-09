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
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is in requirements, this is just a safe fallback
    tqdm = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None  # type: ignore

# Import AgentLaboratory inference (patched to support g4f:)
THIS_DIR = Path(__file__).resolve().parent
AGENTLAB_ROOT = THIS_DIR.parent
sys.path.insert(0, str(AGENTLAB_ROOT))
from inference import query_model, MissingLLMCredentials, _best_effort_release_memory  # type: ignore

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
    for part in (raw or "").replace("|", ",").split(","):
        m = normalize_model_name(part)
        if m and m not in seen:
            seen.add(m)
            items.append(m)
    return items


def parse_agent_model_overrides(raw: Optional[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for chunk in re.split(r"[;\n]+", raw or ""):
        entry = chunk.strip()
        if not entry or "=" not in entry:
            continue
        role, model_list = entry.split("=", 1)
        role_key = role.strip().lower().replace("_", "-")
        parsed = parse_models(model_list)
        if parsed:
            mapping[role_key] = parsed
    return mapping


def apply_agent_model_override(mapping: Dict[str, List[str]], role: str, raw_models: Optional[str]) -> None:
    parsed = parse_models(raw_models or "")
    if parsed:
        mapping[role.strip().lower().replace("_", "-")] = parsed


def resolve_agent_models(role: str, fallback: Sequence[str], overrides: Dict[str, List[str]]) -> List[str]:
    key = role.strip().lower().replace("_", "-")
    resolved = overrides.get(key)
    if resolved:
        return list(resolved)
    return list(fallback)


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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or str(default))
    except Exception:
        return default


def _is_remote_model(model: str) -> bool:
    return not (model or '').strip().startswith('local:')


def _use_remote_subprocess_isolation(model: str) -> bool:
    if not _is_remote_model(model):
        return False
    return not ((os.getenv('AGENTLAB_REMOTE_SUBPROCESS', '1') or '').strip().lower() in {'0', 'false', 'no', 'off'})


def _current_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None



def _system_total_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return float(psutil.virtual_memory().total) / (1024.0 * 1024.0)
    except Exception:
        return None



def _default_max_rss_mb() -> int:
    explicit = os.getenv('AGENTLAB_MAX_RSS_MB')
    if explicit not in {None, ''}:
        return _env_int('AGENTLAB_MAX_RSS_MB', 0)
    total_mb = _system_total_mb()
    if total_mb is None:
        return 0
    if _is_colab_env():
        return max(1024, int(total_mb * 0.72))
    return 0



def _memory_limit_exceeded() -> Tuple[bool, Optional[float], int]:
    limit_mb = _default_max_rss_mb()
    if limit_mb <= 0:
        return False, _current_rss_mb(), limit_mb
    rss_mb = _current_rss_mb()
    if rss_mb is None:
        return False, None, limit_mb
    return rss_mb >= float(limit_mb), rss_mb, limit_mb


def _query_model_stable(
    model: str,
    prompt: str,
    system_prompt: str,
    *,
    tries: int = 5,
    timeout: float = 20.0,
    temp: Optional[float] = None,
    print_cost: bool = False,
    version: str = '1.5',
) -> str:
    if not _use_remote_subprocess_isolation(model):
        return query_model(model, prompt, system_prompt, tries=tries, timeout=timeout, temp=temp, print_cost=print_cost, version=version)

    worker_path = THIS_DIR / 'query_model_worker.py'
    if not worker_path.exists():
        return query_model(model, prompt, system_prompt, tries=tries, timeout=timeout, temp=temp, print_cost=print_cost, version=version)

    with tempfile.TemporaryDirectory(prefix='agentlab_query_') as tmpdir:
        tmpdir_path = Path(tmpdir)
        prompt_file = tmpdir_path / 'prompt.txt'
        system_file = tmpdir_path / 'system.txt'
        out_json = tmpdir_path / 'result.json'
        prompt_file.write_text(prompt, encoding='utf-8')
        system_file.write_text(system_prompt, encoding='utf-8')

        cmd = [
            sys.executable,
            str(worker_path),
            '--model',
            model,
            '--prompt-file',
            str(prompt_file),
            '--system-file',
            str(system_file),
            '--out-json',
            str(out_json),
            '--tries',
            str(int(tries)),
            '--timeout',
            str(float(timeout)),
            '--version',
            str(version),
        ]
        if print_cost:
            cmd.append('--print-cost')
        if temp is not None:
            cmd.extend(['--temp', str(temp)])

        env = dict(os.environ)
        env['AGENTLAB_REMOTE_SUBPROCESS'] = '0'
        proc_timeout = max(30, int(float(timeout)) + 15)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=proc_timeout)

        if not out_json.exists():
            stderr = (proc.stderr or proc.stdout or '').strip()
            raise RuntimeError(f'{model}: remote worker did not produce a result file. {stderr}'.strip())

        try:
            payload = json.loads(out_json.read_text(encoding='utf-8'))
        except Exception as e:
            stderr = (proc.stderr or proc.stdout or '').strip()
            raise RuntimeError(f'{model}: failed to parse remote worker output ({e}). {stderr}'.strip()) from e

        if payload.get('ok'):
            answer = payload.get('answer', '')
            return answer if isinstance(answer, str) else ''

        error = str(payload.get('error', '') or '').strip()
        error_type = str(payload.get('error_type', '') or '').strip()
        if error_type == 'MissingLLMCredentials':
            raise MissingLLMCredentials(error or 'credentials required')
        raise RuntimeError(error or f'{model}: remote worker failed')


def _clip_middle(text: str, max_chars: int) -> str:
    marker = "\n...<trimmed>...\n"
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head - len(marker)
    if keep_tail < 0:
        keep_tail = 0
    return text[:keep_head] + marker + text[-keep_tail:]


def _is_colab_env() -> bool:
    return any(
        key in os.environ
        for key in (
            "COLAB_GPU",
            "COLAB_RELEASE_TAG",
            "COLAB_BACKEND_VERSION",
            "GCE_METADATA_TIMEOUT",
        )
    )



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
        resp = _query_model_stable(model, prompt, system, tries=1, timeout=12.0, print_cost=False)
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
            resp = _query_model_stable(model, prompt, system_prompt)
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


def log_status(message: str, *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    if tqdm is not None:
        tqdm.write(message, file=stream)
    else:
        print(message, file=stream)



def _make_model_progress(total_models: int):
    if total_models <= 0 or tqdm is None:
        return None
    return tqdm(
        total=total_models,
        desc="models",
        unit="model",
        dynamic_ncols=True,
        leave=True,
        position=0,
        file=sys.stderr,
    )



def _make_iteration_progress(model: str, max_iters: int):
    if max_iters <= 0 or tqdm is None:
        return None
    return tqdm(
        total=max_iters,
        desc=f"fix {model}",
        unit="iter",
        dynamic_ncols=True,
        leave=False,
        position=1,
        file=sys.stderr,
    )


def build_initial_codegen_prompt(user_prompt: str, plan: str, *, baseline_code: Optional[str] = None) -> str:
    parts = [
        f"USER TASK:\n{user_prompt}",
        f"PLANNER NOTES:\n{plan}",
    ]
    if baseline_code is None:
        parts.append("Now write the solver file.")
    else:
        parts.extend(
            [
                "KNOWN-GOOD BASELINE SOLVER:",
                f"```python\n{baseline_code}\n```",
                (
                    "Modify the baseline minimally to better solve the task while preserving the public entrypoints, "
                    "stdout contract, and dependency-free behavior. Return the complete updated solver file."
                ),
            ]
        )
    return "\n\n".join(parts)


def _query_code_block_with_rescue(
    *,
    model: str,
    prompt: str,
    system_prompt: str,
    stage_label: str,
) -> Tuple[Optional[str], Optional[str]]:
    try:
        resp = _query_model_stable(model, prompt, system_prompt)
    except MissingLLMCredentials as e:
        return None, f"{model}: {stage_label} credentials required ({e})"
    except Exception as e:
        return None, f"{model}: {stage_label} failed ({e})"

    code = extract_python(resp or "")
    if code:
        return code, None

    rescue_prompt = (
        f"{prompt}\n\n"
        "CRITICAL OUTPUT FORMAT REPAIR:\n"
        "Return exactly one complete Python file inside a single ```python``` block.\n"
        "Do not include explanations, bullet points, markdown outside the code fence, or truncated snippets."
    )
    try:
        resp = _query_model_stable(model, rescue_prompt, system_prompt, tries=1)
    except MissingLLMCredentials as e:
        return None, f"{model}: {stage_label} format-rescue credentials required ({e})"
    except Exception as e:
        return None, f"{model}: {stage_label} format-rescue failed ({e})"

    code = extract_python(resp or "")
    if code:
        return code, None
    return None, f"{model}: {stage_label} did not return a python file"


def _run_fixer_loop(
    *,
    fixer_models: Sequence[str],
    user_prompt: str,
    prompts: Dict[str, str],
    out_path: Path,
    validator_path: Path,
    tests: Sequence[List[int]],
    max_iters: int,
    current_code: str,
    last_report: str,
    progress_label: str,
) -> Tuple[bool, str]:
    max_code_chars = _env_int("AGENTLAB_MAX_CODE_PROMPT_CHARS", 24000)
    max_report_chars = _env_int("AGENTLAB_MAX_FAILURE_REPORT_CHARS", 12000)

    exceeded, rss_mb, limit_mb = _memory_limit_exceeded()
    if exceeded:
        _best_effort_release_memory(clear_local_cache=False)
        return False, (
            f"{progress_label}: aborting before fixer loop because RSS {rss_mb:.1f} MB reached the configured limit "
            f"{limit_mb} MB. Reduce --max-iters or raise AGENTLAB_MAX_RSS_MB."
        )

    progress = _make_iteration_progress(progress_label, max_iters)
    if progress is not None:
        progress.set_postfix_str(f"iter 0/{max_iters}")

    try:
        for it in range(1, max_iters + 1):
            if progress is not None:
                progress.set_postfix_str(f"iter {it}/{max_iters}")
            exceeded, rss_mb, limit_mb = _memory_limit_exceeded()
            if exceeded:
                return False, (
                    f"{progress_label}: stopped fixer loop at iteration {it} because RSS {rss_mb:.1f} MB reached "
                    f"the configured limit {limit_mb} MB. Reduce --max-iters or raise AGENTLAB_MAX_RSS_MB."
                )

            fix_prompt = (
                f"USER TASK:\n{user_prompt}\n\n"
                f"CURRENT CODE:\n```python\n{_clip_middle(current_code, max_code_chars)}\n```\n\n"
                f"FAILURE REPORT:\n{_clip_middle(last_report, max_report_chars)}\n\n"
                "Return a corrected full python file."
            )

            new_code = None
            fixer_errors: List[str] = []
            for fix_model in fixer_models:
                log_status(f"[fixer] iteration {it} trying model: {fix_model}")
                candidate, err = _query_code_block_with_rescue(
                    model=fix_model,
                    prompt=fix_prompt,
                    system_prompt=prompts["fixer"],
                    stage_label=f"fixer iteration {it}",
                )
                if candidate:
                    new_code = candidate
                    break
                if err:
                    fixer_errors.append(err)

            if new_code is None:
                return False, "\n".join(fixer_errors) if fixer_errors else f"{progress_label}: fixer iteration {it} returned no python file"

            ok, compile_err = compile_python(new_code)
            current_code = new_code
            if progress is not None:
                progress.update(1)

            if not ok:
                last_report = f"Fix iteration {it} compile check failed.\n{compile_err}\n"
                _best_effort_release_memory(clear_local_cache=False)
                continue

            out_path.write_text(current_code, encoding="utf-8")
            valid, last_report = validate_solver_suite(validator_path, out_path, tests)
            _best_effort_release_memory(clear_local_cache=False)
            exceeded, rss_mb, limit_mb = _memory_limit_exceeded()
            if valid:
                return True, f"{progress_label}: validated after fixer iteration {it}"
            if exceeded:
                return False, (
                    f"{progress_label}: stopping after validation step {it} because RSS {rss_mb:.1f} MB reached "
                    f"the configured limit {limit_mb} MB."
                )
    finally:
        if progress is not None:
            progress.close()
        _best_effort_release_memory(clear_local_cache=False)

    return False, f"{progress_label}: failed validation after {max_iters} fixer iterations\n{_clip_middle(last_report, max_report_chars)}"


def try_generate_with_model(
    *,
    model: str,
    fixer_models: Sequence[str],
    user_prompt: str,
    plan: str,
    prompts: Dict[str, str],
    out_path: Path,
    validator_path: Path,
    tests: Sequence[List[int]],
    max_iters: int,
    baseline_code: Optional[str] = None,
    stage_label: str = "coder",
) -> Tuple[bool, str]:
    coder_prompt = build_initial_codegen_prompt(user_prompt, plan, baseline_code=baseline_code)
    code, err = _query_code_block_with_rescue(
        model=model,
        prompt=coder_prompt,
        system_prompt=prompts["coder"],
        stage_label=stage_label,
    )
    if not code:
        return False, err or f"{model}: {stage_label} did not return a python file"

    ok, compile_err = compile_python(code)
    if not ok:
        last_report = f"Initial compile check failed.\n{compile_err}\n"
    else:
        out_path.write_text(code, encoding="utf-8")
        valid, last_report = validate_solver_suite(validator_path, out_path, tests)
        if valid:
            immediate_label = f"{stage_label} output validated immediately" if stage_label != "coder" else "coder output validated immediately"
            return True, f"{model}: {immediate_label}"

    progress_label = f"{stage_label}:{model}" if stage_label != "coder" else model
    return _run_fixer_loop(
        fixer_models=fixer_models,
        user_prompt=user_prompt,
        prompts=prompts,
        out_path=out_path,
        validator_path=validator_path,
        tests=tests,
        max_iters=max_iters,
        current_code=code,
        last_report=last_report,
        progress_label=progress_label,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user-prompt", default="", help="User prompt (inline string).")
    p.add_argument("--user-prompt-file", default=None, help="Path to a text file with the user prompt.")
    p.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=(
            "Comma-separated default model list. Bare names use g4f backend (remote providers). "
            "You can also pass explicit backends like local:<hf_model_id> to run Transformers locally (CUDA-supported)."
        ),
    )
    p.add_argument(
        "--agent-models",
        default=None,
        help=(
            "Optional per-agent override mapping, e.g. 'planner=gpt-4;coder=local:Qwen/Qwen2.5-Coder-1.5B;fixer=gpt-4o-mini'. "
            "Each value accepts the same comma-separated syntax as --models."
        ),
    )
    p.add_argument("--planner-models", default=None, help="Optional model list override for the planner agent.")
    p.add_argument("--coder-models", default=None, help="Optional model list override for the coder agent.")
    p.add_argument("--fixer-models", default=None, help="Optional model list override for the fixer agent.")
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
        log_status("[!] Empty user prompt. Provide --user-prompt or --user-prompt-file.", error=True)
        sys.exit(2)

    prompts = load_prompts(args.custom_prompts)
    models = parse_models(args.models)
    if not models and not args.no_llm:
        log_status("[!] No models configured. Pass --models or set G4F_MODELS.", error=True)
        sys.exit(2)
    ordered_models = order_models_for_codegen(models)

    agent_model_overrides = parse_agent_model_overrides(args.agent_models)
    apply_agent_model_override(agent_model_overrides, "planner", args.planner_models)
    apply_agent_model_override(agent_model_overrides, "coder", args.coder_models)
    apply_agent_model_override(agent_model_overrides, "fixer", args.fixer_models)

    planner_models = resolve_agent_models("planner", ordered_models, agent_model_overrides)
    coder_models = resolve_agent_models("coder", ordered_models, agent_model_overrides)
    fixer_models = resolve_agent_models("fixer", coder_models, agent_model_overrides)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validator_path = Path(args.validator).resolve()

    baseline_path = Path(args.baseline) if args.baseline else (Path.cwd() / "solve_module.py")
    if baseline_path.exists():
        baseline_code = baseline_path.read_text(encoding="utf-8")
    else:
        baseline_code = make_baseline_stub()

    if any(_use_remote_subprocess_isolation(model) for model in set(planner_models + coder_models + fixer_models)):
        log_status('[memory] Remote LLM queries run in isolated subprocesses to keep notebook RAM stable.')

    if agent_model_overrides:
        log_status(
            "[models] "
            + ", ".join(
                [
                    f"planner={','.join(planner_models)}",
                    f"coder={','.join(coder_models)}",
                    f"fixer={','.join(fixer_models)}",
                ]
            )
        )

    memory_cap_mb = _default_max_rss_mb()
    if memory_cap_mb > 0:
        log_status(
            f"[memory] RSS guard is enabled at ~{memory_cap_mb} MB. "
            "Set AGENTLAB_MAX_RSS_MB=0 to disable or choose a larger value."
        )

    if args.no_llm:
        out_path.write_text(baseline_code, encoding="utf-8")
        log_status(f"[+] Wrote baseline solver to {out_path}")
        sys.exit(0)

    def _fallback_to_baseline(reason: str) -> None:
        log_status(f"[!] {reason}", error=True)
        if args.strict:
            sys.exit(1)
        out_path.write_text(baseline_code, encoding="utf-8")
        log_status("[!] Falling back to the offline baseline solver.", error=True)
        log_status(f"[+] Wrote baseline solver to {out_path}")
        sys.exit(0)

    try:
        plan, planner_model = ask_first_nonempty(planner_models, user_prompt, prompts["planner"])
        if not plan:
            plan = "(planner failed; proceeding without planner notes)"
        log_status(f"[planner] selected model: {planner_model or 'none'}")
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


    model_progress = _make_model_progress(len(coder_models))
    if model_progress is not None:
        model_progress.set_postfix_str(f"model 0/{len(coder_models)}")

    generation_reports: List[str] = []
    try:
        for idx, model in enumerate(coder_models, start=1):
            if model_progress is not None:
                model_progress.set_postfix_str(f"model {idx}/{len(coder_models)}: {model}")
            log_status(f"[coder] trying model: {model}")
            ok, report = try_generate_with_model(
                model=model,
                fixer_models=fixer_models,
                user_prompt=user_prompt,
                plan=plan,
                prompts=prompts,
                out_path=out_path,
                validator_path=validator_path,
                tests=tests,
                max_iters=args.max_iters,
            )
            generation_reports.append(report)
            if model_progress is not None:
                model_progress.update(1)
            if ok:
                log_status(f"[+] {report}. Saved to {out_path}")
                sys.exit(0)
            log_status(f"[coder] {report}")
    finally:
        if model_progress is not None:
            model_progress.close()

    baseline_patch_models = resolve_agent_models("baseline-patcher", fixer_models or coder_models, agent_model_overrides)
    if baseline_code.strip() and baseline_patch_models:
        patch_iters = max(1, min(args.max_iters, _env_int("AGENTLAB_BASELINE_PATCH_MAX_ITERS", 2)))
        log_status(
            "[baseline-patcher] attempting a minimal validated patch of the known-good baseline before offline fallback."
        )
        for model in baseline_patch_models:
            log_status(f"[baseline-patcher] trying model: {model}")
            ok, report = try_generate_with_model(
                model=model,
                fixer_models=fixer_models,
                user_prompt=user_prompt,
                plan=plan,
                prompts=prompts,
                out_path=out_path,
                validator_path=validator_path,
                tests=tests,
                max_iters=patch_iters,
                baseline_code=baseline_code,
                stage_label="baseline-patcher",
            )
            generation_reports.append(report)
            if ok:
                log_status(f"[+] {report}. Saved to {out_path}")
                sys.exit(0)
            log_status(f"[baseline-patcher] {report}")

    detail = "\n".join(generation_reports[-6:]).strip()
    reason = "Failed to generate a locally validated solver with the configured models."
    if detail:
        reason = f"{reason}\n{detail}"
    _fallback_to_baseline(reason)


if __name__ == "__main__":
    main()
