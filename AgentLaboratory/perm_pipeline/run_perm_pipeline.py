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
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
REPO_ROOT = AGENTLAB_ROOT.parent
sys.path.insert(0, str(AGENTLAB_ROOT))
sys.path.insert(0, str(REPO_ROOT))
from inference import query_model, MissingLLMCredentials, _best_effort_release_memory, _run_json_worker_subprocess  # type: ignore
import llm_code_contract as code_contract

RE_PY_BLOCK = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
RE_ANY_BLOCK = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```", re.DOTALL)
RE_FENCED_BLOCK = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*(?P<code>.*?)```", re.DOTALL)
RE_RAW_CODE_START = re.compile(
    r"^(?:#!\s*/|from\s+\S+\s+import|import\s+\S+|async\s+def\s+\w+\s*\(|def\s+\w+\s*\(|class\s+\w+\s*(?:\(|:)|if __name__ == [\"']__main__[\"']\s*:|@[A-Za-z_][A-Za-z0-9_\.\(\), ]*)",
    re.IGNORECASE,
)
RE_CODE_LIKE_LINE = re.compile(
    r"^(?:@|async\s+def\s+\w+\s*\(|def\s+\w+\s*\(|class\s+\w+\s*(?:\(|:)|from\s+\S+\s+import|import\s+\S+|if\b|elif\b|else:|for\b|while\b|try:|except\b|finally:|with\b|return\b|raise\b|assert\b|pass\b|break\b|continue\b|[A-Za-z_][A-Za-z0-9_\[\], ]*\s*=)",
    re.IGNORECASE,
)

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


def _model_backend_family(model: str) -> str:
    raw = (model or "").strip().lower()
    if raw.startswith("local:"):
        return "local-transformers"
    if raw.startswith("ollama:"):
        return "ollama"
    if raw.startswith("vllm:"):
        return "vllm"
    if raw.startswith("lmstudio:"):
        return "lmstudio"
    if raw.startswith("openai-compatible:") or raw.startswith("openai_compatible:") or raw.startswith("compat:"):
        return "openai-compatible"
    if raw.startswith("g4fapi:"):
        return "g4fapi"
    if raw.startswith("g4f:"):
        return "g4f"
    return "api"


def _interleave_by_backend_diversity(models: Sequence[str]) -> List[str]:
    buckets: Dict[str, List[str]] = {}
    for model in models:
        buckets.setdefault(_model_backend_family(model), []).append(model)
    ordered_families = sorted(buckets.keys(), key=lambda name: (name not in {"local-transformers", "ollama", "vllm", "lmstudio", "openai-compatible", "g4fapi"}, name))
    merged: List[str] = []
    while True:
        progressed = False
        for family in ordered_families:
            items = buckets.get(family) or []
            if not items:
                continue
            merged.append(items.pop(0))
            progressed = True
        if not progressed:
            break
    return merged


@dataclass
class PlanCandidate:
    plan_text: str
    planner_model: str
    score: float
    variant_index: int
    depth: int = 0
    source: str = "planner"
    planner_payload: Optional[Dict[str, Any]] = None
    strategy_package: Optional[Dict[str, Any]] = None
    parent_signature: str = ""
    prompt_score: float = 0.0


@dataclass
class ArchiveEntry:
    plan_text: str
    planner_model: str
    coder_model: str
    ok: bool
    report: str
    code: str = ""
    stage_label: str = "coder"
    score: float = 0.0


@dataclass
class CandidateArchive:
    max_items: int = 6
    entries: List[ArchiveEntry] = field(default_factory=list)

    def add(self, entry: ArchiveEntry) -> None:
        self.entries.append(entry)
        self.entries.sort(key=lambda e: (-e.score, e.ok, len(e.report or "")))
        if self.max_items > 0:
            self.entries = self.entries[: self.max_items]

    def best_failures(self, limit: int = 3) -> List[ArchiveEntry]:
        return [e for e in self.entries if not e.ok][: max(0, limit)]

    def summary_text(self, limit: int = 3) -> str:
        failures = self.best_failures(limit=limit)
        if not failures:
            return ""
        blocks = []
        for idx, entry in enumerate(failures, start=1):
            blocks.append(
                f"ATTEMPT {idx}: planner={entry.planner_model} coder={entry.coder_model}\n"
                f"PLAN:\n{_clip_middle(entry.plan_text or '', 1200)}\n\n"
                f"FAILURE:\n{_clip_middle(entry.report or '', 1600)}"
            )
        return "\n\n".join(blocks)


COMMON_PLAN_MUST_PRESERVE: Tuple[str, ...] = (
    "exact_lookup_first",
    "solve_signature",
    "script_json_output",
    "dependency_free_python",
    "deterministic_behavior",
    "legal_move_names_only",
)

COMMON_PLAN_FORBIDDEN: Tuple[str, ...] = (
    "instance-growing BFS",
    "instance-growing DFS",
    "IDA*",
    "beam search over full puzzle state",
    "brute force over bundled rows",
    "UNSOLVED for bundled rows",
    "changing public solve(vec) contract",
)

PLANNER_STRATEGY_PACKAGES: Tuple[Dict[str, Any], ...] = (
    {
        "strategy_family": "stronger_exact_table",
        "label": "Variant A / stronger exact short-word table",
        "goal": "Keep exact lookup first and improve constant-depth exact replacements for short move words.",
        "edit_targets": ["_short_word_data", "_reduce_commuting_word", "_optimize_word"],
        "proposed_changes": [
            "Strengthen fixed-depth exact replacement table construction.",
            "Canonicalize equivalent short effects before storing them.",
            "Reuse cached packed effects instead of widening search depth.",
        ],
        "validation_plan": [
            "Compile solver and preserve solve(vec) contract.",
            "Replay bundled rows and compare final_state against baseline semantics.",
            "Check score gain comes from shorter equivalent words rather than skipped moves.",
        ],
    },
    {
        "strategy_family": "bounded_window_dp",
        "label": "Variant B / bounded-window DP rewrite",
        "goal": "Keep per-row runtime polynomial by improving fixed-window local optimization passes only.",
        "edit_targets": ["_optimize_local_windows", "_optimize_word", "_compose_words"],
        "proposed_changes": [
            "Use stronger bounded-window dynamic programming with fixed pass counts.",
            "Memoize repeated local windows by packed effect.",
            "Prefer deterministic left-to-right canonicalization before each pass.",
        ],
        "validation_plan": [
            "Compile and run validator on bundled smoke vectors.",
            "Ensure window size and pass count stay constant with input size.",
            "Compare rewritten word length against baseline on sample rows.",
        ],
    },
    {
        "strategy_family": "bidirectional_local_replacement",
        "label": "Variant C / bidirectional local replacement",
        "goal": "Improve local exact replacement strength with a constant-radius bidirectional table, never with full-state search.",
        "edit_targets": ["_short_word_data", "_best_local_rewrite", "_optimize_word"],
        "proposed_changes": [
            "Precompute constant-radius forward and reverse local effects.",
            "Match windows by effect and replace them with the shortest equivalent word.",
            "Keep all tables bounded by fixed radii independent of bundled row difficulty.",
        ],
        "validation_plan": [
            "Compile solver and replay exact lookup outputs after replacement.",
            "Verify no generic frontier or queue over puzzle states appears.",
            "Benchmark per-row runtime on short and long bundled paths.",
        ],
    },
    {
        "strategy_family": "offline_parameter_sweep",
        "label": "Variant D / offline parameter sweep",
        "goal": "Add safe parameterization and deterministic auto-selection without changing the high-level exact-lookup-plus-rewrite architecture.",
        "edit_targets": ["_short_word_data", "_optimize_local_windows", "solve"],
        "proposed_changes": [
            "Expose a tiny fixed grid of rewrite parameters.",
            "Evaluate candidates with deterministic local score estimates only.",
            "Choose the best bounded candidate without instance-growing search.",
        ],
        "validation_plan": [
            "Compile solver and keep bundled output deterministic across runs.",
            "Confirm the sweep grid is fixed and small.",
            "Validate that the selected candidate still replays to the correct central state.",
        ],
    },
)


def _strategy_package_for_variant(variant_index: int) -> Dict[str, Any]:
    if not PLANNER_STRATEGY_PACKAGES:
        raise RuntimeError('planner strategy packages are not configured')
    zero_based = max(0, int(variant_index) - 1)
    template = PLANNER_STRATEGY_PACKAGES[zero_based % len(PLANNER_STRATEGY_PACKAGES)]
    package = dict(template)
    package.setdefault('must_preserve', list(COMMON_PLAN_MUST_PRESERVE))
    package.setdefault('forbidden', list(COMMON_PLAN_FORBIDDEN))
    package.setdefault('patch_scope', 'minimal_patch')
    package.setdefault('complexity_claim', {
        'precompute': 'constant with respect to row length',
        'per_row': 'polynomial in the emitted baseline path length',
        'why_polynomial': 'all search radii, window sizes, and pass counts stay fixed constants',
    })
    return package


def _strategy_package_text(package: Optional[Dict[str, Any]]) -> str:
    if not package:
        return ''
    lines = [
        f"Family: {package.get('strategy_family', 'unspecified')}",
        f"Label: {package.get('label', 'unnamed package')}",
        f"Goal: {package.get('goal', '')}",
    ]
    edit_targets = [str(x).strip() for x in package.get('edit_targets', []) if str(x).strip()]
    if edit_targets:
        lines.append('Preferred edit targets: ' + ', '.join(edit_targets))
    must_preserve = [str(x).strip() for x in package.get('must_preserve', []) if str(x).strip()]
    if must_preserve:
        lines.append('Must preserve: ' + ', '.join(must_preserve))
    forbidden = [str(x).strip() for x in package.get('forbidden', []) if str(x).strip()]
    if forbidden:
        lines.append('Forbidden: ' + ', '.join(forbidden))
    return '\n'.join(lines)


def _planner_schema_for_package(package: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    enum_value = str((package or {}).get('strategy_family') or 'structured_plan')
    return {
        'type': 'object',
        'required': [
            'strategy_family',
            'goal',
            'edit_targets',
            'must_preserve',
            'complexity_claim',
            'proposed_changes',
            'validation_plan',
            'forbidden',
        ],
        'properties': {
            'strategy_family': {'type': 'string', 'enum': [enum_value]},
            'goal': {'type': 'string'},
            'edit_targets': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1},
            'must_preserve': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 3},
            'complexity_claim': {
                'type': 'object',
                'required': ['precompute', 'per_row', 'why_polynomial'],
                'properties': {
                    'precompute': {'type': 'string'},
                    'per_row': {'type': 'string'},
                    'why_polynomial': {'type': 'string'},
                },
            },
            'proposed_changes': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 2},
            'validation_plan': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 2},
            'forbidden': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 3},
            'patch_scope': {'type': 'string'},
            'notes': {'type': 'string'},
        },
        'additionalProperties': True,
    }


def _balanced_json_object(text: str) -> Optional[str]:
    start = text.find('{')
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    quote = ''
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == quote:
                in_string = False
            continue
        if ch in {'"', "'"}:
            in_string = True
            quote = ch
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start: idx + 1]
    return None


def _lenient_load_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidate = str(text or '').strip()
    if not candidate:
        return None
    candidates: List[str] = [candidate]
    fenced = RE_ANY_BLOCK.findall(candidate)
    for block in fenced:
        if block and block.strip() not in candidates:
            candidates.append(block.strip())
    balanced = _balanced_json_object(candidate)
    if balanced and balanced not in candidates:
        candidates.append(balanced)
    for payload_text in candidates:
        try:
            loaded = json.loads(payload_text)
        except Exception:
            try:
                loaded = ast.literal_eval(payload_text)
            except Exception:
                continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _string_list(value: Any, fallback: Sequence[str]) -> List[str]:
    if isinstance(value, (list, tuple)):
        out = [str(item).strip() for item in value if str(item).strip()]
        if out:
            return out
    return [str(item).strip() for item in fallback if str(item).strip()]


def _normalize_complexity_claim(value: Any, package: Optional[Dict[str, Any]]) -> Dict[str, str]:
    default_claim = dict((package or {}).get('complexity_claim') or {})
    out = {
        'precompute': str(default_claim.get('precompute') or 'constant with respect to row length'),
        'per_row': str(default_claim.get('per_row') or 'polynomial in the emitted baseline path length'),
        'why_polynomial': str(default_claim.get('why_polynomial') or 'all search radii, window sizes, and pass counts stay fixed constants'),
    }
    if isinstance(value, dict):
        for key in ('precompute', 'per_row', 'why_polynomial'):
            raw = str(value.get(key, '') or '').strip()
            if raw:
                out[key] = raw
    return out


def _fallback_plan_notes(raw_text: str) -> str:
    lines = [line.strip(' -*\t') for line in str(raw_text or '').splitlines() if line.strip()]
    if not lines:
        return ''
    return ' | '.join(lines[:4])


def _normalize_structured_plan(payload: Dict[str, Any], package: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pkg = package or {}
    normalized: Dict[str, Any] = {
        'strategy_family': str(payload.get('strategy_family') or pkg.get('strategy_family') or 'structured_plan').strip(),
        'goal': str(payload.get('goal') or pkg.get('goal') or '').strip(),
        'edit_targets': _string_list(payload.get('edit_targets'), pkg.get('edit_targets', [])),
        'must_preserve': _string_list(payload.get('must_preserve'), pkg.get('must_preserve', COMMON_PLAN_MUST_PRESERVE)),
        'complexity_claim': _normalize_complexity_claim(payload.get('complexity_claim'), pkg),
        'proposed_changes': _string_list(payload.get('proposed_changes'), pkg.get('proposed_changes', [])),
        'validation_plan': _string_list(payload.get('validation_plan'), pkg.get('validation_plan', [])),
        'forbidden': _string_list(payload.get('forbidden'), pkg.get('forbidden', COMMON_PLAN_FORBIDDEN)),
        'patch_scope': str(payload.get('patch_scope') or pkg.get('patch_scope') or 'minimal_patch').strip(),
    }
    notes = str(payload.get('notes') or '').strip()
    if notes:
        normalized['notes'] = notes
    return normalized


def _fallback_structured_plan(raw_text: str, package: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pkg = package or {}
    fallback = {
        'strategy_family': str(pkg.get('strategy_family') or 'structured_plan'),
        'goal': str(pkg.get('goal') or 'Improve the baseline via bounded local rewrites.'),
        'edit_targets': list(pkg.get('edit_targets') or []),
        'must_preserve': list(pkg.get('must_preserve') or COMMON_PLAN_MUST_PRESERVE),
        'complexity_claim': _normalize_complexity_claim(None, pkg),
        'proposed_changes': list(pkg.get('proposed_changes') or []),
        'validation_plan': list(pkg.get('validation_plan') or []),
        'forbidden': list(pkg.get('forbidden') or COMMON_PLAN_FORBIDDEN),
        'patch_scope': str(pkg.get('patch_scope') or 'minimal_patch'),
        'notes': _fallback_plan_notes(raw_text),
    }
    return _normalize_structured_plan(fallback, pkg)


def _coerce_structured_plan(raw_text: str, package: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = _lenient_load_json_object(raw_text)
    if isinstance(payload, dict):
        return _normalize_structured_plan(payload, package)
    return _fallback_structured_plan(raw_text, package)


def _structured_plan_json(plan_payload: Optional[Dict[str, Any]]) -> str:
    if not plan_payload:
        return ''
    return json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True)


def _render_structured_plan(plan_payload: Optional[Dict[str, Any]]) -> str:
    if not plan_payload:
        return ''
    complexity = plan_payload.get('complexity_claim') if isinstance(plan_payload.get('complexity_claim'), dict) else {}
    lines = [
        f"Algorithm family: {plan_payload.get('strategy_family', 'structured_plan')}",
        f"Goal: {plan_payload.get('goal', '')}",
    ]
    if plan_payload.get('edit_targets'):
        lines.append('Edit targets: ' + ', '.join(str(x) for x in plan_payload['edit_targets']))
    if plan_payload.get('must_preserve'):
        lines.append('Invariants: ' + ', '.join(str(x) for x in plan_payload['must_preserve']))
    if complexity:
        lines.append(
            'Complexity: precompute='
            + str(complexity.get('precompute', ''))
            + '; per_row='
            + str(complexity.get('per_row', ''))
            + '; proof='
            + str(complexity.get('why_polynomial', ''))
        )
    if plan_payload.get('proposed_changes'):
        lines.append('Repair strategy: ' + '; '.join(str(x) for x in plan_payload['proposed_changes']))
    if plan_payload.get('validation_plan'):
        lines.append('Validation plan: ' + '; '.join(str(x) for x in plan_payload['validation_plan']))
    if plan_payload.get('forbidden'):
        lines.append('Forbidden: ' + '; '.join(str(x) for x in plan_payload['forbidden']))
    if plan_payload.get('notes'):
        lines.append('Notes: ' + str(plan_payload.get('notes')))
    return '\n'.join(line for line in lines if line.strip())


def _plan_signature(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return normalized[:320]


def _plan_quality_score(text: str) -> float:
    body = str(text or "").strip()
    low = body.lower()
    score = 0.0
    length = len(body)
    if 120 <= length <= 1800:
        score += 18.0
    elif length >= 40:
        score += 8.0
    hints = {
        "invariant": 18.0,
        "correct": 12.0,
        "complexity": 14.0,
        "o(": 10.0,
        "primitive": 10.0,
        "allowed move": 10.0,
        "construct": 8.0,
        "iterative": 6.0,
        "adjacent swap": 8.0,
        "patch": 4.0,
    }
    for needle, value in hints.items():
        if needle in low:
            score += value
    penalties = {
        "bfs": -25.0,
        "dfs": -25.0,
        "a*": -25.0,
        "beam search": -18.0,
        "brute force": -25.0,
        "exponential": -25.0,
    }
    for needle, value in penalties.items():
        if needle in low:
            score += value
    return score


def _string_items(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        s = str(item or "").strip()
        if s:
            out.append(s)
    return out


def _score_plan_payload(plan_payload: Optional[Dict[str, Any]]) -> float:
    if not isinstance(plan_payload, dict):
        return 0.0
    score = 0.0
    edit_targets = _string_items(plan_payload.get("edit_targets"))
    proposed_changes = _string_items(plan_payload.get("proposed_changes"))
    validation_plan = _string_items(plan_payload.get("validation_plan"))
    must_preserve = set(_string_items(plan_payload.get("must_preserve")))
    forbidden = set(_string_items(plan_payload.get("forbidden")))
    score += min(len(edit_targets), 4) * 4.0
    score += min(len(proposed_changes), 4) * 3.0
    score += min(len(validation_plan), 4) * 3.0
    score += len(must_preserve & set(COMMON_PLAN_MUST_PRESERVE)) * 2.0
    score += len(forbidden & set(COMMON_PLAN_FORBIDDEN)) * 1.5
    if str(plan_payload.get("goal") or "").strip():
        score += 4.0
    if str(plan_payload.get("patch_scope") or "").strip():
        score += 3.0
    if str(plan_payload.get("notes") or "").strip():
        score += 2.0
    complexity = plan_payload.get("complexity_claim") or {}
    if isinstance(complexity, dict):
        for key in ("precompute", "per_row", "why_polynomial"):
            if str(complexity.get(key) or "").strip():
                score += 3.0
    return score


def _combined_plan_score(
    plan_text: str,
    plan_payload: Optional[Dict[str, Any]],
    *,
    parent: Optional[PlanCandidate] = None,
    archive_summary: str = "",
    structured_payload: bool = True,
) -> float:
    score = _plan_quality_score(plan_text)
    if structured_payload:
        score += _score_plan_payload(plan_payload)
    low = str(plan_text or "").lower()
    if archive_summary:
        if "avoid" in low or "do not repeat" in low or "failure" in low or "regression" in low:
            score += 5.0
        if "validation plan:" in low:
            score += 2.0
    if parent is None:
        return score
    parent_payload = parent.planner_payload if isinstance(parent.planner_payload, dict) else {}
    cand_signature = _plan_signature(_structured_plan_json(plan_payload) if isinstance(plan_payload, dict) else plan_text)
    if cand_signature == (parent.parent_signature or _plan_signature(parent.plan_text)):
        score -= 40.0
    parent_targets = set(_string_items(parent_payload.get("edit_targets")))
    cand_targets = set(_string_items((plan_payload or {}).get("edit_targets")))
    target_delta = cand_targets - parent_targets
    if target_delta:
        score += min(len(target_delta), 2) * 6.0
    elif cand_targets == parent_targets:
        score -= 3.0
    parent_changes = set(_string_items(parent_payload.get("proposed_changes")))
    cand_changes = set(_string_items((plan_payload or {}).get("proposed_changes")))
    change_delta = cand_changes - parent_changes
    if change_delta:
        score += min(len(change_delta), 2) * 5.0
    elif cand_changes == parent_changes:
        score -= 3.0
    parent_valid = set(_string_items(parent_payload.get("validation_plan")))
    cand_valid = set(_string_items((plan_payload or {}).get("validation_plan")))
    valid_delta = cand_valid - parent_valid
    if valid_delta:
        score += min(len(valid_delta), 2) * 4.0
    elif cand_valid == parent_valid:
        score -= 2.0
    parent_len = len(str(parent.plan_text or ""))
    cand_len = len(str(plan_text or ""))
    if parent_len > 0 and cand_len < max(60, int(parent_len * 0.6)):
        score -= 4.0
    return score


def _record_plan_candidates(history: List[Dict[str, Any]], round_idx: int, plans: Sequence[PlanCandidate], *, phase: str) -> None:
    for plan in plans:
        history.append(
            {
                "phase": phase,
                "round": round_idx,
                "variant_index": plan.variant_index,
                "planner_model": plan.planner_model,
                "score": round(plan.score, 3),
                "prompt_score": round(plan.prompt_score or plan.score, 3),
                "source": plan.source,
                "parent_signature": plan.parent_signature,
                "signature": _plan_signature(_structured_plan_json(plan.planner_payload) if isinstance(plan.planner_payload, dict) else plan.plan_text),
                "summary": _clip_middle(plan.plan_text, 1200),
            }
        )


def _write_plan_history(out_path: Path, history: Sequence[Dict[str, Any]]) -> None:
    try:
        target = out_path.with_name(out_path.stem + "_plan_history.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(list(history), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _build_plan_variant_prompt(
    user_prompt: str,
    *,
    variant_index: int,
    beam_width: int,
    archive_summary: str = "",
    baseline_code: str | None = None,
    strategy_package: Optional[Dict[str, Any]] = None,
) -> str:
    package = strategy_package or _strategy_package_for_variant(variant_index)
    schema_json = json.dumps(_planner_schema_for_package(package), ensure_ascii=False, indent=2, sort_keys=True)
    parts = [
        "## USER TASK",
        user_prompt,
        "## STRATEGY PACKAGE",
        _strategy_package_text(package),
        "## REQUIRED OUTPUT",
        (
            f"Generate planner variant {variant_index} of {beam_width}. Return exactly one JSON object and no prose outside JSON. "
            "The object must describe a minimal patch-over-baseline plan that preserves correctness and polynomial-time guarantees."
        ),
        "## JSON SCHEMA",
        f"```json\n{schema_json}\n```",
    ]
    if baseline_code:
        parts.extend(
            [
                "## KNOWN-GOOD BASELINE SOLVER",
                f"```python\n{_clip_middle(baseline_code, 12000)}\n```",
                "Prefer a minimal patch of the baseline if it can be upgraded into a stronger constructive solver.",
            ]
        )
    if archive_summary:
        parts.extend(
            [
                "## RECENT FAILURE MEMORY",
                archive_summary,
                "Do not repeat the same failure modes; update the patch scope or validation plan instead.",
            ]
        )
    parts.extend(
        [
            "## HARD RULES",
            "Do not write code yet. Do not propose BFS/DFS/beam search over puzzle states. Keep all radii, windows, and passes fixed constants.",
        ]
    )
    return "\n\n".join(parts)


def generate_plan_candidates(
    planner_models: Sequence[str],
    user_prompt: str,
    planner_system_prompt: str,
    *,
    beam_width: int,
    archive_summary: str = "",
    baseline_code: str | None = None,
) -> List[PlanCandidate]:
    if beam_width <= 0:
        return []
    dedup: Dict[str, PlanCandidate] = {}
    ordered_models = _interleave_by_backend_diversity(list(planner_models)) or list(planner_models)
    if not ordered_models:
        return []

    attempts = max(beam_width, min(len(ordered_models) * max(1, beam_width), beam_width * 3))
    for idx in range(attempts):
        model = ordered_models[idx % len(ordered_models)]
        strategy_package = _strategy_package_for_variant(idx + 1)
        plan_prompt = _build_plan_variant_prompt(
            user_prompt,
            variant_index=idx + 1,
            beam_width=beam_width,
            archive_summary=archive_summary,
            baseline_code=baseline_code,
            strategy_package=strategy_package,
        )
        try:
            raw_plan_text = _query_model_stable(model, plan_prompt, planner_system_prompt, tries=1, timeout=18.0)
        except MissingLLMCredentials:
            continue
        except Exception:
            continue
        raw_plan_text = str(raw_plan_text or "").strip()
        if not raw_plan_text:
            continue
        parsed_payload = _lenient_load_json_object(raw_plan_text)
        plan_payload = _normalize_structured_plan(parsed_payload, strategy_package) if isinstance(parsed_payload, dict) else _fallback_structured_plan(raw_plan_text, strategy_package)
        plan_text = _render_structured_plan(plan_payload) if isinstance(parsed_payload, dict) else raw_plan_text
        signature = _plan_signature(_structured_plan_json(plan_payload) if isinstance(parsed_payload, dict) else raw_plan_text)
        score = _combined_plan_score(plan_text, plan_payload, archive_summary=archive_summary, structured_payload=isinstance(parsed_payload, dict))
        candidate = PlanCandidate(
            plan_text=plan_text,
            planner_model=model,
            score=score,
            variant_index=idx + 1,
            depth=0 if not archive_summary else 1,
            source="planner" if not archive_summary else "refiner",
            planner_payload=plan_payload,
            strategy_package=strategy_package,
            parent_signature="",
            prompt_score=score,
        )
        prev = dedup.get(signature)
        if prev is None or candidate.score > prev.score:
            dedup[signature] = candidate
        if len(dedup) >= beam_width and idx + 1 >= beam_width:
            break

    ranked = sorted(dedup.values(), key=lambda c: (-c.score, c.variant_index, c.planner_model))
    return ranked[:beam_width]


def _build_plan_refinement_prompt(
    user_prompt: str,
    *,
    parent: PlanCandidate,
    refinement_round: int,
    archive_summary: str,
    baseline_code: str | None = None,
) -> str:
    package = parent.strategy_package or _strategy_package_for_variant(parent.variant_index)
    schema_json = json.dumps(_planner_schema_for_package(package), ensure_ascii=False, indent=2, sort_keys=True)
    parts = [
        "## USER TASK",
        user_prompt,
        "## STRATEGY PACKAGE",
        _strategy_package_text(package),
        "## PREVIOUS ACCEPTED PLAN",
        _clip_middle(parent.plan_text, 2400),
    ]
    if parent.planner_payload:
        parts.extend(["## PREVIOUS PLANNER JSON", f"```json\n{_structured_plan_json(parent.planner_payload)}\n```"])
    parts.extend(
        [
            "## FAILURE MEMORY",
            archive_summary,
            "## IMPROVEMENT GOAL",
            (
                f"Refinement round {refinement_round}. Return exactly one JSON object and no prose outside JSON. "
                "Produce a STRICTLY BETTER planner prompt/plan than the previous accepted one: preserve all good constraints, "
                "explicitly address at least one failure mode from memory, and add at least one concrete improvement in edit_targets, "
                "proposed_changes, or validation_plan. Do not merely rephrase the previous plan."
            ),
            "## JSON SCHEMA",
            f"```json\n{schema_json}\n```",
        ]
    )
    if baseline_code:
        parts.extend(
            [
                "## KNOWN-GOOD BASELINE SOLVER",
                f"```python\n{_clip_middle(baseline_code, 12000)}\n```",
            ]
        )
    parts.extend(
        [
            "## HARD RULES",
            "Do not write code yet. Keep the plan polynomial-time, bounded-radius, and bundle-safe.",
            "If the new JSON would not beat the previous plan on specificity or validation strength, produce a more concrete plan instead.",
        ]
    )
    return "\n\n".join(parts)
def generate_refined_plan_candidates(
    planner_models: Sequence[str],
    user_prompt: str,
    planner_system_prompt: str,
    *,
    parent_candidates: Sequence[PlanCandidate],
    beam_width: int,
    archive_summary: str,
    baseline_code: str | None = None,
    min_improvement: float = 4.0,
) -> List[PlanCandidate]:
    if beam_width <= 0 or not parent_candidates:
        return []
    dedup: Dict[str, PlanCandidate] = {}
    ordered_models = _interleave_by_backend_diversity(list(planner_models)) or list(planner_models)
    if not ordered_models:
        return []

    for parent_idx, parent in enumerate(parent_candidates[:beam_width], start=1):
        model = ordered_models[(parent_idx - 1) % len(ordered_models)]
        prompt = _build_plan_refinement_prompt(
            user_prompt,
            parent=parent,
            refinement_round=max(1, parent.depth + 1),
            archive_summary=archive_summary,
            baseline_code=baseline_code,
        )
        try:
            raw_plan_text = _query_model_stable(model, prompt, planner_system_prompt, tries=1, timeout=18.0)
        except MissingLLMCredentials:
            continue
        except Exception:
            continue
        raw_plan_text = str(raw_plan_text or "").strip()
        if not raw_plan_text:
            continue
        parsed_payload = _lenient_load_json_object(raw_plan_text)
        strategy_package = parent.strategy_package or _strategy_package_for_variant(parent.variant_index)
        plan_payload = _normalize_structured_plan(parsed_payload, strategy_package) if isinstance(parsed_payload, dict) else _fallback_structured_plan(raw_plan_text, strategy_package)
        plan_text = _render_structured_plan(plan_payload) if isinstance(parsed_payload, dict) else raw_plan_text
        signature = _plan_signature(_structured_plan_json(plan_payload) if isinstance(parsed_payload, dict) else raw_plan_text)
        score = _combined_plan_score(plan_text, plan_payload, parent=parent, archive_summary=archive_summary, structured_payload=isinstance(parsed_payload, dict))
        if signature == _plan_signature(_structured_plan_json(parent.planner_payload) if isinstance(parent.planner_payload, dict) else parent.plan_text):
            continue
        if score <= (parent.score + min_improvement):
            continue
        candidate = PlanCandidate(
            plan_text=plan_text,
            planner_model=model,
            score=score,
            variant_index=parent.variant_index,
            depth=parent.depth + 1,
            source="refiner",
            planner_payload=plan_payload,
            strategy_package=strategy_package,
            parent_signature=_plan_signature(_structured_plan_json(parent.planner_payload) if isinstance(parent.planner_payload, dict) else parent.plan_text),
            prompt_score=score,
        )
        prev = dedup.get(signature)
        if prev is None or candidate.score > prev.score:
            dedup[signature] = candidate

    ranked = sorted(dedup.values(), key=lambda c: (-c.score, c.variant_index, c.planner_model))
    return ranked[:beam_width]


def build_plan_model_frontier(
    plans: Sequence[PlanCandidate],
    coder_models: Sequence[str],
    *,
    frontier_width: int,
) -> List[tuple[PlanCandidate, str]]:
    if not plans or not coder_models or frontier_width <= 0:
        return []
    diverse_models = _interleave_by_backend_diversity(list(coder_models)) or list(coder_models)
    max_pairs = max(frontier_width, min(len(plans) * len(diverse_models), len(plans) * max(1, frontier_width)))
    used = set()
    frontier: List[tuple[PlanCandidate, str]] = []
    for round_idx in range(len(diverse_models)):
        for plan_idx, plan in enumerate(plans[:frontier_width]):
            model = diverse_models[(plan_idx + round_idx) % len(diverse_models)]
            key = (plan.variant_index, model)
            if key in used:
                continue
            used.add(key)
            frontier.append((plan, model))
            if len(frontier) >= max_pairs:
                return frontier
    return frontier


def _attempt_score(ok: bool, report: str) -> float:
    if ok:
        return 1000.0
    lowered = (report or "").lower()
    if "validated after fixer" in lowered:
        return 850.0
    if "solver contract" in lowered:
        return 320.0
    if "compile check failed" in lowered:
        return 260.0
    if "failed validation" in lowered or "test" in lowered:
        return 420.0
    if "credentials required" in lowered:
        return 80.0
    if "timeout" in lowered or "remote worker" in lowered:
        return 120.0
    return 180.0


def _augment_plan_with_archive_context(plan_text: str, archive: CandidateArchive) -> str:
    summary = archive.summary_text(limit=2)
    if not summary:
        return plan_text
    return (
        f"{plan_text}\n\n"
        "EXPERIMENT MANAGER MEMORY:\n"
        "Use the following failed attempts as negative examples. Preserve the good parts, but explicitly avoid repeating the same mistakes.\n\n"
        f"{summary}"
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


def _credential_env_present() -> bool:
    keys = (
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "DEEPINFRA_API_KEY",
        "TOGETHER_API_KEY",
        "HUGGINGFACE_API_KEY",
        "PUTER_API_KEY",
    )
    return any((os.getenv(name, "") or "").strip() for name in keys)


def _model_likely_requires_credentials(model: str) -> bool:
    m = (model or "").strip().lower()
    markers = (
        "gpt-4", "gpt-4o", "claude", "o1", "o3", "gemini", "puter", "openai", "anthropic"
    )
    return any(marker in m for marker in markers)


def _prefer_credentialless_models(models: Sequence[str]) -> List[str]:
    ordered = [m for m in models if m]
    if _credential_env_present():
        return ordered
    credentialless = [m for m in ordered if not _model_likely_requires_credentials(m)]
    return credentialless or ordered


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
    if _model_likely_requires_credentials(model) and not _credential_env_present():
        score -= 40
    return score


def rank_models_for_codegen(models: Sequence[str]) -> List[str]:
    return sorted(models, key=lambda m: (-model_quality_score(m), m.lower()))


def _pos_le(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[0] or (a[0] == b[0] and a[1] <= b[1])


def _span_contains(
    span: Tuple[Tuple[int, int], Tuple[int, int]],
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> bool:
    return _pos_le(span[0], start) and _pos_le(end, span[1])


def _collect_docstring_spans(code: str) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    spans: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    def visit_body(body: Sequence[ast.stmt]) -> None:
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                end_lineno = getattr(value, 'end_lineno', value.lineno)
                end_col = getattr(value, 'end_col_offset', value.col_offset)
                spans.append(((value.lineno, value.col_offset), (end_lineno, end_col)))
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit_body(node.body)

    visit_body(tree.body)
    return spans


def _heuristic_strip_comments_and_docstrings(code: str) -> str:
    source = (code or '').strip()
    if not source:
        return ''

    kept: List[str] = []
    in_triple: Optional[str] = None
    for line in source.splitlines():
        stripped = line.lstrip()

        if in_triple is not None:
            if in_triple in stripped:
                in_triple = None
            continue

        if stripped.startswith('#'):
            continue

        match = re.match(r"^(?:[rRuUbBfF]{0,2})?(?P<quote>'''|\"\"\")", stripped)
        if match:
            quote = match.group('quote')
            tail = stripped[match.end():]
            if quote in tail:
                continue
            in_triple = quote
            continue

        kept.append(line.rstrip())

    cleaned = '\n'.join(kept)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def _strip_python_comments_and_docstrings(code: str) -> str:
    source = (code or '').strip()
    if not source:
        return ''

    original_lines = source.splitlines()
    preserved_prefix: List[str] = []
    if original_lines and original_lines[0].startswith('#!'):
        preserved_prefix.append(original_lines[0].rstrip())
    if len(original_lines) > 1 and re.match(r'^#.*coding[:=]', original_lines[1]):
        preserved_prefix.append(original_lines[1].rstrip())

    try:
        ast.parse(source)
        parsed_ok = True
    except Exception:
        parsed_ok = False

    spans = _collect_docstring_spans(source) if parsed_ok else []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except Exception:
        cleaned = _heuristic_strip_comments_and_docstrings(source) or source
        if preserved_prefix:
            body_lines = cleaned.splitlines()
            while body_lines and body_lines[0].rstrip() in preserved_prefix:
                body_lines.pop(0)
            cleaned = '\n'.join(preserved_prefix + body_lines)
        return cleaned.strip()

    kept = []
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            token_text = tok.string or ''
            if tok.start == (1, 0) and token_text.startswith('#!'):
                kept.append(tok)
            elif tok.start[0] == 2 and preserved_prefix and len(preserved_prefix) > 1 and re.match(r'^#.*coding[:=]', token_text):
                kept.append(tok)
            continue
        if tok.type == tokenize.STRING and any(_span_contains(span, tok.start, tok.end) for span in spans):
            continue
        kept.append(tok)

    try:
        cleaned = tokenize.untokenize(kept)
    except Exception:
        cleaned = _heuristic_strip_comments_and_docstrings(source) or source
        if preserved_prefix:
            body_lines = cleaned.splitlines()
            while body_lines and body_lines[0].rstrip() in preserved_prefix:
                body_lines.pop(0)
            cleaned = '\n'.join(preserved_prefix + body_lines)
        return cleaned.strip()

    cleaned = '\n'.join(line.rstrip() for line in cleaned.splitlines())
    if not parsed_ok and ('\"\"\"' in cleaned or "'''" in cleaned):
        cleaned = _heuristic_strip_comments_and_docstrings(cleaned) or cleaned
    if preserved_prefix:
        body_lines = cleaned.splitlines()
        while body_lines and body_lines[0].rstrip() in preserved_prefix:
            body_lines.pop(0)
        cleaned = '\n'.join(preserved_prefix + body_lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()
def _looks_like_python(code: str) -> bool:
    low = (code or '').lower()
    indicators = (
        'def ',
        'class ',
        'import ',
        'from ',
        '__main__',
        'return ',
        'for ',
        'while ',
        'if ',
        'try:',
        'with ',
    )
    return any(token in low for token in indicators)


def _looks_like_python_line(line: str) -> bool:
    stripped = (line or '').strip()
    if not stripped:
        return True
    if stripped.startswith(('#!', '#', '"""', "'''")):
        return True
    if RE_CODE_LIKE_LINE.match(stripped):
        return True
    if line.startswith((' ', '	')):
        return True
    if re.match(r"^[\]\)\}\],.:]+$", stripped):
        return True
    if stripped.endswith((':', '(', '[', '{', '\\')):
        return True
    return False


def _looks_like_narrative_line(line: str) -> bool:
    stripped = (line or '').strip()
    if not stripped:
        return False
    low = stripped.lower()
    if stripped.startswith('```'):
        return True
    if low.startswith((
        'content of ',
        'code starts here',
        'save as ',
        'note:',
        'explanation',
        'here is',
        "here's",
        'the following code',
        'this is ',
    )):
        return True
    if stripped.startswith(('- ', '* ', '> ', '### ')):
        return True
    if re.match(r"^\d+[.)]\s", stripped):
        return True
    return False


def _trim_candidate_edges(code: str) -> str:
    lines = (code or '').splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and _looks_like_narrative_line(lines[-1]) and not _looks_like_python_line(lines[-1]):
        lines.pop()
    while lines and _looks_like_narrative_line(lines[0]) and not RE_RAW_CODE_START.match(lines[0].strip()):
        lines.pop(0)
    return '\n'.join(lines).strip()


def _extract_raw_python_candidates(text: str) -> List[str]:
    source = (text or '').strip()
    if not source:
        return []
    if not any(token in source for token in ('def solve', 'import ', 'from __future__', 'if __name__', 'class ')):
        return []

    lines = source.splitlines()
    start_indexes: List[int] = []
    for idx, line in enumerate(lines):
        if RE_RAW_CODE_START.match(line.strip()):
            start_indexes.append(idx)

    if not start_indexes:
        return []

    candidates: List[str] = []
    seen = set()
    for start_idx in start_indexes:
        kept: List[str] = []
        for line in lines[start_idx:]:
            stripped = line.strip()
            if stripped.startswith('```'):
                break
            if not stripped:
                kept.append(line)
                continue
            if _looks_like_python_line(line):
                kept.append(line.rstrip())
                continue
            if kept and _looks_like_narrative_line(line):
                break
            if kept:
                break

        candidate = _trim_candidate_edges('\n'.join(kept))
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def _python_candidate_score(code: str, *, lang: str = '', fenced: bool = False) -> int:
    score = 0
    norm_lang = (lang or '').strip().lower()
    low = (code or '').lower()

    if norm_lang in {'python', 'py', 'python3'}:
        score += 140
    elif norm_lang:
        score -= 20

    if fenced:
        score += 5
    if 'def solve' in low:
        score += 120
    if '__main__' in low:
        score += 25
    if 'json.dumps' in low:
        score += 15
    if _looks_like_python(code):
        score += 20

    try:
        ast.parse(code)
        score += 60
    except Exception:
        score -= 10

    nonempty_lines = sum(1 for line in code.splitlines() if line.strip())
    score += min(nonempty_lines, 80)
    return score


def extract_python(resp: str) -> Optional[str]:
    code = code_contract.extract_python_candidate(
        resp or '',
        strip_comments_docstrings=True,
    )
    return code or None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or str(default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or str(default))
    except Exception:
        return default


def _remote_worker_per_attempt_budget(timeout: float, *, model: Optional[str] = None) -> float:
    base_timeout = max(1.0, float(timeout))
    if not _is_remote_model(model or ""):
        return base_timeout
    stream_timeout_s = _env_float('AGENTLAB_G4F_STREAM_TIMEOUT_S', max(3.0, base_timeout + 5.0))
    idle_timeout_s = _env_float('AGENTLAB_G4F_STREAM_IDLE_TIMEOUT_S', max(5.0, min(15.0, base_timeout)))
    budget = base_timeout
    if stream_timeout_s > 0:
        budget = max(budget, float(stream_timeout_s))
    if idle_timeout_s > 0:
        budget = max(budget, float(idle_timeout_s))
    return budget


def _remote_worker_timeout_s(*, tries: int, timeout: float, model: Optional[str] = None) -> int:
    explicit = _env_float('AGENTLAB_REMOTE_WORKER_TIMEOUT_S', 0.0)
    if explicit > 0:
        return max(30, int(math.ceil(explicit)))
    attempts = max(1, int(tries))
    per_attempt_budget = _remote_worker_per_attempt_budget(timeout, model=model)
    per_attempt_buffer = max(5.0, _env_float('AGENTLAB_REMOTE_WORKER_ATTEMPT_BUFFER_S', 5.0))
    startup_buffer = max(10.0, _env_float('AGENTLAB_REMOTE_WORKER_STARTUP_BUFFER_S', 10.0))
    total_budget = startup_buffer + attempts * (per_attempt_budget + per_attempt_buffer)
    return max(30, int(math.ceil(total_budget)))


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
        proc_timeout = _remote_worker_timeout_s(tries=tries, timeout=timeout, model=model)
        payload = _run_json_worker_subprocess(
            cmd=cmd,
            env=env,
            proc_timeout=proc_timeout,
            out_json=out_json,
            tmpdir_path=tmpdir_path,
            model_label=model,
        )

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


def _should_print_generation() -> bool:
    return (os.getenv("AGENTLAB_PRINT_GENERATION", "0") or "").strip().lower() in {"1", "true", "yes", "on"}


def _print_generation_preview(stage_label: str, model: str, text: str) -> None:
    if not _should_print_generation():
        return
    max_chars = _env_int("AGENTLAB_PRINT_GENERATION_MAX_CHARS", 16000)
    body = text or ""
    if max_chars > 0 and len(body) > max_chars:
        body = body[:max_chars] + "\n...<trimmed>..."
    log_status(f"[generation:{stage_label}] model={model}\n{body}")


def _safe_slug(value: str, *, max_len: int = 96) -> str:
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value or "")).strip("_")
    return (value or "item")[:max_len]


def _attempt_archive_dir() -> Optional[Path]:
    raw = os.getenv("AGENTLAB_ATTEMPT_ARCHIVE_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return path


def _archive_text_artifact(stage_label: str, model: str, name: str, text: str) -> None:
    archive_dir = _attempt_archive_dir()
    if archive_dir is None:
        return
    path = archive_dir / f"{_safe_slug(stage_label)}__{_safe_slug(model)}__{_safe_slug(name)}.txt"
    try:
        path.write_text(text or "", encoding="utf-8")
    except Exception:
        pass


def _archive_json_artifact(name: str, payload: Dict[str, Any]) -> None:
    archive_dir = _attempt_archive_dir()
    if archive_dir is None:
        return
    path = archive_dir / f"{_safe_slug(name)}.json"
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


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


def _canonicalize_common_cli_json_patterns(code: str) -> str:
    source = str(code or "").strip()
    if not source:
        return ""
    try:
        tree = ast.parse(source)
    except Exception:
        return source

    class _JsonLoadsArgvFixer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):  # type: ignore[override]
            self.generic_visit(node)
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == 'loads'
                and isinstance(func.value, ast.Name)
                and func.value.id == 'json'
                and node.args
            ):
                return node

            first_arg = node.args[0]
            replacement = None
            if isinstance(first_arg, ast.Attribute) and isinstance(first_arg.value, ast.Name):
                if first_arg.value.id == 'sys' and first_arg.attr == 'argv':
                    replacement = ast.Subscript(
                        value=ast.Attribute(value=ast.Name(id='sys', ctx=ast.Load()), attr='argv', ctx=ast.Load()),
                        slice=ast.Constant(value=1),
                        ctx=ast.Load(),
                    )
            elif isinstance(first_arg, ast.Name) and first_arg.id == 'argv':
                replacement = ast.Subscript(
                    value=ast.Name(id='argv', ctx=ast.Load()),
                    slice=ast.Constant(value=1),
                    ctx=ast.Load(),
                )

            if replacement is not None:
                node.args[0] = ast.copy_location(replacement, first_arg)
            return node

    fixed_tree = _JsonLoadsArgvFixer().visit(tree)
    ast.fix_missing_locations(fixed_tree)
    try:
        normalized = ast.unparse(fixed_tree)
    except Exception:
        return source
    return normalized.strip() or source


def _sanitize_candidate_python(code: str) -> str:
    source = str(code or "").strip()
    if not source:
        return ""

    variants: List[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        cleaned = (value or "").strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        variants.append(cleaned)

    _push(source)
    try:
        extracted = code_contract.extract_python_candidate(source, strip_comments_docstrings=False)
    except Exception:
        extracted = ""
    if extracted:
        _push(extracted)

    for candidate in list(variants):
        try:
            normalized = _canonicalize_common_cli_json_patterns(candidate)
        except Exception:
            normalized = ""
        if normalized:
            _push(normalized)

    current = source
    for _ in range(3):
        changed = False
        if len(current) >= 2 and current[0] == current[-1] and current[0] in {'"', "'"}:
            try:
                unquoted = ast.literal_eval(current)
                if isinstance(unquoted, str) and unquoted.strip() and unquoted.strip() != current.strip():
                    current = unquoted
                    _push(current)
                    changed = True
            except Exception:
                pass
        if any(token in current for token in ("\n", "\t", "\r", '\"', "\\")):
            try:
                decoded = bytes(current, "utf-8").decode("unicode_escape")
                if decoded.strip() and decoded.strip() != current.strip():
                    current = decoded
                    _push(current)
                    changed = True
            except Exception:
                pass
        if not changed:
            break

    best = source
    best_score = -10**9
    for candidate in variants:
        ok, _ = compile_python(candidate)
        score = len(candidate)
        if ok:
            score += 100000
        if "\n" in candidate or "\\" in candidate:
            score -= 100
        if score > best_score:
            best = candidate
            best_score = score
    return best


def _validator_timeout_s() -> float:
    raw = os.getenv("AGENTLAB_VALIDATOR_TIMEOUT_S", os.getenv("PIPELINE_SOLVER_TIMEOUT_S", "20"))
    try:
        return max(1.0, float(raw))
    except Exception:
        return 20.0


def _parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(',') if part.strip() != '']


def _extract_state_from_row(row: Dict[str, str]) -> Optional[List[int]]:
    preferred_keys = ("initial_state", "vector", "permutation", "state")
    for key in preferred_keys:
        raw = row.get(key)
        if raw is None or str(raw).strip() == '':
            continue
        try:
            return _parse_int_list(str(raw))
        except Exception:
            continue

    non_empty = {k: v for k, v in row.items() if v is not None and str(v).strip() != ''}
    if len(non_empty) == 1:
        try:
            return _parse_int_list(next(iter(non_empty.values())))
        except Exception:
            return None

    for key, value in non_empty.items():
        if key.lower() in {"id", "index", "initial_state_id", "puzzle_id"}:
            continue
        try:
            return _parse_int_list(str(value))
        except Exception:
            continue
    return None


def resolve_validator_smoke_vectors(validator_path: Path, *, extra_rows: int = 2) -> List[List[int]]:
    vectors: List[List[int]] = []
    seen: set[Tuple[int, ...]] = set()

    def _add(vec: Optional[Sequence[int]]) -> None:
        if vec is None:
            return
        norm = [int(x) for x in vec]
        if not norm:
            return
        sig = tuple(norm)
        if sig in seen:
            return
        seen.add(sig)
        vectors.append(norm)

    validator_dir = validator_path.resolve().parent
    candidate_csvs = [
        validator_dir / 'data' / 'test.csv',
        validator_dir / 'data' / 'puzzles.csv',
        validator_dir / 'test.csv',
        validator_dir / 'puzzles.csv',
    ]

    for csv_path in candidate_csvs:
        if not csv_path.exists():
            continue
        try:
            with csv_path.open(newline='', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vec = _extract_state_from_row(row)
                    if vec:
                        _add(vec)
                    if len(vectors) >= 1 + max(0, int(extra_rows)):
                        return vectors
        except Exception:
            continue

    fallback_tests: List[List[int]] = [
        [3, 1, 2, 5, 4],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [2, 0, 3, 1],
        [10, -1, 7, 3, 5],
    ]
    for vec in fallback_tests:
        _add(vec)
    return vectors


def validate_solver_contract(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno}, offset {e.offset})"
    except Exception as e:  # pragma: no cover - defensive only
        return False, f"ParseError: {type(e).__name__}: {e}"

    solve_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'solve':
            solve_node = node
            break
    if solve_node is None:
        return False, 'missing required function solve(vec)'

    arg_count = len(getattr(solve_node.args, 'posonlyargs', [])) + len(getattr(solve_node.args, 'args', []))
    if arg_count < 1 and solve_node.args.vararg is None:
        return False, 'solve() must accept at least one positional argument'

    return True, ''


def run_validator(validator_path: Path, solver_path: Path, vec: List[int]) -> Tuple[int, str, str]:
    cmd = [sys.executable, str(validator_path), "--solver", str(solver_path), "--vector", json.dumps(vec)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=_validator_timeout_s())
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if err and not err.endswith("\n"):
            err += "\n"
        err += (
            f"[!] validator timed out after {_validator_timeout_s():g}s while checking {solver_path.name}. "
            "Most often this means the generated solve(vec) entered an infinite loop or an unexpectedly expensive search.\n"
        )
        return 124, out, err


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
        "Return exactly one JSON object containing a minimal Python solver module. "
        "Set the code field to a complete solve_module.py that defines solve(vec) and returns the input unchanged. "
        "Do not add any explanation outside JSON.\n\n"
        + code_contract.strict_code_response_requirements(prefer_minimal_patch=False, filename='solve_module.py')
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
    contract_ok, contract_reason = validate_solver_contract(code)
    if not contract_ok:
        return False, contract_reason
    return True, 'ok'


def order_models_for_codegen(models: Sequence[str]) -> List[str]:
    ranked = rank_models_for_codegen(models)
    if os.getenv("AGENTLAB_MODEL_PROBE", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return _interleave_by_backend_diversity(ranked)

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
    if good:
        return _interleave_by_backend_diversity(good + tail)
    return _interleave_by_backend_diversity(tail) + bad


def ask_first_nonempty(models: Sequence[str], prompt: str, system_prompt: str) -> Tuple[str, Optional[str]]:
    last_error: Optional[Exception] = None
    for model in models:
        try:
            resp = _query_model_stable(model, prompt, system_prompt)
            if isinstance(resp, str) and resp.strip():
                _print_generation_preview("planner", model, resp.strip())
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


def ask_first_structured_plan(
    models: Sequence[str],
    user_prompt: str,
    system_prompt: str,
    *,
    baseline_code: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    last_error: Optional[Exception] = None
    for idx, model in enumerate(models, start=1):
        strategy_package = _strategy_package_for_variant(idx)
        prompt = _build_plan_variant_prompt(
            user_prompt,
            variant_index=idx,
            beam_width=max(1, len(models)),
            baseline_code=baseline_code,
            strategy_package=strategy_package,
        )
        try:
            resp = _query_model_stable(model, prompt, system_prompt)
            if isinstance(resp, str) and resp.strip():
                _print_generation_preview("planner", model, resp.strip())
                parsed_payload = _lenient_load_json_object(resp.strip())
                plan_payload = _normalize_structured_plan(parsed_payload, strategy_package) if isinstance(parsed_payload, dict) else _fallback_structured_plan(resp.strip(), strategy_package)
                plan_text = _render_structured_plan(plan_payload) if isinstance(parsed_payload, dict) else resp.strip()
                return plan_text, model, plan_payload, strategy_package
        except MissingLLMCredentials as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue
    if last_error is not None:
        raise last_error
    return "", None, None, None


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


def _strict_output_requirements(*, prefer_minimal_patch: bool) -> str:
    return code_contract.strict_code_response_requirements(
        prefer_minimal_patch=prefer_minimal_patch,
        filename='solve_module.py',
    )


_FROM_SCRATCH_MARKERS = (
    'no_baseline_patch_bias',
    'no reference baseline will be shown',
    'do not expect any baseline section',
    'write the code from scratch',
    'fresh solver from scratch',
    'do not rely on, patch, wrap, or extend any baseline implementation',
)

_CREATIVE_SCORE_MARKERS = (
    'creative_score_search',
    'creative score search',
    'creative search',
    'multiple candidate families',
    'best local score',
    'best bundled-data score',
    'best bundled data score',
)


def _prompt_requests_from_scratch(user_prompt: str) -> bool:
    lowered = str(user_prompt or '').lower()
    return any(marker in lowered for marker in _FROM_SCRATCH_MARKERS)


def _prompt_requests_creative_score_search(user_prompt: str) -> bool:
    lowered = str(user_prompt or '').lower()
    return any(marker in lowered for marker in _CREATIVE_SCORE_MARKERS)


def build_initial_codegen_prompt(
    user_prompt: str,
    plan: str,
    *,
    baseline_code: Optional[str] = None,
    plan_payload: Optional[Dict[str, Any]] = None,
    strategy_package: Optional[Dict[str, Any]] = None,
) -> str:
    max_plan_chars = _env_int("AGENTLAB_MAX_PLAN_PROMPT_CHARS", 12000)
    from_scratch = _prompt_requests_from_scratch(user_prompt)
    creative_score_search = _prompt_requests_creative_score_search(user_prompt)
    parts = [
        "## OUTPUT CONTRACT (HIGHEST PRIORITY)",
        code_contract.concise_code_response_directive(filename='solve_module.py'),
        "## USER TASK",
        user_prompt,
    ]
    if strategy_package:
        parts.extend(["## STRATEGY PACKAGE", _strategy_package_text(strategy_package)])
    if plan_payload:
        parts.extend(
            [
                "## PLANNER JSON",
                f"```json\n{_structured_plan_json(plan_payload)}\n```",
            ]
        )
    parts.extend(
        [
            "## PLANNER SUMMARY",
            _clip_middle(plan, max_plan_chars),
        ]
    )
    if baseline_code is None:
        if from_scratch:
            parts.append('Now write the solver file from scratch as a fully self-contained implementation.')
        else:
            parts.append('Now write the solver file as a bounded patch-consistent implementation.')
    elif from_scratch:
        parts.extend(
            [
                '## REFERENCE BASELINE (compatibility and score target only)',
                f"```python\n{baseline_code}\n```",
                (
                    'Write a fresh solver from scratch. Use the reference baseline only to understand the compatibility contract, '
                    'bundled-data expectations, and the current score target. Do NOT patch, wrap, subclass, or extend the baseline. '
                    'Return a completely rewritten solver file that preserves the public entrypoints, stdout contract, dependency-free behavior, '
                    'exact lookup first, and deterministic replay semantics.'
                ),
            ]
        )
    else:
        parts.extend(
            [
                '## KNOWN-GOOD BASELINE SOLVER',
                f"```python\n{baseline_code}\n```",
                (
                    'Modify the baseline minimally to better solve the task while preserving the public entrypoints, '
                    'stdout contract, dependency-free behavior, exact lookup first, and deterministic replay semantics. '
                    'Return the complete updated solver file.'
                ),
            ]
        )
    implementation_rules = [
        '## IMPLEMENTATION RULES',
        'Keep all search radii, window sizes, and pass counts bounded by constants.',
    ]
    if from_scratch:
        implementation_rules.append(
            'You may introduce new helper layers, data structures, and bounded rewrite passes if they remain deterministic, standard-library-only, and polynomial-time.'
        )
    else:
        implementation_rules.append(
            'Patch only the named edit targets unless another change is strictly required for correctness.'
        )
    if creative_score_search:
        implementation_rules.append(
            'Be creatively search-oriented within the bounded architecture: synthesize a small bank of deterministic candidate optimizers, evaluate them with a local score proxy, and keep only the best valid variant.'
        )
    implementation_rules.append(_strict_output_requirements(prefer_minimal_patch=(baseline_code is not None and not from_scratch)))
    parts.extend(implementation_rules)
    return '\n\n'.join(parts)


def _build_fixer_prompt(
    *,
    user_prompt: str,
    plan: str,
    current_code: str,
    last_report: str,
    baseline_code: Optional[str],
    plan_payload: Optional[Dict[str, Any]],
    strategy_package: Optional[Dict[str, Any]],
    max_code_chars: int,
    max_report_chars: int,
    max_plan_chars: int,
) -> str:
    from_scratch = _prompt_requests_from_scratch(user_prompt)
    parts = [
        "## OUTPUT CONTRACT (HIGHEST PRIORITY)",
        code_contract.concise_code_response_directive(filename='solve_module.py'),
        "## USER TASK",
        user_prompt,
    ]
    if strategy_package:
        parts.extend(["## STRATEGY PACKAGE", _strategy_package_text(strategy_package)])
    if plan_payload:
        parts.extend(["## PLANNER JSON", f"```json\n{_structured_plan_json(plan_payload)}\n```"])
    parts.extend(
        [
            "## PLANNER SUMMARY",
            _clip_middle(plan, max_plan_chars),
            "## CURRENT CODE",
            f"```python\n{_clip_middle(current_code, max_code_chars)}\n```",
            "## FAILURE REPORT",
            _clip_middle(last_report, max_report_chars),
        ]
    )
    if baseline_code:
        parts.extend(
            [
                "## REFERENCE BASELINE" if from_scratch else "## KNOWN-GOOD BASELINE",
                f"```python\n{_clip_middle(baseline_code, max_code_chars)}\n```",
            ]
        )
    repair_order = [
        "## REPAIR ORDER",
        "1. Preserve solve(vec), stdout JSON contract, and legal move names.",
    ]
    if from_scratch:
        repair_order.append("2. You may rewrite a subsystem or regenerate the file from scratch if that is the safest route to a better valid solver.")
    else:
        repair_order.append("2. Fix the smallest possible region that explains the failure.")
    repair_order.extend(
        [
            "3. Do not introduce BFS/DFS/beam search or any instance-growing frontier.",
            _strict_output_requirements(prefer_minimal_patch=not from_scratch),
        ]
    )
    parts.extend(repair_order)
    return '\n\n'.join(parts)


def _query_code_block_with_rescue(
    *,
    model: str,
    prompt: str,
    system_prompt: str,
    stage_label: str,
) -> Tuple[Optional[str], Optional[str]]:
    try:
        _archive_text_artifact(stage_label, model, 'prompt', prompt)
        _archive_text_artifact(stage_label, model, 'system', system_prompt)
        resp = _query_model_stable(model, prompt, system_prompt)
        if isinstance(resp, str) and resp.strip():
            _print_generation_preview(stage_label, model, resp.strip())
            _archive_text_artifact(stage_label, model, 'raw_response', resp.strip())
    except MissingLLMCredentials as e:
        return None, f"{model}: {stage_label} credentials required ({e})"
    except Exception as e:
        return None, f"{model}: {stage_label} failed ({e})"

    code = _sanitize_candidate_python(extract_python(resp or ""))
    if code:
        _archive_text_artifact(stage_label, model, 'extracted_candidate.py', code)
        return code, None

    rescue_prompt = code_contract.repair_code_response_prompt(
        prompt,
        filename='solve_module.py',
    )
    try:
        _archive_text_artifact(f"{stage_label}:format-rescue", model, 'prompt', rescue_prompt)
        resp = _query_model_stable(model, rescue_prompt, system_prompt, tries=1)
        if isinstance(resp, str) and resp.strip():
            _print_generation_preview(f"{stage_label}:format-rescue", model, resp.strip())
            _archive_text_artifact(f"{stage_label}:format-rescue", model, 'raw_response', resp.strip())
    except MissingLLMCredentials as e:
        return None, f"{model}: {stage_label} format-rescue credentials required ({e})"
    except Exception as e:
        return None, f"{model}: {stage_label} format-rescue failed ({e})"

    code = _sanitize_candidate_python(extract_python(resp or ""))
    if code:
        _archive_text_artifact(f"{stage_label}:format-rescue", model, 'extracted_candidate.py', code)
        return code, None
    _archive_json_artifact('no_python_returned', {'stage': stage_label, 'model': model, 'reason': 'did not return a python file'})
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
    plan: str,
    baseline_code: Optional[str] = None,
    plan_payload: Optional[Dict[str, Any]] = None,
    strategy_package: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    max_code_chars = _env_int("AGENTLAB_MAX_CODE_PROMPT_CHARS", 24000)
    max_report_chars = _env_int("AGENTLAB_MAX_FAILURE_REPORT_CHARS", 12000)
    max_plan_chars = _env_int("AGENTLAB_MAX_PLAN_PROMPT_CHARS", 12000)

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

    repair_context_code = current_code if (compile_python(current_code)[0] if current_code.strip() else False) else (baseline_code or current_code)
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

            fix_prompt = _build_fixer_prompt(
                user_prompt=user_prompt,
                plan=plan,
                current_code=repair_context_code,
                last_report=last_report,
                baseline_code=baseline_code,
                plan_payload=plan_payload,
                strategy_package=strategy_package,
                max_code_chars=max_code_chars,
                max_report_chars=max_report_chars,
                max_plan_chars=max_plan_chars,
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

            new_code = _sanitize_candidate_python(new_code)
            ok, compile_err = compile_python(new_code)
            if progress is not None:
                progress.update(1)

            if not ok:
                last_report = f"Fix iteration {it} compile check failed.\n{compile_err}\n"
                if baseline_code and baseline_code.strip():
                    repair_context_code = baseline_code
                _best_effort_release_memory(clear_local_cache=False)
                continue

            current_code = new_code
            repair_context_code = new_code

            contract_ok, contract_err = validate_solver_contract(new_code)
            if not contract_ok:
                last_report = f"Fix iteration {it} solver contract check failed.\n{contract_err}\n"
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
    plan_payload: Optional[Dict[str, Any]] = None,
    strategy_package: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    coder_prompt = build_initial_codegen_prompt(
        user_prompt,
        plan,
        baseline_code=baseline_code,
        plan_payload=plan_payload,
        strategy_package=strategy_package,
    )
    code, err = _query_code_block_with_rescue(
        model=model,
        prompt=coder_prompt,
        system_prompt=prompts["coder"],
        stage_label=stage_label,
    )
    if not code:
        return False, err or f"{model}: {stage_label} did not return a python file"

    code = _sanitize_candidate_python(code)
    ok, compile_err = compile_python(code)
    if not ok:
        last_report = f"Initial compile check failed.\n{compile_err}\n"
    else:
        contract_ok, contract_err = validate_solver_contract(code)
        if not contract_ok:
            last_report = f"Initial solver contract check failed.\n{contract_err}\n"
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
        plan=plan,
        baseline_code=baseline_code,
        plan_payload=plan_payload,
        strategy_package=strategy_package,
    )


def _recovery_enabled() -> bool:
    return _env_int("AGENTLAB_G4F_RECOVERY_ROUNDS", 1) > 0


def _report_is_recoverable(report: str) -> bool:
    lowered = (report or "").lower()
    markers = (
        "did not return a python file",
        "format-rescue failed",
        "remote worker timed out",
        "remote worker failed",
        "remote worker did not produce a result file",
        "failed to parse remote worker output",
        "no python block",
        "provider",
        "timeout",
    )
    return any(marker in lowered for marker in markers)


def _build_recovery_plan(plan: str, generation_reports: Sequence[str]) -> str:
    recent = [r.strip() for r in generation_reports[-4:] if str(r or "").strip()]
    recent_text = "\n\n".join(_clip_middle(r, 1200) for r in recent)
    guidance = (
        "RECOVERY MODE:\n"
        "A previous remote-model attempt failed because the provider returned malformed output or became unstable.\n"
        "Return exactly one complete dependency-free Python solver file.\n"
        "Prefer a minimal patch of the known-good baseline over a rewrite.\n"
        "Do not include explanations, bullet points, markdown outside the code block, or partial snippets.\n"
        "Keep the public entrypoints and stdout contract unchanged."
    )
    if recent_text:
        guidance += f"\n\nRECENT FAILURES TO AVOID:\n{recent_text}"
    if plan.strip():
        return f"{plan}\n\n{guidance}"
    return guidance


def attempt_recovery_rounds(
    *,
    recovery_models: Sequence[str],
    fixer_models: Sequence[str],
    user_prompt: str,
    plan: str,
    prompts: Dict[str, str],
    out_path: Path,
    validator_path: Path,
    tests: Sequence[List[int]],
    max_iters: int,
    baseline_code: Optional[str],
    generation_reports: List[str],
) -> Tuple[bool, Optional[str]]:
    rounds = max(0, _env_int("AGENTLAB_G4F_RECOVERY_ROUNDS", 1))
    safe_baseline_code = baseline_code or ""
    # main() passes prompt_baseline_code=None for from-scratch prompts. The old
    # guard called baseline_code.strip() and crashed before model fallback could
    # continue to the next g4f model.
    if rounds <= 0 or not recovery_models or not safe_baseline_code.strip():
        return False, None

    recovery_iters = max(1, min(max_iters, _env_int("AGENTLAB_G4F_RECOVERY_MAX_ITERS", max_iters)))
    sleep_s = max(0.0, _env_float("AGENTLAB_G4F_RECOVERY_SLEEP_S", 1.5))

    if not any(_report_is_recoverable(r) for r in generation_reports[-6:]):
        return False, None

    for round_idx in range(1, rounds + 1):
        log_status(
            f"[recovery] round {round_idx}/{rounds}: releasing memory and retrying remote models before offline fallback."
        )
        _best_effort_release_memory(clear_local_cache=True)
        if sleep_s > 0:
            time.sleep(sleep_s)

        recovery_plan = _build_recovery_plan(plan, generation_reports)
        offset = (round_idx - 1) % len(recovery_models)
        rotated_models = list(recovery_models[offset:]) + list(recovery_models[:offset])
        for model in rotated_models:
            log_status(f"[recovery] trying model: {model}")
            ok, report = try_generate_with_model(
                model=model,
                fixer_models=fixer_models,
                user_prompt=user_prompt,
                plan=recovery_plan,
                prompts=prompts,
                out_path=out_path,
                validator_path=validator_path,
                tests=tests,
                max_iters=recovery_iters,
                baseline_code=safe_baseline_code,
                stage_label=f"recovery round {round_idx}",
            )
            generation_reports.append(report)
            if ok:
                return True, report
            log_status(f"[recovery] {report}")

    return False, None


def _attempt_identity(plan_text: str, coder_model: str) -> tuple[str, str]:
    return _plan_signature(plan_text), str(coder_model or "").strip()


def _resolve_search_mode(raw: str | None) -> str:
    mode = str(raw or os.getenv("AGENTLAB_SEARCH_MODE", "hybrid")).strip().lower()
    return mode if mode in {"classic", "hybrid"} else "hybrid"


def _search_settings_from_args(args: argparse.Namespace) -> tuple[str, int, int, int, int]:
    mode = _resolve_search_mode(getattr(args, "search_mode", None))
    plan_beam_width = max(1, int(getattr(args, "plan_beam_width", 3) or 3))
    frontier_width = max(1, int(getattr(args, "frontier_width", 6) or 6))
    archive_size = max(1, int(getattr(args, "archive_size", 6) or 6))
    refine_rounds = max(0, int(getattr(args, "refine_rounds", 1) or 0))
    return mode, plan_beam_width, frontier_width, archive_size, refine_rounds


def run_hybrid_codegen_search(
    *,
    planner_models: Sequence[str],
    coder_models: Sequence[str],
    fixer_models: Sequence[str],
    user_prompt: str,
    prompts: Dict[str, str],
    out_path: Path,
    validator_path: Path,
    tests: Sequence[List[int]],
    max_iters: int,
    baseline_code: str,
    plan_beam_width: int,
    frontier_width: int,
    archive_size: int,
    refine_rounds: int,
) -> tuple[bool, List[str], CandidateArchive, str, Optional[str], Optional[str]]:
    archive = CandidateArchive(max_items=archive_size)
    generation_reports: List[str] = []
    attempts_seen: set[tuple[str, str]] = set()
    last_plan = ""
    last_planner_model: Optional[str] = None
    prompt_history: List[Dict[str, Any]] = []

    initial_plans = generate_plan_candidates(
        planner_models,
        user_prompt,
        prompts["planner"],
        beam_width=plan_beam_width,
        archive_summary="",
        baseline_code=baseline_code,
    )
    if not initial_plans:
        return False, generation_reports, archive, "", None, None
    _record_plan_candidates(prompt_history, 0, initial_plans, phase="initial")
    _write_plan_history(out_path, prompt_history)

    if len(initial_plans) > 1:
        log_status(
            f"[planner] hybrid search prepared {len(initial_plans)} distinct plan candidates; "
            f"frontier_width={frontier_width}, refine_rounds={refine_rounds}."
        )

    plan_rounds: List[List[PlanCandidate]] = [initial_plans]
    round_idx = 0
    while round_idx < len(plan_rounds) and round_idx <= refine_rounds:
        current_plans = plan_rounds[round_idx]
        frontier = build_plan_model_frontier(current_plans, coder_models, frontier_width=frontier_width)
        if frontier:
            for pair_idx, (plan_candidate, coder_model) in enumerate(frontier, start=1):
                raw_plan_text = plan_candidate.plan_text
                plan_text = _augment_plan_with_archive_context(raw_plan_text, archive)
                attempt_key = _attempt_identity(plan_text, coder_model)
                if attempt_key in attempts_seen:
                    continue
                attempts_seen.add(attempt_key)

                last_plan = raw_plan_text
                last_planner_model = plan_candidate.planner_model
                label_suffix = f"variant {plan_candidate.variant_index} planner={plan_candidate.planner_model} coder={coder_model}"
                if round_idx > 0:
                    label_suffix += f" refine_round={round_idx}"
                log_status(f"[hybrid] attempt {len(attempts_seen)}: {label_suffix} score={plan_candidate.score:.1f}")

                ok, report = try_generate_with_model(
                    model=coder_model,
                    fixer_models=fixer_models,
                    user_prompt=user_prompt,
                    plan=plan_text,
                    prompts=prompts,
                    out_path=out_path,
                    validator_path=validator_path,
                    tests=tests,
                    max_iters=max_iters,
                    baseline_code=baseline_code,
                    stage_label=f"coder variant {plan_candidate.variant_index}" if round_idx == 0 else f"coder refine {round_idx} variant {plan_candidate.variant_index}",
                    plan_payload=plan_candidate.planner_payload,
                    strategy_package=plan_candidate.strategy_package,
                )
                generation_reports.append(report)
                archive.add(
                    ArchiveEntry(
                        plan_text=raw_plan_text,
                        planner_model=plan_candidate.planner_model,
                        coder_model=coder_model,
                        ok=ok,
                        report=report,
                        stage_label="coder",
                        score=_attempt_score(ok, report),
                    )
                )
                _write_plan_history(out_path, prompt_history)
                if ok:
                    return True, generation_reports, archive, last_plan, last_planner_model, coder_model
                log_status(f"[hybrid] {report}")

        if round_idx >= refine_rounds:
            round_idx += 1
            continue

        archive_summary = archive.summary_text(limit=3)
        if not archive_summary:
            round_idx += 1
            continue
        refined = generate_refined_plan_candidates(
            planner_models,
            user_prompt,
            prompts["planner"],
            parent_candidates=current_plans,
            beam_width=plan_beam_width,
            archive_summary=archive_summary,
            baseline_code=baseline_code,
        )
        if refined:
            _record_plan_candidates(prompt_history, round_idx + 1, refined, phase="refinement")
            _write_plan_history(out_path, prompt_history)
            log_status(
                f"[planner] experiment-manager refinement {round_idx + 1}/{refine_rounds} "
                f"accepted {len(refined)} strictly improved plan candidates."
            )
            plan_rounds.append(refined)
        else:
            log_status(
                f"[planner] experiment-manager refinement {round_idx + 1}/{refine_rounds} produced no strictly improved plan candidates; "
                "stopping further refinement rounds."
            )
        round_idx += 1

    primary_plan = last_plan or initial_plans[0].plan_text
    primary_planner_model = last_planner_model or initial_plans[0].planner_model
    _write_plan_history(out_path, prompt_history)
    return False, generation_reports, archive, primary_plan, primary_planner_model, None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user-prompt", default="", help="User prompt (inline string).")
    p.add_argument("--user-prompt-file", default=None, help="Path to a text file with the user prompt.")
    p.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=(
            "Comma-separated default model list. Bare names use g4f backend (remote providers). "
            "You can also pass explicit backends like local:<hf_model_id>, ollama:<model>, vllm:<model>, "
            "lmstudio:<model>, g4fapi:<model>, or plain g4f model names like gpt-4o-mini."
        ),
    )
    p.add_argument(
        "--agent-models",
        default=None,
        help=(
            "Optional per-agent override mapping, e.g. 'planner=claude-3.5-sonnet;coder=deepseek-chat,qwen2.5-coder;fixer=gpt-4o-mini'. "
            "Each value accepts the same comma-separated syntax as --models."
        ),
    )
    p.add_argument("--planner-models", default=None, help="Optional model list override for the planner agent.")
    p.add_argument("--coder-models", default=None, help="Optional model list override for the coder agent.")
    p.add_argument("--fixer-models", default=None, help="Optional model list override for the fixer agent.")
    p.add_argument("--custom-prompts", default=None, help="Path to JSON overriding default system prompts.")
    p.add_argument("--out", default=str(Path.cwd() / "generated" / "solve_module.py"), help="Where to write the final solver.")
    p.add_argument("--max-iters", type=int, default=100000, help="Max repair iterations per model candidate.")
    p.add_argument("--search-mode", default=os.getenv("AGENTLAB_SEARCH_MODE", "hybrid"), choices=["classic", "hybrid"], help="classic = single-plan linear search; hybrid = multi-plan frontier search with experiment-memory refinement.")
    p.add_argument("--plan-beam-width", type=int, default=_env_int("AGENTLAB_PLAN_BEAM_WIDTH", 3), help="How many planner hypotheses to keep per planning round in hybrid mode.")
    p.add_argument("--frontier-width", type=int, default=_env_int("AGENTLAB_FRONTIER_WIDTH", 6), help="How many planner/coder frontier attempts to schedule per round in hybrid mode.")
    p.add_argument("--archive-size", type=int, default=_env_int("AGENTLAB_ARCHIVE_SIZE", 6), help="How many failed attempts to retain as experiment-manager memory in hybrid mode.")
    p.add_argument("--refine-rounds", type=int, default=_env_int("AGENTLAB_REFINE_ROUNDS", 1), help="How many planner refinement rounds to run using failure memory in hybrid mode.")
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
    p.add_argument("--g4f-recovery-rounds", type=int, default=None, help="Optional extra recovery rounds before falling back to baseline (default from AGENTLAB_G4F_RECOVERY_ROUNDS or 1).")
    p.add_argument("--g4f-recovery-max-iters", type=int, default=None, help="Optional fixer iterations per recovery round (default from AGENTLAB_G4F_RECOVERY_MAX_ITERS or 2).")
    p.add_argument("--g4f-recovery-sleep", type=float, default=None, help="Optional cooldown in seconds before each recovery round (default from AGENTLAB_G4F_RECOVERY_SLEEP_S or 1.5).")
    p.add_argument("--worker-no-kill-process-group", action="store_true", help="Do not hard-kill the entire worker process group on timeout; only terminate the worker process itself.")
    p.add_argument("--print-generation", action="store_true", help="Print raw model generations for planner/coder/fixer stages.")
    p.add_argument("--print-generation-max-chars", type=int, default=None, help="Maximum number of characters to print per generation (default from AGENTLAB_PRINT_GENERATION_MAX_CHARS or 16000).")
    p.add_argument("--g4f-async", dest="g4f_async", action="store_true", help="Use g4f AsyncClient in the pipeline worker path.")
    p.add_argument("--no-g4f-async", dest="g4f_async", action="store_false", help="Disable g4f AsyncClient and fall back to ChatCompletion.create.")
    p.add_argument("--max-response-chars", type=int, default=None, help="Optional hard cap on captured g4f response size. 0 disables clipping.")
    p.add_argument("--g4f-request-timeout", type=float, default=None, help="Optional timeout passed to g4f requests. Higher values help slower providers.")
    p.add_argument("--attempt-archive-dir", default=None, help="Optional directory where raw prompts, generations, extracted code and strict failure reports are saved per attempt.")
    p.add_argument("--g4f-stop-at-python-fence", dest="g4f_stop_at_python_fence", action="store_true", help="Trim g4f output right after a complete ```python``` fence is received.")
    p.add_argument("--no-g4f-stop-at-python-fence", dest="g4f_stop_at_python_fence", action="store_false", help="Do not trim g4f output at the first python fence.")
    p.set_defaults(g4f_async=None, g4f_stop_at_python_fence=None)
    args = p.parse_args()

    if args.g4f_recovery_rounds is not None:
        os.environ["AGENTLAB_G4F_RECOVERY_ROUNDS"] = str(max(0, int(args.g4f_recovery_rounds)))
    if args.g4f_recovery_max_iters is not None:
        os.environ["AGENTLAB_G4F_RECOVERY_MAX_ITERS"] = str(max(1, int(args.g4f_recovery_max_iters)))
    if args.g4f_recovery_sleep is not None:
        os.environ["AGENTLAB_G4F_RECOVERY_SLEEP_S"] = str(max(0.0, float(args.g4f_recovery_sleep)))
    if args.worker_no_kill_process_group:
        os.environ["AGENTLAB_WORKER_KILL_PROCESS_GROUP"] = "0"
    if args.print_generation:
        os.environ["AGENTLAB_PRINT_GENERATION"] = "1"
    if args.print_generation_max_chars is not None:
        os.environ["AGENTLAB_PRINT_GENERATION_MAX_CHARS"] = str(int(args.print_generation_max_chars))
    if args.g4f_async is not None:
        os.environ["AGENTLAB_G4F_USE_ASYNC"] = "1" if args.g4f_async else "0"
    if args.max_response_chars is not None:
        os.environ["AGENTLAB_MAX_RESPONSE_CHARS"] = str(int(args.max_response_chars))
    if args.g4f_request_timeout is not None:
        os.environ["AGENTLAB_G4F_REQUEST_TIMEOUT_S"] = str(max(0.0, float(args.g4f_request_timeout)))
    if args.attempt_archive_dir:
        os.environ["AGENTLAB_ATTEMPT_ARCHIVE_DIR"] = str(Path(args.attempt_archive_dir).resolve())
    if args.g4f_stop_at_python_fence is not None:
        os.environ["AGENTLAB_G4F_STOP_AT_PYTHON_FENCE"] = "1" if args.g4f_stop_at_python_fence else "0"

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

    planner_models = _prefer_credentialless_models(resolve_agent_models("planner", ordered_models, agent_model_overrides))
    coder_models = _prefer_credentialless_models(resolve_agent_models("coder", ordered_models, agent_model_overrides))
    fixer_models = _prefer_credentialless_models(resolve_agent_models("fixer", coder_models, agent_model_overrides))

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validator_path = Path(args.validator).resolve()

    baseline_path = Path(args.baseline) if args.baseline else (Path.cwd() / "solve_module.py")
    if baseline_path.exists():
        baseline_code = baseline_path.read_text(encoding="utf-8")
    else:
        baseline_code = make_baseline_stub()

    prompt_baseline_code = None if _prompt_requests_from_scratch(user_prompt) else baseline_code
    if prompt_baseline_code is None:
        log_status('[prompt] from-scratch bundle detected: omitting baseline code from planner/coder/fixer prompts.')

    if any(_use_remote_subprocess_isolation(model) for model in set(planner_models + coder_models + fixer_models)):
        log_status('[memory] Remote LLM queries run in isolated subprocesses to keep notebook RAM stable.')
        if os.getenv('AGENTLAB_WORKER_KILL_PROCESS_GROUP', '1').strip().lower() in {'0', 'false', 'no', 'off'}:
            log_status('[memory] Worker timeout cleanup will not kill the entire process group (AGENTLAB_WORKER_KILL_PROCESS_GROUP=0).')

    if _recovery_enabled():
        log_status(
            f"[recovery] enabled: rounds={max(0, _env_int('AGENTLAB_G4F_RECOVERY_ROUNDS', 1))}, "
            f"max_iters={max(1, _env_int('AGENTLAB_G4F_RECOVERY_MAX_ITERS', 2))}, "
            f"sleep_s={max(0.0, _env_float('AGENTLAB_G4F_RECOVERY_SLEEP_S', 1.5)):.1f}"
        )

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

    search_mode, plan_beam_width, frontier_width, archive_size, refine_rounds = _search_settings_from_args(args)
    if search_mode == "hybrid":
        log_status(
            f"[search] hybrid mode enabled: plan_beam_width={plan_beam_width}, frontier_width={frontier_width}, "
            f"archive_size={archive_size}, refine_rounds={refine_rounds}."
        )

    if args.no_llm:
        out_path.write_text(baseline_code, encoding="utf-8")
        log_status(f"[+] Wrote baseline solver to {out_path}")
        sys.exit(0)

    def _fallback_to_baseline(reason: str) -> None:
        log_status(f"[!] {reason}", error=True)
        _archive_json_artifact('strict_failure_report', {
            'ok': False,
            'reason': reason,
            'strict': bool(args.strict),
            'out_path': str(out_path),
            'baseline_path': str(baseline_path),
        })
        if args.strict:
            sys.exit(1)
        out_path.write_text(baseline_code, encoding="utf-8")
        log_status("[!] Falling back to the offline baseline solver.", error=True)
        log_status(f"[+] Wrote baseline solver to {out_path}")
        sys.exit(0)

    tests = resolve_validator_smoke_vectors(validator_path)
    log_status(f"[validate] prepared {len(tests)} smoke vector(s) from {validator_path.parent}")

    generation_reports: List[str] = []
    archive = CandidateArchive(max_items=archive_size)
    plan = ""
    planner_model: Optional[str] = None
    plan_payload: Optional[Dict[str, Any]] = None
    strategy_package: Optional[Dict[str, Any]] = None

    if search_mode == "hybrid":
        hybrid_ok, hybrid_reports, archive, plan, planner_model, winner_model = run_hybrid_codegen_search(
            planner_models=planner_models,
            coder_models=coder_models,
            fixer_models=fixer_models,
            user_prompt=user_prompt,
            prompts=prompts,
            out_path=out_path,
            validator_path=validator_path,
            tests=tests,
            max_iters=args.max_iters,
            baseline_code=prompt_baseline_code,
            plan_beam_width=plan_beam_width,
            frontier_width=frontier_width,
            archive_size=archive_size,
            refine_rounds=refine_rounds,
        )
        generation_reports.extend(hybrid_reports)
        if planner_model:
            log_status(f"[planner] selected anchor plan from model: {planner_model}")
        if hybrid_ok:
            log_status(f"[+] {winner_model}: hybrid frontier validated. Saved to {out_path}")
            sys.exit(0)
        if plan:
            log_status("[search] hybrid frontier exhausted; continuing with a classic sweep anchored on the best surviving plan.")

    if not plan:
        try:
            plan, planner_model, plan_payload, strategy_package = ask_first_structured_plan(
                planner_models,
                user_prompt,
                prompts["planner"],
                baseline_code=prompt_baseline_code,
            )
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

    plan_for_codegen = _augment_plan_with_archive_context(plan, archive)

    model_progress = _make_model_progress(len(coder_models))
    if model_progress is not None:
        model_progress.set_postfix_str(f"model 0/{len(coder_models)}")

    try:
        for idx, model in enumerate(coder_models, start=1):
            if model_progress is not None:
                model_progress.set_postfix_str(f"model {idx}/{len(coder_models)}: {model}")
            log_status(f"[coder] trying model: {model}")
            ok, report = try_generate_with_model(
                model=model,
                fixer_models=fixer_models,
                user_prompt=user_prompt,
                plan=plan_for_codegen,
                prompts=prompts,
                out_path=out_path,
                validator_path=validator_path,
                tests=tests,
                max_iters=args.max_iters,
                baseline_code=prompt_baseline_code,
                plan_payload=plan_payload,
                strategy_package=strategy_package,
            )
            generation_reports.append(report)
            archive.add(
                ArchiveEntry(
                    plan_text=plan,
                    planner_model=planner_model or "planner",
                    coder_model=model,
                    ok=ok,
                    report=report,
                    stage_label="coder",
                    score=_attempt_score(ok, report),
                )
            )
            if model_progress is not None:
                model_progress.update(1)
            if ok:
                log_status(f"[+] {report}. Saved to {out_path}")
                sys.exit(0)
            log_status(f"[coder] {report}")
    finally:
        if model_progress is not None:
            model_progress.close()

    if archive.best_failures(limit=1):
        plan_for_codegen = _augment_plan_with_archive_context(plan, archive)

    baseline_patch_models = resolve_agent_models("baseline-patcher", fixer_models or coder_models, agent_model_overrides)
    if prompt_baseline_code and prompt_baseline_code.strip() and baseline_patch_models:
        patch_iters = max(1, min(args.max_iters, _env_int("AGENTLAB_BASELINE_PATCH_MAX_ITERS", args.max_iters)))
        log_status(
            "[baseline-patcher] attempting a minimal validated patch of the known-good baseline before offline fallback."
        )
        for model in baseline_patch_models:
            log_status(f"[baseline-patcher] trying model: {model}")
            ok, report = try_generate_with_model(
                model=model,
                fixer_models=fixer_models,
                user_prompt=user_prompt,
                plan=plan_for_codegen,
                prompts=prompts,
                out_path=out_path,
                validator_path=validator_path,
                tests=tests,
                max_iters=patch_iters,
                baseline_code=prompt_baseline_code,
                stage_label="baseline-patcher",
                plan_payload=plan_payload,
                strategy_package=strategy_package,
            )
            generation_reports.append(report)
            archive.add(
                ArchiveEntry(
                    plan_text=plan,
                    planner_model=planner_model or "planner",
                    coder_model=model,
                    ok=ok,
                    report=report,
                    stage_label="baseline-patcher",
                    score=_attempt_score(ok, report),
                )
            )
            if ok:
                log_status(f"[+] {report}. Saved to {out_path}")
                sys.exit(0)
            log_status(f"[baseline-patcher] {report}")

    recovered, recovery_report = attempt_recovery_rounds(
        recovery_models=baseline_patch_models or fixer_models or coder_models,
        fixer_models=fixer_models,
        user_prompt=user_prompt,
        plan=plan_for_codegen,
        prompts=prompts,
        out_path=out_path,
        validator_path=validator_path,
        tests=tests,
        max_iters=args.max_iters,
        baseline_code=prompt_baseline_code,
        generation_reports=generation_reports,
    )
    if recovered:
        log_status(f"[+] {recovery_report}. Saved to {out_path}")
        sys.exit(0)

    detail = "\n".join(generation_reports[-8:]).strip()
    reason = "Failed to generate a locally validated solver with the configured models."
    if detail:
        reason = f"{reason}\n{detail}"
    _fallback_to_baseline(reason)


if __name__ == "__main__":
    main()
