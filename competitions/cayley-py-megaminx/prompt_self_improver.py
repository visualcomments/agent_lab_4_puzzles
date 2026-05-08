import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from failure_aware_self_improvement.capability_audit import inspect_solver_code_ast_aware
    from failure_aware_self_improvement.directive_evidence import build_evidence_checks, evidence_prompt_block
    from failure_aware_self_improvement.row_profile_memory import load_row_profile_summary, row_profile_prompt_block
except Exception:
    inspect_solver_code_ast_aware = None
    build_evidence_checks = None
    evidence_prompt_block = None
    load_row_profile_summary = None
    row_profile_prompt_block = None


def _safe_read_text(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _solver_fingerprint(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()[:16]


def _parse_small_int(code: str, name: str) -> Optional[int]:
    pattern = rf"^\s*{re.escape(name)}\s*=\s*([0-9]+)\s*$"
    match = re.search(pattern, code or "", re.MULTILINE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


@dataclass(frozen=True)
class Directive:
    key: str
    title: str
    instruction: str
    rationale: str
    priority: int


DIRECTIVE_LIBRARY: Dict[str, Directive] = {
    "multi_policy_sweep": Directive(
        key="multi_policy_sweep",
        title="Multi-policy candidate sweep",
        instruction=(
            "Introduce a tiny fixed bank of deterministic commuting-order / rewrite policies and choose the shortest valid equivalent candidate per row, "
            "instead of relying on a single normalization order."
        ),
        rationale="Different Megaminx rows compress under different legal commuting orders, so one global order leaves score on the table.",
        priority=100,
    ),
    "score_regression_guard": Directive(
        key="score_regression_guard",
        title="Score-regression guard",
        instruction=(
            "Introduce an explicit rollback / acceptance guard: risky optimization stages must keep the previous exact-valid candidate when they fail to shorten a fixed deterministic evaluation shard or violate exact replay equivalence."
        ),
        rationale="Many prompt rounds are syntactically valid but score-regressive; the model should be forced to design safe fallback logic instead of assuming every new optimizer stage helps.",
        priority=98,
    ),
    "semantic_equivalence_replay": Directive(
        key="semantic_equivalence_replay",
        title="Semantic-equivalence replay",
        instruction=(
            "After each non-trivial rewrite family, replay the optimized local word under official permutations and keep the rewrite only when the exact local effect matches and the word is strictly shorter."
        ),
        rationale="Algorithmic prompting works better when the solver proves semantic preservation programmatically instead of trusting heuristic transformations.",
        priority=95,
    ),
    "bidirectional_window_rewrite": Directive(
        key="bidirectional_window_rewrite",
        title="Bidirectional bounded local rewrite",
        instruction=(
            "Add a stronger bounded local rewrite layer that evaluates both left-to-right and right-to-left exact window replacements with a tiny fixed pass budget."
        ),
        rationale="The current local DP often depends on scan direction; a deterministic bidirectional pass can improve local minima without changing asymptotic guarantees.",
        priority=90,
    ),
    "validator_triad_recheck": Directive(
        key="validator_triad_recheck",
        title="Validator triad recheck",
        instruction=(
            "Harden the final contract with a deterministic three-part check: compile/import health, legal official move-name check, and exact replay-to-final-state verification before returning the candidate result."
        ),
        rationale="For this repository, the best process supervision signal is a strict validator-aware triad.",
        priority=88,
    ),
    "compile_first_then_optimize": Directive(
        key="compile_first_then_optimize",
        title="Compile first, then optimize",
        instruction=(
            "Stage the patch so that the lookup-first baseline path stays intact and import-safe before adding heavier optimizer deltas; when a later stage is uncertain, fail back to the exact baseline-compatible path rather than breaking the whole solver."
        ),
        rationale="Rounds that mix architectural rewrites and fragile plumbing often fail for avoidable reasons; forcing a compile-safe baseline-preserving layer improves outer-loop efficiency.",
        priority=86,
    ),
    "policy_ablation_search": Directive(
        key="policy_ablation_search",
        title="Policy ablation search",
        instruction=(
            "Use a tiny fixed ablation bank of optimizer policy packages and keep per-row or per-window winners under a deterministic local score proxy, rather than betting on a single composite heuristic."
        ),
        rationale="APE-style prompt search is a strong fit for this repository because different deterministic policy mixes win on different bundled rows.",
        priority=84,
    ),
    "small_support_macro_mining": Directive(
        key="small_support_macro_mining",
        title="Small-support commutator/conjugate mining",
        instruction=(
            "Mine only bounded short commutators/conjugates with small official permutation support and use them strictly as exact local rewrite candidates, never as fake move names."
        ),
        rationale="Megaminx local word compression benefits from structured short effects that do not appear in a plain same-face / commuting rewrite atlas.",
        priority=80,
    ),
    "stronger_effect_atlas": Directive(
        key="stronger_effect_atlas",
        title="Stronger exact short-word atlas",
        instruction=(
            "Strengthen the fixed-radius exact effect atlas: keep the shortest representative per local effect, canonicalize equivalent short words before storage, and memoize packed effects."
        ),
        rationale="A weak exact atlas makes the downstream optimizer blind to many safe equivalences already available at small radius.",
        priority=72,
    ),
    "candidate_bank_scoring": Directive(
        key="candidate_bank_scoring",
        title="Deterministic candidate bank scoring",
        instruction=(
            "Construct a small deterministic bank of optimizer variants, score them with a local exact-validity-preserving proxy, and keep only the best valid result."
        ),
        rationale="The solver itself should exploit a bounded candidate bank instead of committing to a single rewrite schedule.",
        priority=65,
    ),
    "prompt_population_search": Directive(
        key="prompt_population_search",
        title="Prompt-population search",
        instruction=(
            "Treat this round as one member of a small prompt population instead of a single linear patch chain: produce a candidate that is deliberately differentiated from recent accepted rounds in search policy, acceptance guard design, and candidate-ranking logic."
        ),
        rationale="Promptbreeder-style diversity is a better fit for algorithmic plateau breaking than repeatedly paraphrasing the last accepted prompt.",
        priority=110,
    ),
    "exact_evaluator_shard": Directive(
        key="exact_evaluator_shard",
        title="Exact evaluator shard",
        instruction=(
            "Introduce or strengthen a tiny exact evaluation shard inside the generated solver workflow: every risky optimizer stage must justify itself against a fixed deterministic replay-based micro-benchmark before replacing the previous exact-valid candidate."
        ),
        rationale="Algorithmic improvement should be driven by executable validation and score deltas, not by prose confidence alone.",
        priority=108,
    ),
    "solver_archive_lineage": Directive(
        key="solver_archive_lineage",
        title="Solver archive and lineage",
        instruction=(
            "Preserve explicit lineage and candidate provenance in the design: keep track of baseline, patch lane, fresh lane, acceptance metric, and why the winning candidate survived."
        ),
        rationale="AlphaEvolve-style improvement works better when each accepted candidate is grounded in verifiable lineage rather than hidden prompt drift.",
        priority=104,
    ),
    "patch_fresh_lane_split": Directive(
        key="patch_fresh_lane_split",
        title="Patch/fresh lane split",
        instruction=(
            "Ask for two bounded candidate lanes: a minimal patch over the best validated solver and a deliberately fresh candidate that preserves the public contract but is not anchored to the same local rewrite ordering."
        ),
        rationale="A mixed patch-vs-fresh proposal bank reduces premature convergence while keeping the repository's contract-safe bias.",
        priority=102,
    ),
    "pareto_candidate_selection": Directive(
        key="pareto_candidate_selection",
        title="Pareto candidate selection",
        instruction=(
            "Select winners using a simple Pareto-style view over exact validity, score delta, runtime risk, and novelty instead of a one-dimensional preference for whichever patch sounds strongest."
        ),
        rationale="Megaminx prompt evolution should trade off score, safety, and diversity explicitly when search begins to plateau.",
        priority=101,
    ),
    "portfolio_orchestration": Directive(
    key="portfolio_orchestration",
    title="Portfolio orchestration",
    instruction=(
        "Design the solver or outer loop as a bounded multi-lane portfolio: exact row-level scoring, row-wise best-of-lanes fusion, lineage logging, and promotion only when a lane wins on exact bundled move count."
    ),
    rationale="Megaminx is already an asset-first pipeline, so portfolio selection across safe lanes is more valuable than a single monolithic heuristic.",
    priority=107,
),
"hard_row_routing": Directive(
    key="hard_row_routing",
    title="Hard-row routed search",
    instruction=(
        "Build an explicit row difficulty profile and spend aggressive search budget only on the hardest rows first; optimize for saved moves per CPU-hour, not uniform per-row effort."
    ),
    rationale="The current bundled score is dominated by a hard tail, so routing search by difficulty is a better use of bounded compute.",
    priority=106,
),
"exact_metric_acceptance": Directive(
    key="exact_metric_acceptance",
    title="Exact metric acceptance",
    instruction=(
        "Treat acceptance as an exact bundled-score problem: a refinement must beat the incumbent on a fixed deterministic dev shard with zero validator regressions before it can replace the parent."
    ),
    rationale="A prompt that sounds stronger but does not shorten exact paths should not be promoted in a move-count competition.",
    priority=109,
),
"shadow_split_benchmarking": Directive(
    key="shadow_split_benchmarking",
    title="Shadow split benchmarking",
    instruction=(
        "Keep separate deterministic train/dev/holdout shards for prompt and code evolution, and report exact scores on each split so that the outer loop does not overfit the whole bundled set."
    ),
    rationale="Exact-score search over a fixed bundle needs a lightweight anti-overfitting protocol, especially when prompts mutate aggressively.",
    priority=103,
),

"no_novelty_rejection": Directive(
    key="no_novelty_rejection",
    title="No-novelty rejection",
    instruction=(
        "Produce a candidate that changes a real optimization lane, not only comments, wrappers, lookup replay, or optimized_submission copying; the outer loop rejects identical solver hashes and identical submission digests."
    ),
    rationale="The failed sweep repeatedly produced byte-identical scripts and identical submissions, so prompt evolution must explicitly demand measurable novelty.",
    priority=112,
),
"per_row_delta_acceptance": Directive(
    key="per_row_delta_acceptance",
    title="Per-row delta acceptance",
    instruction=(
        "Design the solver so its contribution can be evaluated row-by-row; aim for at least one exact-valid improved row with zero regressions, and make rollback deterministic for unchanged/regressed rows."
    ),
    rationale="A full-score tie hides whether the candidate made any useful row-level contribution; per-row deltas make improvement executable and auditable.",
    priority=111,
),
"provider_preflight_no_fallback": Directive(
    key="provider_preflight_no_fallback",
    title="Provider preflight and no fallback promotion",
    instruction=(
        "Do not convert provider failure, timeout, sample-submission fallback, or offline baseline recovery into a successful solver; fail explicitly when generation did not produce a new exact-valid algorithm."
    ),
    rationale="The failed archive contained rc=0 fallback artifacts from provider failures; these must remain failed attempts and should not guide algorithmic promotion.",
    priority=110,
),


"failure_autopsy_repair_loop": Directive(
    key="failure_autopsy_repair_loop",
    title="Failure-autopsy repair loop",
    instruction=(
        "When the previous generated solver failed compile/import/validator/submission checks, inspect that failed candidate as a supervised negative example: classify the failure, preserve any promising algorithmic delta, then produce a minimal repair that passes the same gate before adding any new optimization layer."
    ),
    rationale="A failed candidate still contains useful search signal; the next prompt should learn from the exact failure instead of resetting to another broad rewrite.",
    priority=118,
),
"compile_validation_ladder": Directive(
    key="compile_validation_ladder",
    title="Compile-validation ladder",
    instruction=(
        "Stage the next candidate through a strict ladder: first py_compile/import safety, then solve(vec) contract and JSON stdout, then official move-name legality, then exact replay validation, and only then score/novelty improvement."
    ),
    rationale="Most failed code rounds waste tokens on optimizer ambition before restoring basic executable correctness; a ladder turns failures into progressively stronger working code.",
    priority=117,
),
"delta_preserving_repair": Directive(
    key="delta_preserving_repair",
    title="Delta-preserving repair",
    instruction=(
        "Do not discard a failed candidate wholesale if it introduced a useful exact-rewrite, portfolio, or replay-guard idea; isolate the unsafe fragment, keep the safe delta, and repair the smallest broken surface."
    ),
    rationale="The best improvement path often comes from rescuing a partially correct algorithmic delta rather than generating an unrelated fresh solver every time.",
    priority=116,
),
"minimal_working_then_improve": Directive(
    key="minimal_working_then_improve",
    title="Minimal working solver, then improvement",
    instruction=(
        "If the previous code failed validation, first return to a minimal exact-valid lookup-first solver shell with the new delta behind a rollback-safe guard; after that, add one bounded measurable improvement rather than a large unvalidated rewrite."
    ),
    rationale="This forces monotonic movement toward working code while still requiring a real improvement beyond fallback or copied baseline behavior.",
    priority=115,
),

}


def inspect_solver_code(code: str) -> Dict[str, Any]:
    if inspect_solver_code_ast_aware is not None:
        return inspect_solver_code_ast_aware(code)
    low = (code or "").lower()
    constants = {
        "short_table_depth": _parse_small_int(code, "_SHORT_TABLE_DEPTH"),
        "local_window": _parse_small_int(code, "_LOCAL_WINDOW"),
        "optimization_passes": _parse_small_int(code, "_OPTIMIZATION_PASSES"),
    }
    return {
        "inspection_mode": "token_fallback",
        "uses_lookup_first": "optimized_lookup" in low or "lookup.get(state_key)" in low or "_best_lookup" in low or "_lookup_cache" in low,
        "has_exact_short_atlas": "_short_word_data" in low or "short_table_depth" in low,
        "has_local_window_dp": "_optimize_local_windows" in low or ("dp =" in low and "nxt =" in low),
        "has_multi_policy_sweep": any(token in low for token in ("policy", "policies", "candidate optimizers", "candidate bank", "best-of-fixed-candidates", "first-occurrence", "last-occurrence")),
        "has_bidirectional_rewrite": any(token in low for token in ("right-to-left", "right_first", "reverse pass", "reversed(", "left-first", "right-first")),
        "has_macro_mining": any(token in low for token in ("commutator", "conjugate", "macro atlas", "small_support", "small-support")),
        "has_candidate_bank_scoring": any(token in low for token in ("candidate optimizers", "local score proxy", "best_candidate", "candidate_metric", "score proxy")),
        "runtime_probe": {},
        "fingerprint": _solver_fingerprint(code),
        "constants": constants,
    }


def _recent_directives(prompt_history: Sequence[Dict[str, Any]], *, limit: int = 2) -> set[str]:
    used: set[str] = set()
    for entry in list(prompt_history)[-max(0, limit):]:
        for item in entry.get("selected_directives", []) or []:
            key = str(item or "").strip()
            if key:
                used.add(key)
    return used


def _entry_failure_text(entry: Dict[str, Any]) -> str:
    """Return a normalized failure signal from all fields a runner may emit.

    Older logic only inspected ``error``. Megaminx prompt sweeps often reject a
    candidate without a traceback: identical solver, no per-row improvement,
    score regression, or provider fallback. Those are still failures for prompt
    evolution and must select repair directives.
    """
    parts: List[str] = []
    for key in (
        "error",
        "failure_kind",
        "rejection_reason",
        "status",
        "provider_status",
        "fallback_reason",
    ):
        value = entry.get(key)
        if value:
            parts.append(str(value))

    reasons = entry.get("rejection_reasons")
    if isinstance(reasons, list):
        parts.extend(str(item) for item in reasons if item)
    elif reasons:
        parts.append(str(reasons))

    metric = entry.get("metric") if isinstance(entry.get("metric"), dict) else {}
    if metric:
        for key in ("source", "name", "status", "reason"):
            if metric.get(key):
                parts.append(str(metric.get(key)))

    report = entry.get("novelty_report")
    if isinstance(report, dict):
        if report.get("identical_solver"):
            parts.append("identical_solver")
        if report.get("identical_submission"):
            parts.append("identical_submission")
        if report.get("copied_baseline"):
            parts.append("copied_baseline")
        delta = report.get("per_row_delta") if isinstance(report.get("per_row_delta"), dict) else {}
        if delta:
            improved = delta.get("improved_rows")
            regressed = delta.get("regressed_rows")
            net_delta = delta.get("net_delta_moves")
            parts.append(f"per_row_delta improved={improved} regressed={regressed} net_delta_moves={net_delta}")
            if improved in (0, "0"):
                parts.append("zero improved rows")
            try:
                if net_delta is not None and float(net_delta) >= 0:
                    parts.append("non_negative_net_delta_moves")
            except Exception:
                pass

    return " ; ".join(parts).lower()


def _error_bucket(entry: Dict[str, Any]) -> str | None:
    text = _entry_failure_text(entry)
    if not text:
        return None
    if any(token in text for token in ("json", "decode", "strict output", "fence", "envelope")):
        return "json_contract"
    if any(token in text for token in ("illegal move", "unknown move", "invalid move", "fake_move", "move-name")):
        return "illegal_move"
    if any(token in text for token in ("replay mismatch", "final state", "unsolved", "central_state", "target mismatch", "exact replay")):
        return "replay_mismatch"
    if any(token in text for token in ("syntax", "compile", "py_compile", "importerror", "modulenotfounderror", "nameerror")):
        return "compile_or_import"
    if any(token in text for token in ("traceback", "exception", "typeerror", "valueerror", "runtime")):
        return "runtime"
    if any(token in text for token in ("no_per_row_improvement", "zero improved rows", "per_row_delta improved=0")):
        return "no_per_row_improvement"
    if any(token in text for token in ("no_novelty", "identical_solver", "identical_submission", "copied_baseline", "same submission")):
        return "no_novelty"
    if any(token in text for token in ("score regression", "regressed rows", "non_negative_net_delta_moves", "worse score", "score_delta")):
        return "score_regression"
    if any(token in text for token in ("credentials required", "provider", "timeout", "offline fallback", "sample_submission fallback", "fallback artifact")):
        return "provider_or_fallback"
    if any(token in text for token in ("validator", "validation", "invalid", "failed gate")):
        return "validation"
    return "other"


def analyze_history_signals(history: Sequence[Dict[str, Any]], *, limit: int = 6) -> Dict[str, Any]:
    window = list(history)[-max(0, limit):]
    signals: Dict[str, Any] = {
        "window": len(window),
        "recent_accepts": 0,
        "recent_failures": 0,
        "json_contract_failures": 0,
        "validation_failures": 0,
        "illegal_move_failures": 0,
        "replay_mismatch_failures": 0,
        "compile_or_import_failures": 0,
        "runtime_failures": 0,
        "other_failures": 0,
        "no_novelty_failures": 0,
        "no_per_row_improvement_failures": 0,
        "provider_or_fallback_failures": 0,
        "validated_not_selected": 0,
        "plateau": False,
        "score_regressions": 0,
        "recent_failure_labels": [],
        "accepted_rounds": 0,
        "accepted_metric_values": [],
        "accepted_metric_plateau": False,
        "solver_hash_novelty_count": 0,
        "submission_hash_novelty_count": 0,
        "identical_solver_rejections": 0,
        "identical_submission_rejections": 0,
        "recent_improved_rows": 0,
        "recent_regressed_rows": 0,
        "recent_net_delta_moves": 0.0,
        "multiaxis_plateau": False,
        "stagnation_axes": [],
    }
    if not window:
        return signals

    labels: List[str] = []
    for entry in window:
        if entry.get("accepted") is True:
            signals["recent_accepts"] += 1
            signals["accepted_rounds"] += 1
            metric = entry.get("metric") if isinstance(entry.get("metric"), dict) else {}
            if metric.get("source") == "local_score" and metric.get("value") is not None:
                signals["accepted_metric_values"].append(metric.get("value"))
            continue
        bucket = _error_bucket(entry)
        if bucket is not None:
            signals["recent_failures"] += 1
            labels.append(bucket)
            if bucket == "json_contract":
                signals["json_contract_failures"] += 1
            elif bucket in ("validation", "illegal_move", "replay_mismatch"):
                signals["validation_failures"] += 1
                if bucket == "illegal_move":
                    signals["illegal_move_failures"] += 1
                if bucket == "replay_mismatch":
                    signals["replay_mismatch_failures"] += 1
            elif bucket in ("compile_or_import", "runtime"):
                signals["runtime_failures"] += 1
                if bucket == "compile_or_import":
                    signals["compile_or_import_failures"] += 1
            elif bucket in ("no_novelty", "no_per_row_improvement"):
                signals["no_novelty_failures"] += 1
                if bucket == "no_per_row_improvement":
                    signals["no_per_row_improvement_failures"] += 1
            elif bucket == "score_regression":
                signals["score_regressions"] += 1
            elif bucket == "provider_or_fallback":
                signals["provider_or_fallback_failures"] += 1
            else:
                signals["other_failures"] += 1
            continue
        signals["validated_not_selected"] += 1
        metric = entry.get("metric") if isinstance(entry.get("metric"), dict) else {}
        if metric.get("source") == "local_score" and metric.get("value") is not None:
            signals["score_regressions"] += 1

    seen_solver_digests: set[str] = set()
    seen_submission_digests: set[str] = set()
    for entry in window:
        report = entry.get("novelty_report") if isinstance(entry.get("novelty_report"), dict) else {}
        solver_digest = report.get("candidate_solver_sha") or report.get("candidate_solver_digest") or entry.get("candidate_solver_sha")
        submission_digest = report.get("candidate_submission_digest") or report.get("candidate_submission_sha") or entry.get("candidate_submission_digest")
        if solver_digest:
            seen_solver_digests.add(str(solver_digest))
        if submission_digest:
            seen_submission_digests.add(str(submission_digest))
        if report.get("identical_solver"):
            signals["identical_solver_rejections"] += 1
        if report.get("identical_submission"):
            signals["identical_submission_rejections"] += 1
        delta = report.get("per_row_delta") if isinstance(report.get("per_row_delta"), dict) else {}
        try:
            signals["recent_improved_rows"] += int(delta.get("improved_rows") or 0)
        except Exception:
            pass
        try:
            signals["recent_regressed_rows"] += int(delta.get("regressed_rows") or 0)
        except Exception:
            pass
        try:
            signals["recent_net_delta_moves"] += float(delta.get("net_delta_moves") or 0.0)
        except Exception:
            pass
    signals["solver_hash_novelty_count"] = len(seen_solver_digests)
    signals["submission_hash_novelty_count"] = len(seen_submission_digests)
    stagnation_axes: List[str] = []
    if signals.get("recent_failures") and signals.get("recent_improved_rows") == 0:
        stagnation_axes.append("zero_improved_rows")
    if signals.get("identical_solver_rejections"):
        stagnation_axes.append("identical_solver")
    if signals.get("identical_submission_rejections"):
        stagnation_axes.append("identical_submission")
    if signals.get("provider_or_fallback_failures"):
        stagnation_axes.append("provider_or_fallback")
    if signals.get("recent_regressed_rows"):
        stagnation_axes.append("row_regressions")
    signals["stagnation_axes"] = stagnation_axes
    signals["multiaxis_plateau"] = bool(len(stagnation_axes) >= 2 or (signals.get("recent_failures", 0) >= 2 and signals.get("recent_improved_rows") == 0))

    tail = window[-3:]
    if tail and all(entry.get("accepted") is False for entry in tail):
        signals["plateau"] = True
    accepted_metric_values = [float(v) for v in signals.get("accepted_metric_values") or [] if v is not None]
    if len(accepted_metric_values) >= 2 and max(accepted_metric_values) - min(accepted_metric_values) <= 1.0:
        signals["accepted_metric_plateau"] = True
    signals["recent_failure_labels"] = labels[-3:]
    return signals


def _safe_excerpt(text: str, *, max_chars: int = 6000) -> str:
    raw = str(text or "")
    if len(raw) <= max_chars:
        return raw
    head = max_chars // 2
    tail = max_chars - head
    return raw[:head].rstrip() + "\n\n...[truncated failed candidate code]...\n\n" + raw[-tail:].lstrip()


def _entry_rejection_text(entry: Dict[str, Any]) -> str:
    parts: List[str] = []
    if entry.get("error"):
        parts.append(str(entry.get("error")))
    for key in ("failure_kind", "rejection_reason", "rejection_reasons", "fallback_reason"):
        value = entry.get(key)
        if isinstance(value, list):
            parts.extend(str(x) for x in value if x)
        elif value:
            parts.append(str(value))
    novelty = _entry_novelty_summary(entry)
    if novelty:
        parts.append(novelty)
    return "; ".join(parts)


def _entry_novelty_summary(entry: Dict[str, Any]) -> str:
    report = entry.get("novelty_report")
    if not isinstance(report, dict):
        return ""
    parts: List[str] = []
    for key in ("identical_solver", "identical_submission", "copied_baseline"):
        if report.get(key):
            parts.append(key)
    delta = report.get("per_row_delta") if isinstance(report.get("per_row_delta"), dict) else {}
    if delta:
        parts.append(
            "per_row_delta "
            f"improved={delta.get('improved_rows')} regressed={delta.get('regressed_rows')} "
            f"net_delta_moves={delta.get('net_delta_moves')}"
        )
    for key in ("rows_improved", "rows_regressed", "total_saved_moves", "score_delta"):
        if key in report:
            parts.append(f"{key}={report.get(key)}")
    return "; ".join(parts)


def _last_failed_entry(history: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for entry in reversed(list(history or [])):
        if entry.get("accepted") is True:
            continue
        if _error_bucket(entry) is not None:
            return entry
        if entry.get("metric") or entry.get("novelty_report"):
            return entry
    return None


def _read_failed_candidate_code(entry: Dict[str, Any]) -> tuple[str, Optional[Path]]:
    for key in ("path", "solver_path", "candidate_path", "generated_solver_path", "file"):
        path_text = str(entry.get(key) or "").strip()
        if not path_text:
            continue
        path = Path(path_text)
        code = _safe_read_text(path)
        if code:
            return code, path
    return "", None


def _inspect_failed_candidate_code(code: str) -> Dict[str, Any]:
    low = (code or "").lower()
    return {
        "fingerprint": _solver_fingerprint(code),
        "has_solve_entrypoint": "def solve(" in low,
        "has_script_entrypoint": "__main__" in low and "json" in low,
        "has_official_move_guard": any(token in low for token in ("puzzle_info", "allowed_moves", "legal_moves", "official")),
        "has_replay_guard": any(token in low for token in ("replay", "apply_move", "final_state", "central_state")),
        "has_rollback_guard": any(token in low for token in ("rollback", "fallback", "keep previous", "best_candidate", "candidate")),
        "has_optimizer_delta": any(token in low for token in ("window", "rewrite", "commutator", "conjugate", "policy", "portfolio", "atlas", "candidate_bank")),
        "has_lane_trace": any(token in low for token in ("solve_with_trace", "selected_lane", "lane", "baseline_len", "candidate_len")),
        "line_count": len((code or "").splitlines()),
    }


def _repair_strategy_for_bucket(bucket: str, changed_from_baseline: bool) -> List[str]:
    bucket = str(bucket or "other")
    common = [
        "Keep a lookup-first rollback path and promote only exact-replay-valid row-level deltas.",
        "Add trace metadata for selected_lane, baseline_len, candidate_len, validity, and rollback_reason.",
    ]
    if bucket == "json_contract":
        return [
            "Repair only the JSON/code-envelope contract first; do not rewrite solver logic while the envelope is broken.",
            "Return exactly the expected strict output and keep imports / entrypoints minimal.",
        ] + common
    if bucket in ("compile_or_import", "runtime"):
        return [
            "Run a compile/import mental pass before adding optimizer ambition.",
            "Preserve only safe deltas from the failed candidate; otherwise fall back to a minimal working shell plus one guarded lane.",
        ] + common
    if bucket == "illegal_move":
        return [
            "Constrain every emitted token to the official move set; never invent move names.",
            "Add a move-legality gate before score counting and rollback any lane that emits an unknown token.",
        ] + common
    if bucket in ("replay_mismatch", "validation"):
        return [
            "Treat shorter paths as worthless unless exact replay reaches the required target state for that row.",
            "Validate every new rewrite family on deterministic shadow rows before it can replace lookup output.",
        ] + common
    if bucket in ("no_novelty", "no_per_row_improvement"):
        prefix = [
            "Previous candidate produced no useful row-level novelty; do not copy or paraphrase the incumbent.",
            "Create exactly one fresh isolated lane that must improve at least one row and regress zero rows before it can fire.",
            "Prefer tiny row-local replacement tables or bounded window rewrites over broad architecture rewrites.",
        ]
        if not changed_from_baseline:
            prefix.append("Suppress baseline-sized failed-code reuse: use the failure only as a no-novelty diagnosis, not as code to imitate.")
        return prefix + common
    if bucket == "score_regression":
        return [
            "A candidate may improve some rows but still lose globally; require per-row Pareto acceptance and row-level rollback.",
            "Log top regressed rows and disable the responsible lane unless it has positive net saved moves with no invalid rows.",
        ] + common
    if bucket == "provider_or_fallback":
        return [
            "Do not promote provider fallback, sample_submission fallback, or credential-recovery artifacts as solver improvements.",
            "Add provider preflight reporting and fail closed when no real generated solver is available.",
        ] + common
    return [
        "Classify the failure before changing code; choose the smallest repair that restores executable correctness and novelty.",
    ] + common


def _failure_repair_report(history: Sequence[Dict[str, Any]], baseline_code: str) -> Dict[str, Any]:
    entry = _last_failed_entry(history)
    if entry is None:
        return {"available": False}
    code, path = _read_failed_candidate_code(entry)
    bucket = _error_bucket(entry) or str(entry.get("failure_kind") or "rejected")
    rejection_text = _entry_rejection_text(entry)
    features = _inspect_failed_candidate_code(code) if code else {}
    baseline_fp = _solver_fingerprint(baseline_code)
    candidate_fp = features.get("fingerprint")
    novelty_summary = _entry_novelty_summary(entry)
    changed_from_baseline = bool(candidate_fp and candidate_fp != baseline_fp)
    suppress_code_excerpt = bucket in ("no_novelty", "no_per_row_improvement") and not changed_from_baseline
    return {
        "available": True,
        "round": entry.get("round"),
        "path": str(path) if path is not None else None,
        "bucket": bucket,
        "error": rejection_text,
        "novelty_summary": novelty_summary,
        "candidate_code_available": bool(code),
        "baseline_fingerprint": baseline_fp,
        "candidate_fingerprint": candidate_fp,
        "changed_from_baseline": changed_from_baseline,
        "suppress_code_excerpt": suppress_code_excerpt,
        "features": features,
        "repair_strategy": _repair_strategy_for_bucket(bucket, changed_from_baseline),
        "code_excerpt": "" if suppress_code_excerpt else (_safe_excerpt(code) if code else ""),
    }


def _failure_repair_block_from_report(report: Dict[str, Any]) -> str:
    if not report.get("available"):
        return "Failure-aware repair memory:\n- no failed generated candidate has been observed yet"
    features = report.get("features") if isinstance(report.get("features"), dict) else {}
    repair_strategy = report.get("repair_strategy") if isinstance(report.get("repair_strategy"), list) else []
    lines = [
        "Failure-aware repair memory:",
        f"- failed_round: {report.get('round')}",
        f"- failure_bucket: {report.get('bucket')}",
        f"- failure_error_or_rejection: {report.get('error') or 'not recorded'}",
        f"- novelty_summary: {report.get('novelty_summary') or 'not recorded'}",
        f"- failed_candidate_path: {report.get('path') or 'not recorded'}",
        f"- baseline_fingerprint: {report.get('baseline_fingerprint')}",
        f"- failed_candidate_fingerprint: {report.get('candidate_fingerprint') or 'unavailable'}",
        f"- changed_from_baseline: {'yes' if report.get('changed_from_baseline') else 'no'}",
        f"- suppress_code_excerpt: {'yes' if report.get('suppress_code_excerpt') else 'no'}",
        "- failed_candidate_features: "
        + (", ".join(f"{k}={v}" for k, v in features.items() if k != "fingerprint") or "unavailable"),
        "Repair policy for the next solver:",
    ]
    for idx, item in enumerate(repair_strategy, start=1):
        lines.append(f"{idx}. {item}")
    excerpt = str(report.get("code_excerpt") or "").strip()
    if excerpt:
        lines.extend([
            "Failed candidate code excerpt for targeted repair:",
            excerpt,
        ])
    return "\n".join(lines)


def _failure_repair_block(history: Sequence[Dict[str, Any]], baseline_code: str) -> str:
    return _failure_repair_block_from_report(_failure_repair_report(history, baseline_code))


def _compact_failure_context(report: Dict[str, Any]) -> str:
    if not isinstance(report, dict) or not report.get("available"):
        return "no failed generated candidate has been observed yet"
    strategy = report.get("repair_strategy") if isinstance(report.get("repair_strategy"), list) else []
    strategy_text = " | ".join(str(item) for item in strategy[:3])
    return (
        f"failure_bucket={report.get('bucket')}; "
        f"changed_from_baseline={'yes' if report.get('changed_from_baseline') else 'no'}; "
        f"novelty={report.get('novelty_summary') or 'not recorded'}; "
        f"repair={strategy_text or 'classify and repair minimally'}"
    )


def select_directives(
    *,
    feature_snapshot: Dict[str, Any],
    round_idx: int,
    prompt_history: Sequence[Dict[str, Any]],
    score_history: Sequence[Dict[str, Any]],
) -> List[Directive]:
    candidates: Dict[str, Directive] = {}
    constants = feature_snapshot.get("constants") if isinstance(feature_snapshot.get("constants"), dict) else {}
    history_signals = analyze_history_signals(score_history)

    def add(key: str) -> None:
        candidates[key] = DIRECTIVE_LIBRARY[key]

    if not feature_snapshot.get("has_multi_policy_sweep"):
        add("multi_policy_sweep")
    if not feature_snapshot.get("has_bidirectional_rewrite"):
        add("bidirectional_window_rewrite")
    if not feature_snapshot.get("has_macro_mining"):
        add("small_support_macro_mining")
    if not feature_snapshot.get("has_exact_short_atlas") or int(constants.get("short_table_depth") or 0) <= 5:
        add("stronger_effect_atlas")
    if not feature_snapshot.get("has_candidate_bank_scoring"):
        add("candidate_bank_scoring")

    plateau = bool(history_signals.get("plateau") or history_signals.get("accepted_metric_plateau") or history_signals.get("validated_not_selected"))
    if plateau or round_idx >= 2:
        add("prompt_population_search")
        add("exact_evaluator_shard")
        add("provider_preflight_no_fallback")
        add("solver_archive_lineage")
        add("patch_fresh_lane_split")
        add("pareto_candidate_selection")
        add("portfolio_orchestration")
        add("exact_metric_acceptance")
        add("shadow_split_benchmarking")
        add("no_novelty_rejection")
        add("per_row_delta_acceptance")

    if history_signals.get("plateau"):
        add("policy_ablation_search")
        add("score_regression_guard")
        add("semantic_equivalence_replay")
        add("hard_row_routing")
        add("no_novelty_rejection")
        add("per_row_delta_acceptance")
        add("portfolio_orchestration")
        add("exact_metric_acceptance")
    if history_signals.get("validation_failures") or history_signals.get("runtime_failures"):
        add("failure_autopsy_repair_loop")
        add("compile_validation_ladder")
        add("delta_preserving_repair")
        add("minimal_working_then_improve")
        add("compile_first_then_optimize")
        add("validator_triad_recheck")
        add("exact_evaluator_shard")
    if history_signals.get("no_novelty_failures") or history_signals.get("no_per_row_improvement_failures"):
        add("failure_autopsy_repair_loop")
        add("minimal_working_then_improve")
        add("patch_fresh_lane_split")
        add("no_novelty_rejection")
        add("per_row_delta_acceptance")
        add("pareto_candidate_selection")
        add("exact_metric_acceptance")
        add("score_regression_guard")
    if history_signals.get("provider_or_fallback_failures"):
        add("provider_preflight_no_fallback")
        add("failure_autopsy_repair_loop")
        add("compile_validation_ladder")
    if history_signals.get("json_contract_failures"):
        add("failure_autopsy_repair_loop")
        add("compile_validation_ladder")
        add("minimal_working_then_improve")
        add("validator_triad_recheck")
        add("compile_first_then_optimize")
    if history_signals.get("score_regressions"):
        add("score_regression_guard")
        add("semantic_equivalence_replay")
        add("pareto_candidate_selection")

    if not candidates:
        for key in ("multi_policy_sweep", "bidirectional_window_rewrite", "stronger_effect_atlas"):
            add(key)

    ordered = sorted(candidates.values(), key=lambda item: (-item.priority, item.key))
    used_recently = _recent_directives(prompt_history)
    fresh = [item for item in ordered if item.key not in used_recently]
    core_required: list[str] = []
    for key in ("multi_policy_sweep", "bidirectional_window_rewrite", "small_support_macro_mining"):
        if key in candidates:
            core_required.append(key)
    if history_signals.get("recent_failures"):
        for key in ("failure_autopsy_repair_loop", "compile_validation_ladder", "minimal_working_then_improve"):
            if key in candidates:
                core_required.append(key)
    if history_signals.get("no_novelty_failures") or history_signals.get("no_per_row_improvement_failures"):
        for key in ("no_novelty_rejection", "per_row_delta_acceptance", "patch_fresh_lane_split"):
            if key in candidates:
                core_required.append(key)
    if history_signals.get("score_regressions") or history_signals.get("plateau"):
        core_required.append("score_regression_guard")
        if "semantic_equivalence_replay" in candidates:
            core_required.append("semantic_equivalence_replay")
        elif "policy_ablation_search" in candidates:
            core_required.append("policy_ablation_search")
    target_size = 8 if (plateau or history_signals.get("recent_failures") or round_idx >= 2) else 4
    target_size = max(target_size, len(core_required))
    selected = fresh[:target_size]
    if len(selected) < target_size:
        for item in ordered:
            if item not in selected:
                selected.append(item)
            if len(selected) >= target_size:
                break
    for key in core_required:
        directive = DIRECTIVE_LIBRARY[key]
        if directive in selected:
            continue
        if len(selected) < target_size:
            selected.append(directive)
            continue
        replaced = False
        for idx in range(len(selected) - 1, -1, -1):
            if selected[idx].key not in core_required:
                selected[idx] = directive
                replaced = True
                break
        if not replaced:
            selected.append(directive)
    deduped: list[Directive] = []
    seen_keys: set[str] = set()
    for item in selected:
        if item.key in seen_keys:
            continue
        deduped.append(item)
        seen_keys.add(item.key)
    return deduped


def summarize_history(history: Sequence[Dict[str, Any]], *, limit: int = 3) -> List[str]:
    out: List[str] = []
    for entry in list(history)[-max(0, limit):]:
        round_idx = entry.get("round")
        if entry.get("accepted") is True:
            metric = entry.get("metric") if isinstance(entry.get("metric"), dict) else {}
            source = metric.get("source")
            value = metric.get("value")
            if source is not None and value is not None:
                out.append(f"round {round_idx}: accepted via {source}={value}")
            else:
                out.append(f"round {round_idx}: accepted validated solver")
            continue
        if entry.get("error"):
            out.append(f"round {round_idx}: rejected after failure {entry.get('error')}")
            continue
        metric = entry.get("metric") if isinstance(entry.get("metric"), dict) else {}
        metric_desc = f" {metric.get('source')}={metric.get('value')}" if metric else ""
        out.append(f"round {round_idx}: validated but not selected{metric_desc}")
    return out


def _insert_before_strict_contract(base_text: str, addition: str) -> str:
    marker = "STRICT OUTPUT REQUIREMENTS"
    idx = base_text.find(marker)
    if idx < 0:
        return (base_text.rstrip() + "\n\n" + addition.strip() + "\n").strip() + "\n"
    prefix = base_text[:idx].rstrip()
    suffix = base_text[idx:].lstrip()
    return prefix + "\n\n" + addition.strip() + "\n\n" + suffix


def _best_metric_text(best_metric: Optional[Dict[str, Any]]) -> str:
    if not isinstance(best_metric, dict) or not best_metric:
        return "unavailable"
    source = best_metric.get("source") or "metric"
    value = best_metric.get("value")
    if value is None:
        return str(source)
    return f"{source}={value}"


def _directive_block(directives: Sequence[Directive]) -> str:
    lines = ["Required architecture deltas for this round:"]
    for idx, directive in enumerate(directives, start=1):
        lines.append(f"{idx}. {directive.title}: {directive.instruction}")
        lines.append(f"   Why: {directive.rationale}")
    return "\n".join(lines)


def _feature_block(feature_snapshot: Dict[str, Any]) -> str:
    constants = feature_snapshot.get("constants") if isinstance(feature_snapshot.get("constants"), dict) else {}
    lines = [
        "Current baseline diagnosis:",
        f"- fingerprint: {feature_snapshot.get('fingerprint')}",
        f"- inspection mode: {feature_snapshot.get('inspection_mode') or 'legacy'}",
        f"- ast parse ok: {'yes' if feature_snapshot.get('ast_parse_ok') else 'unknown/no'}",
        f"- exact lookup first: {'yes' if feature_snapshot.get('uses_lookup_first') else 'no'}",
        f"- exact short-effect atlas: {'yes' if feature_snapshot.get('has_exact_short_atlas') else 'no'}",
        f"- bounded local DP: {'yes' if feature_snapshot.get('has_local_window_dp') else 'no'}",
        f"- multi-policy sweep: {'yes' if feature_snapshot.get('has_multi_policy_sweep') else 'no'}",
        f"- bidirectional rewrite: {'yes' if feature_snapshot.get('has_bidirectional_rewrite') else 'no'}",
        f"- macro mining: {'yes' if feature_snapshot.get('has_macro_mining') else 'no'}",
        f"- candidate-bank scoring: {'yes' if feature_snapshot.get('has_candidate_bank_scoring') else 'no'}",
    ]
    probe = feature_snapshot.get("runtime_probe") if isinstance(feature_snapshot.get("runtime_probe"), dict) else {}
    if probe:
        lines.append("- trace/evidence probes: " + ", ".join(f"{k}={v}" for k, v in sorted(probe.items())))
    rendered = ", ".join(f"{k}={v}" for k, v in constants.items() if v is not None)
    if rendered:
        lines.append(f"- extracted constants: {rendered}")
    return "\n".join(lines)


def _history_block(history: Sequence[Dict[str, Any]]) -> str:
    items = summarize_history(history)
    if not items:
        return "Recent round memory:\n- no prior rounds yet"
    return "Recent round memory:\n- " + "\n- ".join(items)


def _history_signal_block(history: Sequence[Dict[str, Any]]) -> str:
    signals = analyze_history_signals(history)
    labels = ", ".join(signals.get("recent_failure_labels") or []) or "none"
    return "\n".join(
        [
            "Recent failure / plateau signals:",
            f"- recent_accepts: {signals.get('recent_accepts')}",
            f"- recent_failures: {signals.get('recent_failures')} (json={signals.get('json_contract_failures')}, validation={signals.get('validation_failures')}, illegal_move={signals.get('illegal_move_failures')}, replay={signals.get('replay_mismatch_failures')}, compile_import={signals.get('compile_or_import_failures')}, runtime={signals.get('runtime_failures')}, no_novelty={signals.get('no_novelty_failures')}, no_per_row={signals.get('no_per_row_improvement_failures')}, provider_fallback={signals.get('provider_or_fallback_failures')})",
            f"- validated_not_selected: {signals.get('validated_not_selected')}",
            f"- plateau_detected: {'yes' if signals.get('plateau') else 'no'}",
            f"- accepted_metric_plateau: {'yes' if signals.get('accepted_metric_plateau') else 'no'}",
            f"- score_regressions: {signals.get('score_regressions')}",
            f"- recent_failure_labels: {labels}",
            f"- multi_axis_plateau: {'yes' if signals.get('multiaxis_plateau') else 'no'}",
            f"- stagnation_axes: {', '.join(signals.get('stagnation_axes') or []) or 'none'}",
            f"- row_delta_window: improved={signals.get('recent_improved_rows')} regressed={signals.get('recent_regressed_rows')} net_delta_moves={signals.get('recent_net_delta_moves')}",
            f"- novelty_window: solver_hashes={signals.get('solver_hash_novelty_count')} submission_hashes={signals.get('submission_hash_novelty_count')} identical_solver={signals.get('identical_solver_rejections')} identical_submission={signals.get('identical_submission_rejections')}",
        ]
    )


def _diff_first_contract_block() -> str:
    return "\n".join(
        [
            "Diff-first / patch discipline:",
            "- Prefer a small unified-diff-style patch plan over a full rewrite when the incumbent already validates.",
            "- If returning a full Python file is required by the outer contract, still make the code changes traceable to a compact patch lane.",
            "- Do not change only comments, names, formatting, lookup paths, or wrapper plumbing; modify the optimizer/search core or return no_candidate.",
            "- Preserve solve(vec), JSON stdout, official move names, lookup-first safety, and deterministic replay.",
        ]
    )


def _candidate_manifest_contract_block() -> str:
    return "\n".join(
        [
            "Candidate manifest contract:",
            "- The solver must expose a machine-readable CANDIDATE_MANIFEST dict or JSON-like constant.",
            "- Required fields: lane_id, changed_mechanism, target_rows, expected_improved_rows, fallback_policy, novelty_claim.",
            "- Valid lane_id values: lane_patch, lane_fresh, lane_params, lane_hard_row_micro, lane_portfolio.",
            "- changed_mechanism must describe a real algorithmic/search-core change, not a wrapper or baseline replay.",
            "- fallback_policy must guarantee per-row rollback when candidate_len >= baseline_len or exact replay fails.",
        ]
    )


def _breakthrough_lane_block() -> str:
    return "\n".join(
        [
            "Advanced evaluator-driven lane policy:",
            "- lane_patch: minimal diff over the incumbent with exact replay and no-regression rollback.",
            "- lane_fresh: fresh implementation only if it preserves the same public contract and can replay/legal-check every row.",
            "- lane_params: deterministic parameter/policy sweep; no LLM-dependent runtime behavior.",
            "- lane_hard_row_micro: target 1-3 listed hard rows and accept only if at least one row is shorter by exact replay.",
            "- lane_portfolio: per-row router chooses among exact-valid strategies, never among invalid or unknown paths.",
            "- Every lane must emit enough lineage to explain why it survived or rolled back.",
        ]
    )


def _algorithm_search_block(round_idx: int) -> str:
    return "\n".join(
        [
            "Algorithm-search operating mode:",
            f"- round: {round_idx}",
            "- think in terms of a tiny population of bounded candidates, not one fragile linear rewrite",
            "- require an exact evaluator shard or replay-based acceptance gate before risky optimizer stages survive",
            "- keep two lanes when useful: minimal patch lane and fresh-contract-safe lane",
            "- preserve lineage: baseline fingerprint, candidate type, acceptance metric, and rollback reason",
            "- optimize for a simple Pareto frontier over exact validity, score delta, runtime risk, and novelty\n- evaluate risky changes on deterministic shadow splits so prompt evolution is driven by exact dev-score rather than prose confidence",
            "- never use Kaggle leaderboard probing as an inner-loop reward; rely on bundled deterministic evaluation only",
            "- after a no-novelty/no-per-row-improvement round, switch from broad solver refactoring to a hard-row micro-target: improve one listed hard row with exact replay and zero regressions",
        ]
    )


def synthesize_round_prompt_text(
    *,
    base_prompt_text: str,
    baseline_code: str,
    round_idx: int,
    score_history: Sequence[Dict[str, Any]],
    best_metric: Optional[Dict[str, Any]],
    prompt_history: Sequence[Dict[str, Any]],
    row_profile_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    feature_snapshot = inspect_solver_code(baseline_code)
    directives = select_directives(
        feature_snapshot=feature_snapshot,
        round_idx=round_idx,
        prompt_history=prompt_history,
        score_history=score_history,
    )
    directive_evidence_checks = build_evidence_checks([directive.key for directive in directives], feature_snapshot) if build_evidence_checks is not None else []
    row_memory_block = row_profile_prompt_block(row_profile_summary or {}) if row_profile_prompt_block is not None else "Row-level exact-search memory:\n- unavailable"
    evidence_block = evidence_prompt_block(directive_evidence_checks) if evidence_prompt_block is not None else "Executable evidence requirements:\n- unavailable"
    evolution_block = "\n\n".join(
        [
            f"SELF-IMPROVEMENT ROUND {round_idx}",
            "The injected baseline for this round is the previous best validated solver. Do not merely rename helpers or tweak constants; produce a materially stronger alternative while preserving the public Megaminx competition contract.",
            f"Current best selection metric: {_best_metric_text(best_metric)}",
            _feature_block(feature_snapshot),
            _history_block(score_history),
            _history_signal_block(score_history),
            row_memory_block,
            evidence_block,
            _failure_repair_block(score_history, baseline_code),
            _diff_first_contract_block(),
            _candidate_manifest_contract_block(),
            _breakthrough_lane_block(),
            _algorithm_search_block(round_idx),
            _directive_block(directives),
            "Acceptance intent for this round: keep exact lookup first, preserve legal official move names only, preserve deterministic replay, add an explicit anti-regression fallback, maintain candidate lineage, and materially revise the local optimization core so at least one listed hard row can be shortened by exact replay with zero regressions.",
            "Planner quality bar for this round: prefer compile-safe staged patches, require exact semantic-equivalence replay for every new local rewrite family, design a tiny prompt/candidate population instead of a single chain, and build a deterministic policy bank instead of trusting one heuristic ordering.",
            "No-op prohibition: if the candidate would produce an identical submission digest or zero improved rows, it must explicitly report no_candidate instead of wrapping/copying the incumbent optimized_submission.",
        ]
    )
    text = _insert_before_strict_contract(base_prompt_text, evolution_block)
    return {
        "prompt_text": text,
        "feature_snapshot": feature_snapshot,
        "selected_directives": [directive.key for directive in directives],
        "selected_titles": [directive.title for directive in directives],
        "directive_evidence_checks": directive_evidence_checks,
        "row_profile_summary": row_profile_summary or {},
    }


def synthesize_round_custom_prompts(
    *,
    base_custom_prompts_text: str,
    round_idx: int,
    feature_snapshot: Dict[str, Any],
    selected_directives: Sequence[str],
    best_metric: Optional[Dict[str, Any]],
    failure_repair_report: Optional[Dict[str, Any]] = None,
    row_profile_summary: Optional[Dict[str, Any]] = None,
    directive_evidence_checks: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[str]:
    if not str(base_custom_prompts_text or "").strip():
        return None
    try:
        payload = json.loads(base_custom_prompts_text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    titles = [DIRECTIVE_LIBRARY[key].title for key in selected_directives if key in DIRECTIVE_LIBRARY]
    directive_summary = "; ".join(titles)
    best_metric_text = _best_metric_text(best_metric)
    failure_context_text = _compact_failure_context(failure_repair_report or {})
    row_context_text = row_profile_prompt_block(row_profile_summary or {}) if row_profile_prompt_block is not None else "Row-level memory unavailable"
    evidence_context_text = evidence_prompt_block(directive_evidence_checks or []) if evidence_prompt_block is not None else "Evidence checks unavailable"
    per_role_additions = {
        "planner": (
            f"SELF-IMPROVEMENT ROUND {round_idx}: the injected baseline is the previous best validated solver with fingerprint {feature_snapshot.get('fingerprint')}. "
            f"Return a materially stronger plan rather than a paraphrase. Force the plan to rethink the optimization core around: {directive_summary}. "
            f"Current best metric: {best_metric_text}. Failure context: {failure_context_text}. Row context: {row_context_text}. Evidence requirements: {evidence_context_text}. Include an anti-regression story, an exact evaluator-shard story, and a small prompt-population / candidate-lineage story. "
            "Require a CANDIDATE_MANIFEST with lane_id/changed_mechanism/target_rows/expected_improved_rows/fallback_policy/novelty_claim, and prefer a diff-first lane plan. "
            "If the previous candidate failed, start with a failure autopsy and a compile-validation repair ladder before proposing new optimizer ambition."
        ),
        "coder": (
            f"SELF-IMPROVEMENT ROUND {round_idx}: the injected baseline is the previous best validated solver. "
            f"Do not produce a superficial patch. Make the optimization core materially stronger around: {directive_summary}. "
            "Preserve lookup-first semantics, legal move names only, deterministic replay, explicit rollback-safe score guarding, competition-safe bounded search, and a bounded patch-vs-fresh candidate split backed by exact replay-based acceptance. "
            f"Failure context: {failure_context_text}. Row context: {row_context_text}. Evidence requirements: {evidence_context_text}. After a failed round, first make the solver compile/import and pass the validator, then keep or add one bounded exact-valid improvement on at least one listed hard row."
        ),
        "fixer": (
            f"SELF-IMPROVEMENT ROUND {round_idx}: when repairing the candidate, preserve the newly requested architecture deltas ({directive_summary}) instead of collapsing back to the previous answer. "
            "Fix only what is necessary for correctness and validation, and preserve any replay-equivalence, evaluator-shard, lineage, or score-guard logic that was added on purpose. "
            f"Failure context: {failure_context_text}. Row context: {row_context_text}. Evidence requirements: {evidence_context_text}. Use the failed code as a negative example: identify the exact broken contract, repair it minimally, and keep the candidate moving toward a working improved solver."
        ),
    }
    for role, addition in per_role_additions.items():
        role_text = str(payload.get(role) or "").strip()
        if not role_text:
            continue
        payload[role] = _insert_before_strict_contract(role_text, addition)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_round_prompt_bundle(
    *,
    base_prompt_file: Path,
    base_custom_prompts: Optional[Path],
    baseline_solver: Path,
    round_idx: int,
    score_history: Sequence[Dict[str, Any]],
    best_metric: Optional[Dict[str, Any]],
    prompt_history: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    base_prompt_text = _safe_read_text(base_prompt_file)
    base_custom_prompts_text = _safe_read_text(base_custom_prompts)
    baseline_code = _safe_read_text(baseline_solver)
    row_profile_summary = load_row_profile_summary(output_dir=output_dir, baseline_solver_path=baseline_solver) if load_row_profile_summary is not None else {"available": False}

    prompt_result = synthesize_round_prompt_text(
        base_prompt_text=base_prompt_text,
        baseline_code=baseline_code,
        round_idx=round_idx,
        score_history=score_history,
        best_metric=best_metric,
        prompt_history=prompt_history,
        row_profile_summary=row_profile_summary,
    )
    failure_report = _failure_repair_report(score_history, baseline_code)
    custom_prompts_text = synthesize_round_custom_prompts(
        base_custom_prompts_text=base_custom_prompts_text,
        round_idx=round_idx,
        feature_snapshot=prompt_result["feature_snapshot"],
        selected_directives=prompt_result["selected_directives"],
        best_metric=best_metric,
        failure_repair_report=failure_report,
        row_profile_summary=row_profile_summary,
        directive_evidence_checks=prompt_result.get("directive_evidence_checks"),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = output_dir / f"round_{round_idx:04d}_user_prompt.txt"
    prompt_file.write_text(prompt_result["prompt_text"], encoding="utf-8")
    custom_file: Optional[Path] = None
    if custom_prompts_text is not None:
        custom_file = output_dir / f"round_{round_idx:04d}_custom_prompts.json"
        custom_file.write_text(custom_prompts_text, encoding="utf-8")

    meta = {
        "round": int(round_idx),
        "baseline_solver": str(baseline_solver),
        "baseline_fingerprint": prompt_result["feature_snapshot"].get("fingerprint"),
        "best_metric": best_metric,
        "selected_directives": list(prompt_result["selected_directives"]),
        "selected_titles": list(prompt_result["selected_titles"]),
        "feature_snapshot": prompt_result["feature_snapshot"],
        "prompt_file": str(prompt_file),
        "custom_prompts_file": str(custom_file) if custom_file is not None else None,
        "history_summary": summarize_history(score_history),
        "history_signals": analyze_history_signals(score_history),
        "failure_repair_report": failure_report,
        "row_profile_summary": row_profile_summary,
        "directive_evidence_checks": prompt_result.get("directive_evidence_checks"),
    }
    meta_path = output_dir / f"round_{round_idx:04d}_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "prompt_file": prompt_file,
        "custom_prompts_file": custom_file,
        "meta": meta,
        "meta_file": meta_path,
    }
