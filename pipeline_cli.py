#!/usr/bin/env python3
"""Pipeline CLI

This repo started as a single-competition template. We extend it with:
- per-competition baselines
- per-competition validators
- per-competition prompt bundles
- a CLI switch (competition slug) that selects the right pieces

Key idea: `--competition` selects a pipeline from pipeline_registry.

Examples
--------

# List supported pipelines
python pipeline_cli.py list-pipelines

# Run RapaportM2 pipeline (no LLM, baseline solver) on a local test.csv
python pipeline_cli.py run \
  --competition cayleypy-rapapport-m2 \
  --puzzles /path/to/test.csv \
  --output submission.csv \
  --no-llm

# Run Pancake pipeline (no LLM, baseline solver)
python pipeline_cli.py run \
  --competition CayleyPy-pancake \
  --puzzles /path/to/test.csv \
  --output submission.csv \
  --no-llm

# Generate a new solver with AgentLaboratory for RapaportM2
python pipeline_cli.py generate-solver \
  --competition cayleypy-rapapport-m2 \
  --out generated/rapapport_solve_module.py

"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pipeline_registry import PipelineSpec, get_pipeline, list_pipelines


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _stage(title: str) -> float:
    """Simple stage timer + logger."""
    t0 = time.time()
    print(f"\n=== {title} ===", flush=True)
    return t0


def _stage_done(title: str, t0: float) -> None:
    dt = time.time() - t0
    print(f"[done] {title}  ({dt:.2f}s)", flush=True)


def _resolve_default_puzzles(spec: PipelineSpec) -> Path:
    """If --puzzles is omitted, try competitions/<key>/data/test.csv."""
    candidate = ROOT / 'competitions' / spec.key / 'data' / 'test.csv'
    if candidate.exists():
        return candidate
    candidate = ROOT / 'competitions' / spec.competition / 'data' / 'test.csv'
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"No puzzles CSV provided and no bundled test.csv found for '{spec.key}'. "
        f"Expected: {ROOT/'competitions'/spec.key/'data'/'test.csv'}"
    )


def _resolve_sample_submission(spec: PipelineSpec) -> Optional[Path]:
    """Try to locate bundled sample_submission.csv for the given pipeline."""
    candidates = [
        ROOT / 'competitions' / spec.key / 'data' / 'sample_submission.csv',
        ROOT / 'competitions' / spec.competition / 'data' / 'sample_submission.csv',
        ROOT / 'competitions' / spec.key / 'submissions' / 'sample_submission.csv',
        ROOT / 'competitions' / spec.competition / 'submissions' / 'sample_submission.csv',
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _read_csv_header_and_ids(path: Path) -> tuple[list[str], list[str]]:
    with path.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV file is empty: {path}")

        ids: list[str] = []
        id_idx = 0
        for row in reader:
            if not row:
                continue
            if len(row) <= id_idx:
                continue
            ids.append(row[id_idx])
    return [h.strip() for h in header], ids


def _validate_submission_schema(
    *,
    submission_csv: Path,
    sample_submission_csv: Path,
    check_ids: bool = True,
) -> dict:
    """Compare submission.csv schema to sample_submission.csv.

    Checks:
    - column names (and order) match exactly
    - number of rows matches
    - (optional) id set matches

    Returns a small stats dict for run logging.
    """
    sub_header, sub_ids = _read_csv_header_and_ids(submission_csv)
    samp_header, samp_ids = _read_csv_header_and_ids(sample_submission_csv)

    if sub_header != samp_header:
        raise ValueError(
            "submission.csv header does not match sample_submission.csv\n"
            f"  submission: {sub_header}\n"
            f"  sample:     {samp_header}\n"
            "Fix: ensure your pipeline writes the exact columns in the same order as the sample."
        )

    if len(sub_ids) != len(samp_ids):
        raise ValueError(
            "submission.csv row count does not match sample_submission.csv\n"
            f"  submission rows: {len(sub_ids)}\n"
            f"  sample rows:     {len(samp_ids)}\n"
            "Fix: ensure you generate one prediction per test row."
        )

    id_stats: dict[str, Any] = {}
    if check_ids:
        sub_set = set(sub_ids)
        samp_set = set(samp_ids)
        missing = list(sorted(samp_set - sub_set))
        extra = list(sorted(sub_set - samp_set))
        if missing or extra:
            raise ValueError(
                "submission.csv ids do not match sample_submission.csv ids\n"
                f"  missing ids (in submission): {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
                f"  extra ids (not in sample):   {extra[:5]}{'...' if len(extra) > 5 else ''}\n"
                "Fix: keep the same id column values as in test/sample_submission."
            )
        id_stats = {
            "unique_ids": len(sub_set),
            "duplicate_ids": len(sub_ids) - len(sub_set),
        }

    return {
        "columns": sub_header,
        "rows": len(sub_ids),
        **id_stats,
    }


def _append_run_log(path: Path, record: dict) -> None:
    """Append a run record to run_log.json.

    If the file exists:
    - list -> append
    - dict -> convert to list and append
    Otherwise create a new list.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data: Any
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            data = []
    else:
        data = []

    if isinstance(data, list):
        data.append(record)
    elif isinstance(data, dict):
        data = [data, record]
    else:
        data = [record]

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def _file_stats(path: Path, *, csv_stats: bool = False) -> dict[str, Any]:
    """Collect lightweight file stats for run_log.

    For CSVs we record row count (data rows, excluding header) and column names.
    This is best-effort and designed for performance tracking.
    """
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        return out

    try:
        st = path.stat()
        out["bytes"] = int(st.st_size)
        out["mtime"] = datetime.fromtimestamp(st.st_mtime).isoformat()
    except Exception:
        # best-effort
        pass

    if csv_stats:
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, [])
                rows = 0
                for _ in reader:
                    rows += 1
            out["columns"] = header
            out["rows"] = rows
        except UnicodeDecodeError:
            # fallback: try default encoding
            try:
                with path.open("r", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                    rows = 0
                    for _ in reader:
                        rows += 1
                out["columns"] = header
                out["rows"] = rows
            except Exception:
                pass
        except Exception:
            pass

    return out


def _attach_io_stats(
    report: dict[str, Any],
    *,
    puzzles_csv: Path | None = None,
    output_csv: Path | None = None,
    solver_path: Path | None = None,
    sample_submission_csv: Path | None = None,
) -> None:
    """Attach file sizes + row counts for test/submission to the run report."""
    files: dict[str, Any] = report.get("files", {}) if isinstance(report.get("files"), dict) else {}

    if puzzles_csv is not None:
        files["puzzles_csv"] = _file_stats(puzzles_csv, csv_stats=True)
    if output_csv is not None:
        files["output_csv"] = _file_stats(output_csv, csv_stats=True)
    if solver_path is not None:
        files["solver"] = _file_stats(solver_path, csv_stats=False)
    if sample_submission_csv is not None:
        files["sample_submission_csv"] = _file_stats(sample_submission_csv, csv_stats=True)

    report["files"] = files


def _kaggle_submit(
    *,
    competition: str,
    submission_csv: Path,
    message: str,
    kaggle_json: str | None = None,
    submit_via: str = 'auto',
    kaggle_config_dir: str | None = None,
) -> None:
    """Submit to Kaggle using either the Python API or the CLI.

    submit_via: 'auto' | 'api' | 'cli'
    """
    if submit_via not in {'auto', 'api', 'cli'}:
        raise ValueError(f"Unknown submit_via={submit_via}")

    if submit_via in {'auto', 'api'}:
        try:
            _ensure_llm_puzzles_on_path()
            from src.kaggle_utils import ensure_auth, submit_file, latest_scored_submission

            api = ensure_auth(kaggle_json_path=kaggle_json, config_dir=kaggle_config_dir)
            print(f"[kaggle] submitting via API: competition={competition} file={submission_csv}", flush=True)
            submit_file(api, competition=competition, filepath=str(submission_csv), message=message)

            scored = latest_scored_submission(api, competition)
            if scored:
                ps = scored.get('public_score')
                prs = scored.get('private_score')
                sid = scored.get('id') or scored.get('ref')
                print(f"[kaggle] latest scored submission: id={sid} public={ps} private={prs}", flush=True)
            return
        except Exception as e:
            if submit_via == 'api':
                raise
            print(f"[kaggle] API submit failed, falling back to CLI: {e}", flush=True)

    env = os.environ.copy()
    if kaggle_json:
        try:
            _ensure_llm_puzzles_on_path()
            from src.kaggle_utils import prepare_kaggle_config
            cfg_dir = prepare_kaggle_config(kaggle_json, target_dir=kaggle_config_dir)
            env['KAGGLE_CONFIG_DIR'] = cfg_dir
        except Exception as e:
            print(f"[kaggle] could not prepare kaggle.json for CLI: {e}", flush=True)

    cmd = ['kaggle', 'competitions', 'submit', '-c', competition, '-f', str(submission_csv), '-m', message]
    print('[kaggle] ' + ' '.join(cmd), flush=True)
    subprocess.check_call(cmd, env=env)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compile_all() -> None:
    """Compile all python files to catch syntax errors early."""
    print("[compile] Running python -m compileall ...")
    subprocess.check_call([PYTHON, "-m", "compileall", str(ROOT)])
    print("[compile] OK")


def _load_solve_fn(solver_path: Path) -> Callable[[Sequence[int]], Tuple[Any, Any]]:
    """Dynamically import `solve` from an arbitrary solve_module.py."""
    if not solver_path.exists():
        raise FileNotFoundError(solver_path)

    module_name = f"solve_module_dyn_{abs(hash(str(solver_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, solver_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {solver_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    solve = getattr(module, "solve", None)
    if solve is None or not callable(solve):
        raise AttributeError(f"No callable solve(vec) in {solver_path}")

    return solve


def _parse_int_list(s: str) -> List[int]:
    """Parse a list of ints from either JSON '[1,2,3]' or CSV '1,2,3'."""
    s = s.strip()
    if not s:
        return []

    if s[0] in "[(":
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [int(x) for x in obj]
        except Exception:
            # fall back to CSV parsing
            pass

    # CSV style: "3,0,1,4,2"
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _extract_state(row: Dict[str, str], spec: PipelineSpec, vector_col_override: Optional[str] = None) -> List[int]:
    """Extract puzzle state vector from a CSV row based on pipeline spec."""
    # Explicit override wins
    if vector_col_override:
        if vector_col_override not in row:
            raise KeyError(f"vector column {vector_col_override!r} not found in CSV")
        return _parse_int_list(row[vector_col_override])

    # Try pipeline candidates
    if spec.state_columns:
        for col in spec.state_columns:
            if col in row and row[col].strip() != "":
                return _parse_int_list(row[col])

    # Fallback: if row has exactly one non-id column, take it
    non_empty = {k: v for k, v in row.items() if v.strip() != ""}
    for candidate in ("vector", "permutation", "initial_state", "state"):
        if candidate in non_empty:
            return _parse_int_list(non_empty[candidate])

    raise KeyError(
        "Could not infer state column. Please pass --vector-col explicitly or update pipeline_registry.py"
    )


def _validate_solver(solver_path: Path, validator_path: Path, smoke_vector: Sequence[int]) -> None:
    print(f"[validate] {validator_path.name} ...")
    subprocess.check_call(
        [
            PYTHON,
            str(validator_path),
            "--solver",
            str(solver_path),
            "--vector",
            json.dumps(list(smoke_vector)),
        ]
    )


def _ensure_llm_puzzles_on_path() -> None:
    lp_dir = ROOT / "llm-puzzles"
    if str(lp_dir) not in sys.path:
        sys.path.insert(0, str(lp_dir))


def _build_submission(
    *,
    puzzles_csv: Path,
    out_csv: Path,
    competition_format_slug: str,
    solver_path: Path,
    spec: PipelineSpec,
    vector_col_override: Optional[str] = None,
    max_rows: Optional[int] = None,
    no_progress: bool = False,
) -> None:
    """Build a Kaggle submission CSV using llm-puzzles universal_adapter."""
    _ensure_llm_puzzles_on_path()

    from src.universal_adapter import build_submission as lp_build_submission  # type: ignore

    solve_fn = _load_solve_fn(solver_path)

    def row_solver(row: Dict[str, str], cfg: Any) -> Union[List[str], str]:
        vec = _extract_state(row, spec, vector_col_override)
        out = solve_fn(vec)

        # Expected: (moves, sorted_array)
        if isinstance(out, tuple) and len(out) == 2:
            moves = out[0]
            return moves

        # Allow solve() to directly return moves list / string.
        return out  # type: ignore[return-value]

    print(f"[submit] Building submission for format={competition_format_slug}")
    print(f"         puzzles={puzzles_csv}")
    print(f"         output={out_csv}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    lp_build_submission(
        puzzles_csv=str(puzzles_csv),
        output_csv=str(out_csv),
        competition=competition_format_slug,
        solver=row_solver,
        max_rows=max_rows,
        progress=(not no_progress),
        progress_desc=f"{spec.key}: building submission",
    )


# ---------------------------------------------------------------------------
# AgentLaboratory generation
# ---------------------------------------------------------------------------


def _run_agent_laboratory(
    *,
    prompt_file: Path,
    out_path: Path,
    validator: Path,
    baseline: Path,
    custom_prompts: Optional[Path] = None,
    llm: str = "gpt-4o-mini",
    max_iters: int = 8,
    no_llm: bool = False,
    allow_baseline: bool = True,
) -> None:
    """Run AgentLaboratory perm_pipeline to generate/repair a solver."""

    pipeline_script = ROOT / "AgentLaboratory" / "perm_pipeline" / "run_perm_pipeline.py"
    if not pipeline_script.exists():
        raise FileNotFoundError(pipeline_script)

    cmd = [
        PYTHON,
        str(pipeline_script),
        "--user-prompt-file",
        str(prompt_file),
        "--out",
        str(out_path),
        "--validator",
        str(validator),
        "--baseline",
        str(baseline),
        "--max-iters",
        str(max_iters),
        "--models",
        llm,
    ]

    # run_perm_pipeline.py already falls back to baseline unless --strict is used.
    if no_llm:
        cmd.append("--no-llm")
    if custom_prompts:
        cmd.extend(["--custom-prompts", str(custom_prompts)])

    print("[agentlab] " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_list_pipelines(_: argparse.Namespace) -> None:
    print("Available pipelines (competition slugs):")
    for spec in list_pipelines():
        print(f"- {spec.key:45s}  (competition='{spec.competition}', format='{spec.format_slug}')")


def cmd_generate_solver(args: argparse.Namespace) -> None:
    spec = get_pipeline(args.competition) if args.competition else None
    if spec is None:
        raise SystemExit(
            "Unknown --competition. Run `python pipeline_cli.py list-pipelines` to see supported pipelines."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Select prompt bundle (can be overridden)
    prompt_file = Path(args.prompt_file) if args.prompt_file else spec.prompt_file
    if prompt_file is None:
        raise SystemExit(f"No prompt_file configured for {spec.key}; pass --prompt-file.")

    custom_prompts = Path(args.custom_prompts) if args.custom_prompts else spec.custom_prompts_file

    if args.no_llm:
        # Copy baseline
        shutil.copyfile(spec.baseline_solver, out_path)
        print(f"[generate-solver] --no-llm: copied baseline -> {out_path}")
        _validate_solver(out_path, spec.validator, spec.smoke_vector or [0, 1])
        return

    _run_agent_laboratory(
        prompt_file=prompt_file,
        out_path=out_path,
        validator=spec.validator,
        baseline=spec.baseline_solver,
        custom_prompts=custom_prompts,
        llm=args.models,
        max_iters=args.max_iters,
        no_llm=False,
        allow_baseline=args.allow_baseline,
    )

    _validate_solver(out_path, spec.validator, spec.smoke_vector or [0, 1])


def cmd_build_submission(args: argparse.Namespace) -> None:
    spec = get_pipeline(args.competition) if args.competition else None
    if spec is None:
        raise SystemExit(
            "Unknown --competition. Run `python pipeline_cli.py list-pipelines` to see supported pipelines."
        )

    solver_path = Path(args.solver)
    puzzles_csv = Path(args.puzzles) if args.puzzles else _resolve_default_puzzles(spec)
    out_csv = Path(args.output)

    report: dict[str, Any] = {
        "ts": datetime.now().isoformat(),
        "cmd": "build-submission",
        "pipeline": spec.key,
        "competition": spec.competition,
        "format": args.format or spec.format_slug,
        "puzzles_csv": str(puzzles_csv),
        "output_csv": str(out_csv),
        "solver": str(solver_path),
        "stages": {},
    }

    run_log_path = Path(args.run_log) if args.run_log else (out_csv.parent / "run_log.json")

    sample_for_log: Path | None = None
    try:
        sample_for_log = _resolve_sample_submission(spec)
    except Exception:
        sample_for_log = None

    try:
        t2 = _stage("build submission")
        report["stages"]["build_submission"] = {"start": time.time()}
        _build_submission(
            puzzles_csv=puzzles_csv,
            out_csv=out_csv,
            competition_format_slug=args.format or spec.format_slug,
            solver_path=solver_path,
            spec=spec,
            vector_col_override=args.vector_col,
            max_rows=args.max_rows,
            no_progress=args.no_progress,
        )
        report["stages"]["build_submission"]["end"] = time.time()
        report["stages"]["build_submission"]["seconds"] = report["stages"]["build_submission"]["end"] - report["stages"]["build_submission"]["start"]

        if args.schema_check and not args.no_schema_check:
            sample = _resolve_sample_submission(spec)
            if sample is None:
                print(f"[schema] WARNING: no bundled sample_submission.csv found for '{spec.key}'. Skipping.")
            else:
                tsc = _stage("schema check")
                report["stages"]["schema_check"] = {"start": time.time()}
                stats = _validate_submission_schema(
                    submission_csv=out_csv,
                    sample_submission_csv=sample,
                    check_ids=(not args.no_schema_check_ids),
                )
                report["schema"] = {"sample": str(sample), **stats}
                report["stages"]["schema_check"]["end"] = time.time()
                report["stages"]["schema_check"]["seconds"] = report["stages"]["schema_check"]["end"] - report["stages"]["schema_check"]["start"]
                _stage_done("schema check", tsc)

        _stage_done("build submission", t2)
        report["status"] = "ok"
    except Exception as e:
        now = time.time()
        for st in report.get("stages", {}).values():
            if isinstance(st, dict) and "start" in st and "end" not in st:
                st["end"] = now
                st["seconds"] = st["end"] - st["start"]
        report["status"] = "error"
        report["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "stacktrace": traceback.format_exc(),
        }
        raise
    finally:
        # Always attach IO stats (best-effort): file sizes + row counts for test/submission.
        try:
            _attach_io_stats(
                report,
                puzzles_csv=puzzles_csv,
                output_csv=out_csv,
                solver_path=solver_path,
                sample_submission_csv=sample_for_log,
            )
        except Exception:
            pass

        if not args.no_run_log:
            try:
                _append_run_log(run_log_path, report)
                print(f"[run_log] wrote {run_log_path}")
            except Exception as e:
                print(f"[run_log] WARNING: failed to write run log: {e}")


def cmd_validate_solver(args: argparse.Namespace) -> None:
    spec = get_pipeline(args.competition) if args.competition else None
    if spec is None:
        raise SystemExit(
            "Unknown --competition. Run `python pipeline_cli.py list-pipelines` to see supported pipelines."
        )

    solver_path = Path(args.solver)
    vec = json.loads(args.vector) if args.vector is not None else list(spec.smoke_vector or [0, 1])

    _validate_solver(solver_path, spec.validator, vec)


def cmd_run(args: argparse.Namespace) -> None:
    # Pipeline selection by competition slug
    spec = get_pipeline(args.competition)
    if spec is None:
        raise SystemExit(
            f"Unknown competition/pipeline '{args.competition}'. Run `python pipeline_cli.py list-pipelines`."
        )

    generated_dir = ROOT / "generated"
    generated_dir.mkdir(exist_ok=True)

    puzzles_csv = Path(args.puzzles) if args.puzzles else _resolve_default_puzzles(spec)
    out_csv = Path(args.output)
    run_log_path = Path(args.run_log) if args.run_log else (out_csv.parent / "run_log.json")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    solver_path = generated_dir / f"solve_{spec.key}_{ts}.py"

    report: dict[str, Any] = {
        "ts": datetime.now().isoformat(),
        "cmd": "run",
        "pipeline": spec.key,
        "competition": spec.competition,
        "format": args.format or spec.format_slug,
        "puzzles_csv": str(puzzles_csv),
        "output_csv": str(out_csv),
        "solver": str(solver_path),
        "args": {
            "no_llm": bool(args.no_llm),
            "models": args.models,
            "max_iters": args.max_iters,
            "vector_col": args.vector_col,
            "max_rows": args.max_rows,
            "submit": bool(args.submit),
            "submit_via": args.submit_via,
            "submit_competition": args.submit_competition,
        },
        "stages": {},
    }

    sample_for_log: Path | None = None
    try:
        sample_for_log = _resolve_sample_submission(spec)
    except Exception:
        sample_for_log = None

    try:
        t0 = _stage("generate solver")
        report["stages"]["generate_solver"] = {"start": time.time()}
        if args.no_llm:
            shutil.copyfile(spec.baseline_solver, solver_path)
            print(f"[run] --no-llm: copied baseline solver -> {solver_path}")
        else:
            prompt_file = Path(args.prompt_file) if args.prompt_file else spec.prompt_file
            if prompt_file is None:
                raise SystemExit(f"No prompt_file configured for {spec.key}; pass --prompt-file.")

            custom_prompts = Path(args.custom_prompts) if args.custom_prompts else spec.custom_prompts_file

            _run_agent_laboratory(
                prompt_file=prompt_file,
                out_path=solver_path,
                validator=spec.validator,
                baseline=spec.baseline_solver,
                custom_prompts=custom_prompts,
                llm=args.models,
                max_iters=args.max_iters,
                no_llm=False,
                allow_baseline=args.allow_baseline,
            )

        report["stages"]["generate_solver"]["end"] = time.time()
        report["stages"]["generate_solver"]["seconds"] = report["stages"]["generate_solver"]["end"] - report["stages"]["generate_solver"]["start"]
        _stage_done("generate solver", t0)

        # Smoke validate
        t1 = _stage("validate solver")
        report["stages"]["validate_solver"] = {"start": time.time()}
        _validate_solver(solver_path, spec.validator, spec.smoke_vector or [0, 1])
        report["stages"]["validate_solver"]["end"] = time.time()
        report["stages"]["validate_solver"]["seconds"] = report["stages"]["validate_solver"]["end"] - report["stages"]["validate_solver"]["start"]
        _stage_done("validate solver", t1)

        # Build submission
        t2 = _stage("build submission")
        report["stages"]["build_submission"] = {"start": time.time()}
        _build_submission(
            puzzles_csv=puzzles_csv,
            out_csv=out_csv,
            competition_format_slug=args.format or spec.format_slug,
            solver_path=solver_path,
            spec=spec,
            vector_col_override=args.vector_col,
            max_rows=args.max_rows,
            no_progress=args.no_progress,
        )
        report["stages"]["build_submission"]["end"] = time.time()
        report["stages"]["build_submission"]["seconds"] = report["stages"]["build_submission"]["end"] - report["stages"]["build_submission"]["start"]
        _stage_done("build submission", t2)

        # Schema check (auto before submit; optional otherwise)
        do_schema_check = (args.submit or args.schema_check) and (not args.no_schema_check)
        if do_schema_check:
            sample = _resolve_sample_submission(spec)
            if sample is None:
                print(f"[schema] WARNING: no bundled sample_submission.csv found for '{spec.key}'. Skipping.")
            else:
                tsc = _stage("schema check")
                report["stages"]["schema_check"] = {"start": time.time()}
                stats = _validate_submission_schema(
                    submission_csv=out_csv,
                    sample_submission_csv=sample,
                    check_ids=(not args.no_schema_check_ids),
                )
                report["schema"] = {"sample": str(sample), **stats}
                report["stages"]["schema_check"]["end"] = time.time()
                report["stages"]["schema_check"]["seconds"] = report["stages"]["schema_check"]["end"] - report["stages"]["schema_check"]["start"]
                _stage_done("schema check", tsc)

        # Optionally submit to Kaggle
        if args.submit:
            if not args.message:
                raise SystemExit("--submit requires --message")

            t3 = _stage("submit to Kaggle")
            report["stages"]["submit_kaggle"] = {"start": time.time()}
            submit_comp = args.submit_competition or spec.competition
            _kaggle_submit(
                competition=submit_comp,
                submission_csv=out_csv,
                message=args.message,
                kaggle_json=args.kaggle_json,
                submit_via=args.submit_via,
                kaggle_config_dir=args.kaggle_config_dir,
            )
            report["stages"]["submit_kaggle"]["end"] = time.time()
            report["stages"]["submit_kaggle"]["seconds"] = report["stages"]["submit_kaggle"]["end"] - report["stages"]["submit_kaggle"]["start"]
            _stage_done("submit to Kaggle", t3)

        report["status"] = "ok"
        print("[run] Done.")
    except Exception as e:
        now = time.time()
        for st in report.get("stages", {}).values():
            if isinstance(st, dict) and "start" in st and "end" not in st:
                st["end"] = now
                st["seconds"] = st["end"] - st["start"]
        report["status"] = "error"
        report["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "stacktrace": traceback.format_exc(),
        }
        raise
    finally:
        # Always attach IO stats (best-effort): file sizes + row counts for test/submission.
        try:
            _attach_io_stats(
                report,
                puzzles_csv=puzzles_csv,
                output_csv=out_csv,
                solver_path=solver_path,
                sample_submission_csv=sample_for_log,
            )
        except Exception:
            pass

        if not args.no_run_log:
            try:
                _append_run_log(run_log_path, report)
                print(f"[run_log] wrote {run_log_path}")
            except Exception as e:
                print(f"[run_log] WARNING: failed to write run log: {e}")


def cmd_selftest(_: argparse.Namespace) -> None:
    """Offline smoke tests (no Kaggle, no LLM)."""
    compile_all()

    tmp = ROOT / "_selftest"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    # Test a few pipelines end-to-end
    to_test = [
        "lrx-sort",
        "cayleypy-rapapport-m2",
        "CayleyPy-pancake",
    ]

    for comp in to_test:
        spec = get_pipeline(comp)
        assert spec is not None
        print(f"\n[selftest] pipeline={spec.key}")

        # Copy baseline solver
        solver_path = tmp / f"{spec.key}_baseline.py"
        shutil.copyfile(spec.baseline_solver, solver_path)

        # Validate on smoke vector
        _validate_solver(solver_path, spec.validator, spec.smoke_vector or [0, 1])

        # Build a tiny puzzles.csv depending on pipeline
        puzzles_csv = tmp / f"{spec.key}_puzzles.csv"
        out_csv = tmp / f"{spec.key}_submission.csv"

        if spec.key == "cayleypy-rapapport-m2":
            # id,n,permutation
            with puzzles_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "n", "permutation"])
                w.writeheader()
                w.writerow({"id": "0", "n": "5", "permutation": "3,0,1,4,2"})
        elif spec.key == "cayleypy-pancake":
            # initial_state_id,initial_state,state_size
            with puzzles_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["initial_state_id", "initial_state", "state_size"])
                w.writeheader()
                w.writerow({"initial_state_id": "0", "initial_state": "3,1,2,0", "state_size": "4"})
        else:
            # generic vector column
            with puzzles_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "vector"])
                w.writeheader()
                w.writerow({"id": "0", "vector": json.dumps(spec.smoke_vector or [3, 1, 2])})

        _build_submission(
            puzzles_csv=puzzles_csv,
            out_csv=out_csv,
            competition_format_slug=spec.format_slug,
            solver_path=solver_path,
            spec=spec,
            vector_col_override=None,
            max_rows=None,
        )

        print(f"[selftest] wrote {out_csv}")

    print("\n[selftest] All OK")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-competition pipeline CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list-pipelines", help="List built-in pipeline configs")
    sp.set_defaults(func=cmd_list_pipelines)

    sp = sub.add_parser("generate-solver", help="Generate/repair a solver with AgentLaboratory")
    sp.add_argument("--competition", required=True, help="Competition slug / pipeline key")
    sp.add_argument("--out", required=True, help="Output path for generated solver")
    sp.add_argument("--prompt-file", default=None, help="Override user prompt file")
    sp.add_argument("--custom-prompts", default=None, help="Override AgentLaboratory custom prompts JSON")
    sp.add_argument("--models", dest="models", default="gpt-4o-mini", help="Comma-separated g4f model list (passed to AgentLaboratory --models)")
    sp.add_argument("--llm", dest="models", default=None, help=argparse.SUPPRESS)
    sp.add_argument("--max-iters", type=int, default=8)
    sp.add_argument("--allow-baseline", action="store_true")
    sp.add_argument("--no-llm", action="store_true", help="Skip LLM: just copy baseline")
    sp.set_defaults(func=cmd_generate_solver)

    sp = sub.add_parser("build-submission", help="Build a submission CSV from a solver")
    sp.add_argument("--competition", required=True, help="Competition slug / pipeline key")
    sp.add_argument("--puzzles", required=False, default=None, help="Input puzzles/test CSV (optional; uses bundled competitions/<slug>/data/test.csv if omitted)")
    sp.add_argument("--solver", required=True, help="Path to solve_module.py")
    sp.add_argument("--output", required=True, help="Submission CSV output")
    sp.add_argument("--format", default=None, help="Override llm-puzzles format slug")
    sp.add_argument("--vector-col", default=None, help="Override state column")
    sp.add_argument("--max-rows", type=int, default=None)
    sp.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    sp.add_argument("--schema-check", action="store_true", help="Compare output schema to bundled sample_submission.csv")
    sp.add_argument("--no-schema-check", action="store_true", help="Disable schema check (even if --schema-check is set)")
    sp.add_argument("--no-schema-check-ids", action="store_true", help="Skip id set comparison during schema check")
    sp.add_argument("--run-log", default=None, help="Path to run_log.json (default: <output_dir>/run_log.json)")
    sp.add_argument("--no-run-log", action="store_true", help="Disable writing run_log.json")
    sp.set_defaults(func=cmd_build_submission)


    sp = sub.add_parser("validate-solver", help="Validate a solver with the competition-specific validator")
    sp.add_argument("--competition", required=True, help="Competition slug / pipeline key")
    sp.add_argument("--solver", required=True, help="Path to solve_module.py")
    sp.add_argument("--vector", default=None, help="Optional JSON list. If omitted, uses the pipeline smoke vector.")
    sp.set_defaults(func=cmd_validate_solver)

    sp = sub.add_parser("run", help="End-to-end: (generate solver) -> validate -> build submission -> optional Kaggle submit")
    sp.add_argument("--competition", required=True, help="Competition slug / pipeline key")
    sp.add_argument("--puzzles", required=False, default=None, help="Input puzzles/test CSV (optional; uses bundled competitions/<slug>/data/test.csv if omitted)")
    sp.add_argument("--output", required=True, help="Submission CSV output")
    sp.add_argument("--prompt-file", default=None, help="Override user prompt file")
    sp.add_argument("--custom-prompts", default=None, help="Override custom prompts JSON")
    sp.add_argument("--models", dest="models", default="gpt-4o-mini", help="Comma-separated g4f model list (passed to AgentLaboratory --models)")
    sp.add_argument("--llm", dest="models", default=None, help=argparse.SUPPRESS)
    sp.add_argument("--max-iters", type=int, default=8)
    sp.add_argument("--allow-baseline", action="store_true")
    sp.add_argument("--no-llm", action="store_true")
    sp.add_argument("--format", default=None, help="Override llm-puzzles format slug")
    sp.add_argument("--vector-col", default=None, help="Override state column")
    sp.add_argument("--max-rows", type=int, default=None)
    sp.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    sp.add_argument("--submit", action="store_true")
    sp.add_argument("--message", default=None, help="Kaggle submission message")
    sp.add_argument("--kaggle-json", default=None, help="Path to kaggle.json (optional). If set, credentials are loaded via KAGGLE_CONFIG_DIR.")
    sp.add_argument("--kaggle-config-dir", default=None, help="Optional directory to place a temporary kaggle.json copy")
    sp.add_argument("--submit-via", default="auto", choices=["auto","api","cli"], help="How to submit: auto (try API then CLI), api, or cli")
    sp.add_argument("--submit-competition", dest="submit_competition", default=None, help="Override Kaggle competition slug for submission")
    sp.add_argument("--schema-check", action="store_true", help="Compare output schema to bundled sample_submission.csv")
    sp.add_argument("--no-schema-check", action="store_true", help="Disable schema check (auto-enabled before --submit)")
    sp.add_argument("--no-schema-check-ids", action="store_true", help="Skip id set comparison during schema check")
    sp.add_argument("--run-log", default=None, help="Path to run_log.json (default: <output_dir>/run_log.json)")
    sp.add_argument("--no-run-log", action="store_true", help="Disable writing run_log.json")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("selftest", help="Offline smoke tests")
    sp.set_defaults(func=cmd_selftest)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
