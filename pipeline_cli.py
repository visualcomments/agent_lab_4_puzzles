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
import asyncio
import csv
import itertools
import importlib
import importlib.util
import json
import os
import shutil
import zipfile
import subprocess
import sys
import traceback
from datetime import datetime
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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


def _gpu_diag_hint(selected_models: str) -> None:
    """Print a short diagnostic about GPU usage.

    Why this exists:
    - Many users expect that selecting a remote model name (e.g. GPT-4 via g4f) uses the local GPU.
      It does not: those requests go to provider endpoints.
    - GPU is only used when running *local* compute (e.g., Transformers local inference) inside this runtime.
    """
    dev = (os.getenv("AGENTLAB_DEVICE") or "").strip()
    use_gpu = (os.getenv("AGENTLAB_USE_GPU") or "").strip().lower() in {"1", "true", "yes", "on"}
    if not dev and not use_gpu:
        return

    models = [m.strip() for m in (selected_models or "").split(",") if m.strip()]
    has_local = any(m.startswith("local:") for m in models)

    # Torch/CUDA availability (best effort)
    try:
        import torch  # type: ignore

        cuda_ok = bool(torch.cuda.is_available())
        n = int(torch.cuda.device_count()) if cuda_ok else 0
        msg = f"[gpu] requested via env (AGENTLAB_DEVICE/AGENTLAB_USE_GPU). torch={getattr(torch,'__version__','?')} cuda={cuda_ok} devices={n}"
        if cuda_ok and n > 0:
            try:
                msg += f" name={torch.cuda.get_device_name(0)}"
            except Exception:
                pass
        print(msg)
    except Exception:
        print("[gpu] requested via env, but torch is not available in this environment.")

    if not has_local:
        print(
            "[gpu] NOTE: your --models list contains no 'local:*' models. "
            "g4f/OpenAI/Claude/Gemini backends run remotely, so local GPU memory will stay near 0. "
            "If you want to actually use GPU, pass e.g. --models 'local:Qwen/Qwen2.5-0.5B-Instruct' (and ensure torch+transformers are installed)."
        )

def _normalize_g4f_model_name(model: str) -> str:
    raw = str(model or "").strip()
    if raw.lower().startswith("g4f:"):
        raw = raw.split(":", 1)[1].strip()
    return raw


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        s = _normalize_g4f_model_name(item)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _iter_g4f_repo_roots() -> Iterable[Path]:
    candidates = [ROOT / "gpt4free", ROOT]
    for cand in candidates:
        if cand.exists():
            yield cand


def _load_g4f_models_module():
    tried: List[str] = []
    for base in _iter_g4f_repo_roots():
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        try:
            return importlib.import_module("g4f.models")
        except Exception as exc:
            tried.append(f"{base}: {exc}")
            continue
    try:
        return importlib.import_module("g4f.models")
    except Exception as exc:  # pragma: no cover - surfaced in CLI only
        tried.append(f"site-packages: {exc}")
    joined = "; ".join(tried) if tried else "g4f.models import failed"
    raise RuntimeError(
        "g4f.models is not available. Install g4f or use the bundled ./gpt4free checkout. "
        f"Tried: {joined}"
    )


def _fetch_g4f_backend_models(backend_api_url: str, timeout: float = 15.0) -> List[str]:
    url = backend_api_url.rstrip("/") + "/backend-api/v2/models"
    try:
        with urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(f"failed to load {url}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"invalid JSON from {url}: {exc}") from exc

    if isinstance(payload, dict):
        items = payload.get("data")
        if items is None:
            items = payload.get("models")
    else:
        items = payload

    names: List[str] = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, str):
                names.append(item)
                continue
            if not isinstance(item, dict):
                continue
            name = item.get("id") or item.get("name") or item.get("model")
            if isinstance(name, str):
                names.append(name)
    return _dedupe_keep_order(names)


def _discover_g4f_candidate_models(backend_api_url: Optional[str] = None) -> List[str]:
    if backend_api_url:
        return _fetch_g4f_backend_models(backend_api_url)

    gm = _load_g4f_models_module()
    model_registry = getattr(gm, "ModelRegistry", None)
    image_cls = getattr(gm, "ImageModel", tuple())
    audio_cls = getattr(gm, "AudioModel", tuple())
    video_cls = getattr(gm, "VideoModel", tuple())

    names: List[str] = []
    if model_registry is not None and hasattr(model_registry, "all_models"):
        try:
            all_models = model_registry.all_models()
            for name, model in all_models.items():
                if not isinstance(name, str) or not name.strip():
                    continue
                if isinstance(model, (image_cls, audio_cls, video_cls)):
                    continue
                names.append(name.strip())
        except Exception:
            names = []

    if not names:
        raw_all = getattr(gm, "__all__", [])
        if callable(raw_all):
            try:
                raw_all = raw_all()
            except Exception:
                raw_all = []
        for item in raw_all or []:
            if isinstance(item, str) and item.strip():
                low = item.lower()
                if any(tag in low for tag in ("image", "audio", "video")):
                    continue
                names.append(item.strip())

    preferred = {
        "gpt-4o-mini": 0,
        "gpt-4": 1,
        "gpt-4o": 2,
        "claude-3.5-sonnet": 3,
        "claude-3-haiku": 4,
        "command-r": 5,
        "command-r-plus": 6,
        "deepseek-chat": 7,
        "aria": 8,
    }
    deduped = _dedupe_keep_order(names)
    return sorted(deduped, key=lambda s: (preferred.get(s, 9999), s.lower()))


def _load_g4f_async_client_class():
    tried: List[str] = []
    for base in _iter_g4f_repo_roots():
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        try:
            module = importlib.import_module("g4f.client")
            return getattr(module, "AsyncClient")
        except Exception as exc:
            tried.append(f"{base}: {exc}")
            continue
    try:
        module = importlib.import_module("g4f.client")
        return getattr(module, "AsyncClient")
    except Exception as exc:  # pragma: no cover - surfaced in CLI only
        tried.append(f"site-packages: {exc}")
    joined = "; ".join(tried) if tried else "g4f.client import failed"
    raise RuntimeError(
        "g4f AsyncClient is not available. Install g4f or use the bundled ./gpt4free checkout. "
        f"Tried: {joined}"
    )


async def _probe_g4f_model_async(
    model: str,
    timeout: float,
    prompt: str,
    system_prompt: str,
    provider_name: Optional[str] = None,
) -> Tuple[bool, str, float]:
    AsyncClient = _load_g4f_async_client_class()
    normalized = _normalize_g4f_model_name(model)
    started = time.time()
    client = AsyncClient()
    messages = []
    if str(system_prompt or "").strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=normalized,
                provider=(provider_name or None),
                messages=messages,
                web_search=False,
            ),
            timeout=timeout,
        )
        elapsed = time.time() - started
        txt = ""
        choices = getattr(response, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
            txt = str(getattr(message, "content", "") or "").strip()
        if not txt:
            txt = str(response or "").strip()
        if not txt:
            return False, "empty response", elapsed
        preview = txt.replace("\n", " ")[:80]
        return True, preview, elapsed
    except Exception as exc:
        elapsed = time.time() - started
        return False, str(exc), elapsed


async def _probe_g4f_models_async(
    candidates: Sequence[str],
    *,
    timeout: float,
    prompt: str,
    system_prompt: str,
    provider_name: Optional[str] = None,
    concurrency: int = 5,
    on_result: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    total = len(candidates)
    if total == 0:
        return []
    sem = asyncio.Semaphore(max(1, int(concurrency or 1)))
    results: List[Optional[Dict[str, Any]]] = [None] * total

    async def run_one(index: int, model: str) -> Tuple[int, Dict[str, Any]]:
        async with sem:
            ok, info, elapsed = await _probe_g4f_model_async(
                model=model,
                timeout=timeout,
                prompt=prompt,
                system_prompt=system_prompt,
                provider_name=provider_name,
            )
            result = {
                "model": model,
                "ok": ok,
                "detail": info,
                "elapsed_s": round(elapsed, 3),
            }
            return index, result

    tasks = [asyncio.create_task(run_one(idx, model)) for idx, model in enumerate(candidates)]
    try:
        for future in asyncio.as_completed(tasks):
            index, result = await future
            results[index] = result
            if on_result is not None:
                on_result(index + 1, total, result)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()

    return [r for r in results if r is not None]


def cmd_check_g4f_models(args: argparse.Namespace) -> None:
    provider_name = (args.provider or "").strip()
    if provider_name:
        os.environ["G4F_PROVIDER"] = provider_name

    if args.models:
        candidates = _dedupe_keep_order(args.models.split(","))
    else:
        candidates = _discover_g4f_candidate_models(getattr(args, "backend_api_url", None))

    if args.max_models is not None and args.max_models > 0:
        candidates = candidates[: args.max_models]

    if not candidates:
        raise SystemExit("No g4f models found to check.")

    source_name = "backend-api" if getattr(args, "backend_api_url", None) else "registry"

    if getattr(args, "discover_only", False):
        payload = {
            "provider": provider_name or None,
            "source": source_name,
            "discovered_models": candidates,
            "discovered_count": len(candidates),
        }
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Discovered {len(candidates)} g4f candidate models (not probed):")
            for name in candidates:
                print(name)
        return

    if not args.list_only:
        print(
            f"[g4f-check] checking {len(candidates)} model(s) with prompt={args.prompt!r} "
            f"using AsyncClient concurrency={args.concurrency}..."
        )

    def _on_result(idx: int, total: int, result: Dict[str, Any]) -> None:
        if args.list_only:
            return
        status = "OK" if result.get("ok") else "FAIL"
        print(f"[{idx}/{total}] {result['model']}: {status} ({result['elapsed_s']:.2f}s) {result['detail']}")

    results = asyncio.run(
        _probe_g4f_models_async(
            candidates,
            timeout=float(args.timeout),
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            provider_name=provider_name or None,
            concurrency=int(args.concurrency),
            on_result=_on_result,
        )
    )
    working: List[str] = [str(r["model"]) for r in results if r.get("ok")]

    payload = {
        "provider": provider_name or None,
        "source": source_name,
        "probe_prompt": args.prompt,
        "working_models": working,
        "working_count": len(working),
        "checked_count": len(candidates),
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if args.list_only:
            for name in working:
                print(name)
            if not working:
                print("<none>")
        else:
            print("\nWorking g4f models:")
            if working:
                for name in working:
                    print(name)
            else:
                print("<none>")


def _resolve_competition_zip(spec: PipelineSpec) -> Optional[Path]:
    """Locate a competition ZIP bundled with the repo or placed next to the run.

    The uploaded competition archives are treated as the source of truth for
    test.csv/sample_submission.csv when present, because they reflect the exact
    submission schema used by the competition.
    """
    zip_candidates = [
        ROOT / "competitions" / spec.key / "data" / "source.zip",
        ROOT / "competitions" / spec.key / "data.zip",
        ROOT / "competition_files" / f"{spec.key}.zip",
        ROOT / "competition_files" / f"{spec.competition}.zip",
        Path.cwd() / f"{spec.key}.zip",
        Path.cwd() / f"{spec.competition}.zip",
    ]

    glob_candidates = []
    for base_dir in [ROOT / "competition_files", Path.cwd()]:
        if base_dir.exists():
            glob_candidates.extend(sorted(base_dir.glob(f"{spec.key}*.zip")))
            if spec.competition != spec.key:
                glob_candidates.extend(sorted(base_dir.glob(f"{spec.competition}*.zip")))

    return next((p for p in itertools.chain(zip_candidates, glob_candidates) if p.exists()), None)


def _extract_named_member_from_zip(zip_path: Path, member_name: str, out_path: Path) -> Optional[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        member = next((n for n in zf.namelist() if Path(n).name == member_name), None)
        if member is None:
            return None
        with zf.open(member) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return out_path


def _prefer_sample_submission_from_zip(spec: PipelineSpec) -> Optional[Path]:
    """Materialize sample_submission.csv from the competition ZIP when available.

    This keeps the local schema in sync with the actual competition bundle and
    avoids stale checked-in samples.
    """
    zip_path = _resolve_competition_zip(spec)
    if zip_path is None:
        return None
    target = ROOT / "competitions" / spec.key / "data" / "sample_submission.csv"
    try:
        return _extract_named_member_from_zip(zip_path, "sample_submission.csv", target)
    except Exception:
        return None


def _infer_format_slug_from_sample(sample_submission_csv: Path) -> Optional[str]:
    """Infer submission format from sample_submission.csv header."""
    header, _ = _read_csv_header_and_ids(sample_submission_csv)
    mapping = {
        ("initial_state_id", "path"): "format/initial_state_id+path",
        ("id", "permutation", "solution"): "format/id+permutation+solution",
        ("id", "moves"): "format/moves-dot",
        ("permutation", "solution"): "lrx-discover-math-gods-algorithm",
        ("n", "solution"): "lrx-oeis-a-186783-brainstorm-math-conjecture",
    }
    return mapping.get(tuple(header))


def _resolve_default_puzzles(spec: PipelineSpec) -> Path:
    """If --puzzles is omitted, try to locate a bundled test.csv.

    Resolution order:
      1) competitions/<spec.key>/data/test.csv
      2) competitions/<spec.competition>/data/test.csv

    If not found, we additionally try to *bootstrap* the data folder from a
    competition ZIP (Kaggle-style) if one is available locally. This is useful
    on Colab / ephemeral environments where you only have downloaded
    competition files as a .zip.

    ZIP discovery order (first hit wins):
      - competitions/<key>/data/source.zip
      - competitions/<key>/data.zip
      - competition_files/<key>.zip
      - competition_files/<competition>.zip
      - ./<key>.zip  (current working directory)
      - ./<competition>.zip

    If a ZIP is found, it is extracted into competitions/<key>/data/ and we
    re-check for test.csv.

    Notes:
      - Expected ZIP layout: test.csv and sample_submission.csv at ZIP root
        (common Kaggle pattern).
    """
    # 1) direct bundled paths
    candidate = ROOT / 'competitions' / spec.key / 'data' / 'test.csv'
    if candidate.exists():
        return candidate

    candidate = ROOT / 'competitions' / spec.competition / 'data' / 'test.csv'
    if candidate.exists():
        return candidate

    # 2) try bootstrap from a locally available ZIP
    data_dir = ROOT / 'competitions' / spec.key / 'data'
    zip_path = _resolve_competition_zip(spec)
    if zip_path is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(data_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to extract competition ZIP: {zip_path} ({e})") from e

        candidate = data_dir / 'test.csv'
        if candidate.exists():
            return candidate

    expected = ROOT / 'competitions' / spec.key / 'data' / 'test.csv'
    raise FileNotFoundError(
        f"No puzzles CSV provided and no bundled test.csv found for '{spec.key}'.\n"
        f"Expected: {expected}\n\n"
        "How to fix:\n"
        "  - Pass `--puzzles /path/to/test.csv`\n"
        "  - OR place competition files in `competitions/<slug>/data/`\n"
        "  - OR put a Kaggle-style ZIP with `test.csv` at repo root (./<slug>.zip)\n"
        "    or into `competition_files/<slug>.zip` and re-run.\n"
    )


def _resolve_sample_submission(spec: PipelineSpec) -> Optional[Path]:
    """Locate sample_submission.csv, preferring the competition ZIP when present."""
    from_zip = _prefer_sample_submission_from_zip(spec)
    if from_zip is not None and from_zip.exists():
        return from_zip

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


def _ensure_csv_field_size_limit() -> None:
    """Increase Python's CSV field size limit.

    Kaggle submissions for some competitions can contain extremely long fields
    (e.g. a huge moves/path string). The default csv module limit (131072 bytes)
    can raise:

        _csv.Error: field larger than field limit (131072)

    We raise the limit to the maximum supported by the current platform.
    """
    try:
        max_int = sys.maxsize
        while True:
            try:
                csv.field_size_limit(max_int)
                break
            except OverflowError:
                # On some platforms sys.maxsize can overflow a C long.
                max_int = int(max_int / 10)
    except Exception:
        # Best-effort only.
        pass


def _read_csv_header_and_ids(path: Path) -> tuple[list[str], list[str]]:
    _ensure_csv_field_size_limit()
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


def _load_allowed_moves_from_validator(validator_path: Path) -> Optional[set[str]]:
    """Best-effort load of ALLOWED move tokens from a competition validator."""
    try:
        mod_name = f"_validator_{validator_path.stem}_{abs(hash(str(validator_path)))}"
        spec = importlib.util.spec_from_file_location(mod_name, validator_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        allowed = getattr(module, "ALLOWED", None)
        if allowed is None:
            return None
        normalized = {str(x) for x in allowed if str(x)}
        return normalized or None
    except Exception:
        return None


def _validate_submission_move_tokens(
    *,
    submission_csv: Path,
    move_column: str,
    allowed_moves: set[str],
    joiner: str = ".",
) -> dict[str, Any]:
    """Validate that every move token in the built submission is competition-legal."""
    if not allowed_moves:
        return {}

    with submission_csv.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"submission CSV is empty: {submission_csv}")
        if move_column not in reader.fieldnames:
            raise ValueError(
                f"submission CSV is missing move column {move_column!r}. "
                f"Fields: {reader.fieldnames}"
            )

        rows_checked = 0
        non_empty_rows = 0
        max_tokens = 0
        for row_idx, row in enumerate(reader, 1):
            rows_checked += 1
            raw = str(row.get(move_column, '') or '').strip()
            if not raw:
                continue
            non_empty_rows += 1
            tokens = [tok.strip() for tok in raw.split(joiner)] if joiner else [raw]
            tokens = [tok for tok in tokens if tok]
            max_tokens = max(max_tokens, len(tokens))
            for token in tokens:
                if token == 'UNSOLVED':
                    continue
                if token not in allowed_moves:
                    raise ValueError(
                        f"Row {row_idx}: unknown move {token!r}. "
                        f"Allowed: {sorted(allowed_moves)}"
                    )

    return {
        'move_column': move_column,
        'allowed_moves': sorted(allowed_moves),
        'rows_checked': rows_checked,
        'rows_with_moves': non_empty_rows,
        'max_tokens_in_row': max_tokens,
    }


def _resolve_submission_move_column(competition_slug: str) -> tuple[str, str]:
    """Return (output_move_column, joiner) for a competition format."""
    _ensure_llm_puzzles_on_path()
    from src.comp_registry import get_config  # type: ignore

    cfg = get_config(competition_slug)
    headers = list(cfg.submission_headers or [])
    keys = list(cfg.header_keys or [])
    for header, key in zip(headers, keys):
        if key == 'moves':
            return str(header), str(cfg.move_joiner or '.')
    return str(cfg.moves_key or 'moves'), str(cfg.move_joiner or '.')


def _candidate_output_path(out_csv: Path) -> Path:
    return out_csv.with_name(out_csv.name + '.candidate')


def _backup_output_path(out_csv: Path) -> Path:
    return out_csv.with_name(out_csv.name + '.bak')


def _finalize_submission_output(candidate_csv: Path, out_csv: Path) -> None:
    if not candidate_csv.exists():
        raise FileNotFoundError(candidate_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        shutil.copyfile(out_csv, _backup_output_path(out_csv))
    candidate_csv.replace(out_csv)


def _format_kaggle_submit_error(exc: Exception, competition: str) -> str:
    status_code = getattr(getattr(exc, 'response', None), 'status_code', None)
    if status_code == 401:
        return (
            f"Kaggle returned 401 Unauthorized while submitting to '{competition}'. "
            "Most often this means the runtime has no valid Kaggle credentials, the token is stale, "
            "or the competition rules have not been accepted for this account. "
            "Pass --kaggle-json /path/to/kaggle.json (or ~/.kaggle/access_token), regenerate the token if needed, "
            "and make sure the account has joined the competition."
        )
    return f"Kaggle submission failed: {exc}"


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
            _ensure_csv_field_size_limit()
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
                _ensure_csv_field_size_limit()
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
            from src.kaggle_utils import ensure_auth, submit_file

            api = ensure_auth(kaggle_json_path=kaggle_json, config_dir=kaggle_config_dir)
            print(f"[kaggle] submitting via API: competition={competition} file={submission_csv}", flush=True)
            submit_file(api, competition=competition, filepath=str(submission_csv), message=message)
            _poll_kaggle_submission_status(competition, kaggle_json, kaggle_config_dir)
            return
        except Exception as e:
            if submit_via == 'api':
                raise SystemExit(_format_kaggle_submit_error(e, competition))
            print(f"[kaggle] API submit failed, falling back to CLI: {_format_kaggle_submit_error(e, competition)}", flush=True)

    env = os.environ.copy()
    if kaggle_json:
        try:
            _ensure_llm_puzzles_on_path()
            from src.kaggle_utils import build_kaggle_env
            env.update(build_kaggle_env(kaggle_json, config_dir=kaggle_config_dir))
        except Exception as e:
            print(f"[kaggle] could not prepare Kaggle credentials for CLI: {e}", flush=True)

    # Prefer the current official CLI syntax first; retry the legacy `-c` form
    # for environments that still ship the older kaggle package/CLI wrapper.
    cli_attempts = [
        _preferred_kaggle_cli_submit_cmd(competition, submission_csv, message),
        _legacy_kaggle_cli_submit_cmd(competition, submission_csv, message),
    ]
    last_error: Exception | None = None
    for idx, cmd in enumerate(cli_attempts, 1):
        print('[kaggle] ' + ' '.join(cmd), flush=True)
        try:
            subprocess.check_call(cmd, env=env)
            _poll_kaggle_submission_status(competition, kaggle_json, kaggle_config_dir)
            return
        except Exception as e:
            last_error = e
            if idx == len(cli_attempts):
                break
            print(f"[kaggle] CLI submit attempt {idx} failed, retrying with compatibility syntax: {e}", flush=True)
    assert last_error is not None
    raise SystemExit(_format_kaggle_submit_error(last_error, competition))


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


def _legacy_kaggle_cli_submit_cmd(competition: str, submission_csv: Path, message: str) -> list[str]:
    return ['kaggle', 'competitions', 'submit', '-c', competition, '-f', str(submission_csv), '-m', message]


def _preferred_kaggle_cli_submit_cmd(competition: str, submission_csv: Path, message: str) -> list[str]:
    # Current official kaggle-cli docs use a positional competition argument for
    # `competitions submit`, unlike some older community examples that used `-c`.
    return ['kaggle', 'competitions', 'submit', competition, '-f', str(submission_csv), '-m', message]


def _poll_kaggle_submission_status(competition: str, kaggle_json: str | None, kaggle_config_dir: str | None, wait_seconds: int = 45) -> None:
    """Best-effort polling of the latest Kaggle submission status.

    This helps catch the case where upload succeeded but Kaggle later marks the
    submission as a format/scoring error, which would explain why nothing shows
    on the leaderboard.
    """
    try:
        _ensure_llm_puzzles_on_path()
        from src.kaggle_utils import ensure_auth, latest_submission, wait_for_submission_result

        api = ensure_auth(kaggle_json_path=kaggle_json, config_dir=kaggle_config_dir)
        sub = latest_submission(api, competition)
        if not sub:
            print('[kaggle] WARNING: could not retrieve latest submission status after upload.', flush=True)
            return

        sid = sub.get('id') or sub.get('ref') or '?'
        status = sub.get('status') or sub.get('state') or 'unknown'
        print(f"[kaggle] latest submission right after upload: id={sid} status={status}", flush=True)

        final_sub = wait_for_submission_result(api, competition, target_ref=sid, wait_seconds=wait_seconds)
        if not final_sub:
            print('[kaggle] submission accepted by upload step; scoring is still pending.', flush=True)
            return

        sid = final_sub.get('id') or final_sub.get('ref') or '?'
        status = final_sub.get('status') or final_sub.get('state') or 'unknown'
        ps = final_sub.get('public_score')
        prs = final_sub.get('private_score')
        err = final_sub.get('error_description') or final_sub.get('errorDescription')
        msg = f"[kaggle] submission status: id={sid} status={status}"
        if ps not in (None, '', 'None') or prs not in (None, '', 'None'):
            msg += f" public={ps} private={prs}"
        if err not in (None, '', 'None'):
            msg += f" error={err}"
        print(msg, flush=True)
    except Exception as e:
        print(f"[kaggle] WARNING: could not poll submission status: {e}", flush=True)


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

    resolved_format_slug = competition_format_slug
    sample_submission_csv = _resolve_sample_submission(spec)
    if sample_submission_csv is not None:
        inferred = _infer_format_slug_from_sample(sample_submission_csv)
        if inferred and inferred != competition_format_slug:
            print(
                f"[submit] overriding format from {competition_format_slug} -> {inferred} "
                f"based on sample_submission header in {sample_submission_csv}",
                flush=True,
            )
            resolved_format_slug = inferred

    print(f"[submit] Building submission for format={resolved_format_slug}")
    print(f"         puzzles={puzzles_csv}")
    print(f"         output={out_csv}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    lp_build_submission(
        puzzles_csv=str(puzzles_csv),
        output_csv=str(out_csv),
        competition=resolved_format_slug,
        solver=row_solver,
        max_rows=max_rows,
        progress=(not no_progress),
        progress_desc=f"{spec.key}: building submission",
    )


# ---------------------------------------------------------------------------
# AgentLaboratory generation
# ---------------------------------------------------------------------------


def _memory_env_for_codegen(models: str) -> dict[str, str]:
    env: dict[str, str] = {}
    normalized = [m.strip().lower() for m in (models or "").split(",") if m.strip()]
    has_remote = any(not m.startswith("local:") for m in normalized)
    if has_remote:
        env.setdefault("AGENTLAB_REMOTE_SUBPROCESS", os.getenv("AGENTLAB_REMOTE_SUBPROCESS", "1"))
        env.setdefault("AGENTLAB_DISABLE_TOKEN_COUNT", os.getenv("AGENTLAB_DISABLE_TOKEN_COUNT", "1"))
        env.setdefault("AGENTLAB_MAX_RESPONSE_CHARS", os.getenv("AGENTLAB_MAX_RESPONSE_CHARS", "40000"))
        env.setdefault("AGENTLAB_G4F_STOP_AT_PYTHON_FENCE", os.getenv("AGENTLAB_G4F_STOP_AT_PYTHON_FENCE", "1"))
        env.setdefault("AGENTLAB_ARTIFACT_SPILL_CHARS", os.getenv("AGENTLAB_ARTIFACT_SPILL_CHARS", "8000"))
        env.setdefault("AGENTLAB_HEAVY_IMPORTS", os.getenv("AGENTLAB_HEAVY_IMPORTS", "0"))
        env.setdefault("AGENTLAB_G4F_MODEL_SWITCH_SLEEP_S", os.getenv("AGENTLAB_G4F_MODEL_SWITCH_SLEEP_S", "1.5"))
        env.setdefault("AGENTLAB_G4F_REPEAT_MODEL_SLEEP_S", os.getenv("AGENTLAB_G4F_REPEAT_MODEL_SLEEP_S", "0"))
        env.setdefault("MALLOC_ARENA_MAX", os.getenv("MALLOC_ARENA_MAX", "2"))
    return env


def _agent_model_cli_args(*, agent_models: str | None = None, planner_models: str | None = None, coder_models: str | None = None, fixer_models: str | None = None) -> list[str]:
    args: list[str] = []
    if agent_models:
        args.extend(["--agent-models", agent_models])
    if planner_models:
        args.extend(["--planner-models", planner_models])
    if coder_models:
        args.extend(["--coder-models", coder_models])
    if fixer_models:
        args.extend(["--fixer-models", fixer_models])
    return args


def _run_agent_laboratory(
    *,
    prompt_file: Path,
    out_path: Path,
    validator: Path,
    baseline: Path,
    custom_prompts: Optional[Path] = None,
    llm: str = "gpt-4o-mini",
    agent_models: str | None = None,
    planner_models: str | None = None,
    coder_models: str | None = None,
    fixer_models: str | None = None,
    max_iters: int = 8,
    no_llm: bool = False,
    allow_baseline: bool = True,
    g4f_recovery_rounds: int | None = None,
    g4f_recovery_max_iters: int | None = None,
    g4f_recovery_sleep: float | None = None,
    g4f_switch_sleep: float | None = None,
    g4f_repeat_model_sleep: float | None = None,
    print_generation: bool = False,
    print_generation_max_chars: int | None = None,
    worker_no_kill_process_group: bool = False,
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
    cmd.extend(
        _agent_model_cli_args(
            agent_models=agent_models,
            planner_models=planner_models,
            coder_models=coder_models,
            fixer_models=fixer_models,
        )
    )

    # run_perm_pipeline.py already falls back to baseline unless --strict is used.
    if no_llm:
        cmd.append("--no-llm")
    if custom_prompts:
        cmd.extend(["--custom-prompts", str(custom_prompts)])

    env = os.environ.copy()
    effective_codegen_env = _memory_env_for_codegen(llm)
    env.update(effective_codegen_env)
    if g4f_recovery_rounds is not None:
        env["AGENTLAB_G4F_RECOVERY_ROUNDS"] = str(max(0, int(g4f_recovery_rounds)))
        effective_codegen_env["AGENTLAB_G4F_RECOVERY_ROUNDS"] = env["AGENTLAB_G4F_RECOVERY_ROUNDS"]
    if g4f_recovery_max_iters is not None:
        env["AGENTLAB_G4F_RECOVERY_MAX_ITERS"] = str(max(1, int(g4f_recovery_max_iters)))
        effective_codegen_env["AGENTLAB_G4F_RECOVERY_MAX_ITERS"] = env["AGENTLAB_G4F_RECOVERY_MAX_ITERS"]
    if g4f_recovery_sleep is not None:
        env["AGENTLAB_G4F_RECOVERY_SLEEP_S"] = str(max(0.0, float(g4f_recovery_sleep)))
        effective_codegen_env["AGENTLAB_G4F_RECOVERY_SLEEP_S"] = env["AGENTLAB_G4F_RECOVERY_SLEEP_S"]
    if g4f_switch_sleep is not None:
        env["AGENTLAB_G4F_MODEL_SWITCH_SLEEP_S"] = str(max(0.0, float(g4f_switch_sleep)))
        effective_codegen_env["AGENTLAB_G4F_MODEL_SWITCH_SLEEP_S"] = env["AGENTLAB_G4F_MODEL_SWITCH_SLEEP_S"]
    if g4f_repeat_model_sleep is not None:
        env["AGENTLAB_G4F_REPEAT_MODEL_SLEEP_S"] = str(max(0.0, float(g4f_repeat_model_sleep)))
        effective_codegen_env["AGENTLAB_G4F_REPEAT_MODEL_SLEEP_S"] = env["AGENTLAB_G4F_REPEAT_MODEL_SLEEP_S"]
    if print_generation:
        env["AGENTLAB_PRINT_GENERATION"] = "1"
        effective_codegen_env["AGENTLAB_PRINT_GENERATION"] = "1"
    if print_generation_max_chars is not None:
        env["AGENTLAB_PRINT_GENERATION_MAX_CHARS"] = str(max(0, int(print_generation_max_chars)))
        effective_codegen_env["AGENTLAB_PRINT_GENERATION_MAX_CHARS"] = env["AGENTLAB_PRINT_GENERATION_MAX_CHARS"]
    if worker_no_kill_process_group:
        env["AGENTLAB_WORKER_KILL_PROCESS_GROUP"] = "0"
        effective_codegen_env["AGENTLAB_WORKER_KILL_PROCESS_GROUP"] = "0"
    print("[agentlab] " + " ".join(cmd))
    if effective_codegen_env:
        print("[agentlab] low-RAM env: " + ", ".join(f"{k}={effective_codegen_env[k]}" for k in sorted(effective_codegen_env.keys())))
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_list_pipelines(_: argparse.Namespace) -> None:
    print("Available pipelines (competition slugs):")
    for spec in list_pipelines():
        print(f"- {spec.key:45s}  (competition='{spec.competition}', format='{spec.format_slug}')")




def cmd_show_pipeline(args: argparse.Namespace) -> None:
    """Print a human-friendly description of a pipeline (and optionally JSON).

    Includes:
    - all configured paths (baseline, validator, prompts)
    - bundled test/sample_submission locations (if any)
    - llm-puzzles submission format (expected columns, id field, joiner)
    """
    spec = get_pipeline(args.competition) if args.competition else None
    if spec is None:
        raise SystemExit(
            "Unknown --competition. Run `python pipeline_cli.py list-pipelines` to see supported pipelines."
        )

    format_slug = args.format or spec.format_slug

    # Bundle discovery (best-effort; do not crash)
    default_puzzles: Path | None
    try:
        default_puzzles = _resolve_default_puzzles(spec)
    except Exception:
        default_puzzles = None

    sample: Path | None
    try:
        sample = _resolve_sample_submission(spec)
    except Exception:
        sample = None

    # llm-puzzles formatting rules
    cfg_dict: dict[str, Any] = {}
    try:
        _ensure_llm_puzzles_on_path()
        from src.comp_registry import get_config  # type: ignore

        cfg = get_config(format_slug)
        cfg_dict = {
            "slug": cfg.slug,
            "submission_headers": list(cfg.submission_headers or []),
            "header_keys": list(cfg.header_keys or []),
            "puzzles_id_field": cfg.puzzles_id_field,
            "moves_key": cfg.moves_key,
            "move_joiner": cfg.move_joiner,
        }
    except Exception as e:
        cfg_dict = {"error": f"could not load llm-puzzles comp_registry: {e}"}

    def rel(p: Path | None) -> str | None:
        if p is None:
            return None
        try:
            return str(p.relative_to(ROOT))
        except Exception:
            return str(p)

    def exists(p: Path | None) -> bool:
        return bool(p and p.exists())

    sample_header: list[str] | None = None
    if sample and sample.exists():
        try:
            sample_header, _ = _read_csv_header_and_ids(sample)
        except Exception:
            sample_header = None

    record: dict[str, Any] = {
        "pipeline": {
            "key": spec.key,
            "competition": spec.competition,
            "format_slug": format_slug,
            "state_columns": list(spec.state_columns or []),
            "smoke_vector": list(spec.smoke_vector or []),
        },
        "paths": {
            "baseline_solver": {"path": rel(spec.baseline_solver), "exists": spec.baseline_solver.exists()},
            "validator": {"path": rel(spec.validator), "exists": spec.validator.exists()},
            "prompt_file": {"path": rel(spec.prompt_file) if spec.prompt_file else None, "exists": exists(spec.prompt_file) if spec.prompt_file else False},
            "custom_prompts_file": {"path": rel(spec.custom_prompts_file) if spec.custom_prompts_file else None, "exists": exists(spec.custom_prompts_file) if spec.custom_prompts_file else False},
        },
        "bundled_files": {
            "default_puzzles_csv": {"path": rel(default_puzzles) if default_puzzles else None, "exists": exists(default_puzzles)},
            "sample_submission_csv": {"path": rel(sample) if sample else None, "exists": exists(sample), "header": sample_header},
        },
        "submission_format": cfg_dict,
    }

    # Extra directory listing for convenience (non-recursive)
    comp_dir = ROOT / 'competitions' / spec.key

    def list_dir(d: Path) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not d.exists() or not d.is_dir():
            return out
        for p in sorted(d.iterdir()):
            if p.name.startswith('.'):
                continue
            try:
                st = p.stat()
                out.append({"name": p.name, "is_dir": p.is_dir(), "bytes": int(st.st_size)})
            except Exception:
                out.append({"name": p.name, "is_dir": p.is_dir()})
        return out

    record['bundled_files']['competition_dir'] = {
        'path': rel(comp_dir),
        'exists': comp_dir.exists(),
        'data': list_dir(comp_dir / 'data'),
        'submissions': list_dir(comp_dir / 'submissions'),
        'prompts': list_dir(comp_dir / 'prompts'),
    }

    if args.json:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return

    # Human-friendly print
    print(f"\nPipeline: {spec.key}")
    print(f"  Kaggle competition slug: {spec.competition}")
    print(f"  Submission format slug: {format_slug}")

    print("\nSubmission format (llm-puzzles):")
    if 'error' in cfg_dict:
        print(f"  [warn] {cfg_dict['error']}")
    else:
        print(f"  expected columns: {cfg_dict.get('submission_headers')}")
        print(f"  maps from keys:   {cfg_dict.get('header_keys')}")
        print(f"  id field in puzzles/test.csv: {cfg_dict.get('puzzles_id_field')}")
        print(f"  moves column in submission:   {cfg_dict.get('moves_key')}")
        j = cfg_dict.get('move_joiner')
        print(f"  move joiner: {repr(j)}")

    print("\nConfigured files:")
    print(f"  baseline solver: {rel(spec.baseline_solver)}  (exists={spec.baseline_solver.exists()})")
    print(f"  validator:       {rel(spec.validator)}  (exists={spec.validator.exists()})")
    print(f"  prompt file:     {rel(spec.prompt_file) if spec.prompt_file else '(none)'}  (exists={exists(spec.prompt_file)})")
    print(f"  custom prompts:  {rel(spec.custom_prompts_file) if spec.custom_prompts_file else '(none)'}  (exists={exists(spec.custom_prompts_file)})")

    print("\nBundled inputs:")
    print(f"  default puzzles CSV: {rel(default_puzzles) if default_puzzles else '(not found)'}  (exists={exists(default_puzzles)})")
    print(f"  sample_submission.csv: {rel(sample) if sample else '(not found)'}  (exists={exists(sample)})")
    if sample_header:
        print(f"  sample header: {sample_header}")

    print("\nState extraction:")
    print(f"  candidate state columns: {list(spec.state_columns or [])}")
    print(f"  smoke vector: {list(spec.smoke_vector or [])}")

    print("\nBundled competition directory:")
    print(f"  {rel(comp_dir)}  (exists={comp_dir.exists()})")
    for subname in ['data', 'submissions', 'prompts']:
        items = record['bundled_files']['competition_dir'].get(subname, [])
        if items:
            print(f"  - {subname}/")
            for it in items:
                suffix = '/' if it.get('is_dir') else ''
                b = it.get('bytes', None)
                if b is not None:
                    print(f"      {it['name']}{suffix}  ({b} bytes)")
                else:
                    print(f"      {it['name']}{suffix}")
        else:
            print(f"  - {subname}/  (empty or missing)")

    print("\nTip:")
    print(f"  Show as JSON:     python pipeline_cli.py show-pipeline --competition {spec.key} --json")
    print(f"  Build submission: python pipeline_cli.py build-submission --competition {spec.key} --solver {rel(spec.baseline_solver)} --output submissions/submission.csv")
    if spec.prompt_file:
        print(f"  Generate solver:  python pipeline_cli.py generate-solver --competition {spec.key} --out generated/solve_{spec.key}.py")


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

    _gpu_diag_hint(args.models)

    _run_agent_laboratory(
        prompt_file=prompt_file,
        out_path=out_path,
        validator=spec.validator,
        baseline=spec.baseline_solver,
        custom_prompts=custom_prompts,
        llm=args.models,
        agent_models=args.agent_models,
        planner_models=args.planner_models,
        coder_models=args.coder_models,
        fixer_models=args.fixer_models,
        max_iters=args.max_iters,
        no_llm=False,
        allow_baseline=args.allow_baseline,
        g4f_recovery_rounds=args.g4f_recovery_rounds,
        g4f_recovery_max_iters=args.g4f_recovery_max_iters,
        g4f_recovery_sleep=args.g4f_recovery_sleep,
        g4f_switch_sleep=args.g4f_switch_sleep,
        g4f_repeat_model_sleep=args.g4f_repeat_model_sleep,
        print_generation=args.print_generation,
        print_generation_max_chars=args.print_generation_max_chars,
        worker_no_kill_process_group=args.worker_no_kill_process_group,
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
    candidate_out = _candidate_output_path(out_csv)

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
        if candidate_out.exists():
            candidate_out.unlink()

        t2 = _stage("build submission")
        report["stages"]["build_submission"] = {"start": time.time()}
        _build_submission(
            puzzles_csv=puzzles_csv,
            out_csv=candidate_out,
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

        allowed_moves = _load_allowed_moves_from_validator(spec.validator)
        if allowed_moves:
            tmc = _stage("content check")
            report["stages"]["content_check"] = {"start": time.time()}
            move_column, joiner = _resolve_submission_move_column(args.format or spec.competition)
            move_stats = _validate_submission_move_tokens(
                submission_csv=candidate_out,
                move_column=move_column,
                allowed_moves=allowed_moves,
                joiner=joiner,
            )
            report["move_validation"] = move_stats
            report["stages"]["content_check"]["end"] = time.time()
            report["stages"]["content_check"]["seconds"] = report["stages"]["content_check"]["end"] - report["stages"]["content_check"]["start"]
            _stage_done("content check", tmc)

        if args.schema_check and not args.no_schema_check:
            sample = _resolve_sample_submission(spec)
            if sample is None:
                print(f"[schema] WARNING: no bundled sample_submission.csv found for '{spec.key}'. Skipping.")
            else:
                tsc = _stage("schema check")
                report["stages"]["schema_check"] = {"start": time.time()}
                stats = _validate_submission_schema(
                    submission_csv=candidate_out,
                    sample_submission_csv=sample,
                    check_ids=(not args.no_schema_check_ids),
                )
                report["schema"] = {"sample": str(sample), **stats}
                report["stages"]["schema_check"]["end"] = time.time()
                report["stages"]["schema_check"]["seconds"] = report["stages"]["schema_check"]["end"] - report["stages"]["schema_check"]["start"]
                _stage_done("schema check", tsc)

        _finalize_submission_output(candidate_out, out_csv)
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

    # Helpful UX: explain when GPU will/won't be used.
    _gpu_diag_hint(args.models)

    puzzles_csv = Path(args.puzzles) if args.puzzles else _resolve_default_puzzles(spec)
    out_csv = Path(args.output)
    candidate_out = _candidate_output_path(out_csv)
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
            "agent_models": args.agent_models,
            "planner_models": args.planner_models,
            "coder_models": args.coder_models,
            "fixer_models": args.fixer_models,
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
                agent_models=args.agent_models,
                planner_models=args.planner_models,
                coder_models=args.coder_models,
                fixer_models=args.fixer_models,
                max_iters=args.max_iters,
                no_llm=False,
                allow_baseline=args.allow_baseline,
                g4f_recovery_rounds=args.g4f_recovery_rounds,
                g4f_recovery_max_iters=args.g4f_recovery_max_iters,
                g4f_recovery_sleep=args.g4f_recovery_sleep,
                g4f_switch_sleep=args.g4f_switch_sleep,
                g4f_repeat_model_sleep=args.g4f_repeat_model_sleep,
                print_generation=args.print_generation,
                print_generation_max_chars=args.print_generation_max_chars,
                worker_no_kill_process_group=args.worker_no_kill_process_group,
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
        if candidate_out.exists():
            candidate_out.unlink()

        t2 = _stage("build submission")
        report["stages"]["build_submission"] = {"start": time.time()}
        _build_submission(
            puzzles_csv=puzzles_csv,
            out_csv=candidate_out,
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

        allowed_moves = _load_allowed_moves_from_validator(spec.validator)
        if allowed_moves:
            tmc = _stage("content check")
            report["stages"]["content_check"] = {"start": time.time()}
            move_column, joiner = _resolve_submission_move_column(args.format or spec.competition)
            move_stats = _validate_submission_move_tokens(
                submission_csv=candidate_out,
                move_column=move_column,
                allowed_moves=allowed_moves,
                joiner=joiner,
            )
            report["move_validation"] = move_stats
            report["stages"]["content_check"]["end"] = time.time()
            report["stages"]["content_check"]["seconds"] = report["stages"]["content_check"]["end"] - report["stages"]["content_check"]["start"]
            _stage_done("content check", tmc)

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
                    submission_csv=candidate_out,
                    sample_submission_csv=sample,
                    check_ids=(not args.no_schema_check_ids),
                )
                report["schema"] = {"sample": str(sample), **stats}
                report["stages"]["schema_check"]["end"] = time.time()
                report["stages"]["schema_check"]["seconds"] = report["stages"]["schema_check"]["end"] - report["stages"]["schema_check"]["start"]
                _stage_done("schema check", tsc)

        _finalize_submission_output(candidate_out, out_csv)

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
        "demo-bubble-sort",
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

    sp = sub.add_parser("show-pipeline", help="Show details for a single pipeline (paths, bundled files, submission schema)")
    sp.add_argument("--competition", required=True, help="Competition slug / pipeline key")
    sp.add_argument("--format", default=None, help="Override llm-puzzles format slug (for inspection)")
    sp.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    sp.add_argument("--concurrency", type=int, default=5, help="Maximum number of concurrent AsyncClient probes")
    sp.set_defaults(func=cmd_show_pipeline)

    sp = sub.add_parser("generate-solver", help="Generate/repair a solver with AgentLaboratory")
    sp.add_argument("--competition", required=True, help="Competition slug / pipeline key")
    sp.add_argument("--out", required=True, help="Output path for generated solver")
    sp.add_argument("--prompt-file", default=None, help="Override user prompt file")
    sp.add_argument("--custom-prompts", default=None, help="Override AgentLaboratory custom prompts JSON")
    sp.add_argument(
        "--models",
        dest="models",
        default="gpt-4o-mini",
        help=(
            "Comma-separated model list (passed to AgentLaboratory --models). "
            "Bare names use g4f backend (remote providers). "
            "To use local GPU inference, pass items like 'local:<hf_model_id>'."
        ),
    )
    sp.add_argument("--llm", dest="models", default=None, help=argparse.SUPPRESS)
    sp.add_argument("--agent-models", default=None, help="Optional per-agent override mapping for AgentLaboratory, e.g. 'planner=gpt-4;coder=local:Qwen/Qwen2.5-Coder-1.5B;fixer=gpt-4o-mini'.")
    sp.add_argument("--planner-models", default=None, help="Optional model list override for the planner agent.")
    sp.add_argument("--coder-models", default=None, help="Optional model list override for the coder agent.")
    sp.add_argument("--fixer-models", default=None, help="Optional model list override for the fixer agent.")
    sp.add_argument("--max-iters", type=int, default=8)
    sp.add_argument("--g4f-recovery-rounds", type=int, default=None, help="Extra recovery rounds before offline fallback (forwarded to AgentLaboratory).")
    sp.add_argument("--g4f-recovery-max-iters", type=int, default=None, help="Fixer iterations per recovery round (forwarded to AgentLaboratory).")
    sp.add_argument("--g4f-recovery-sleep", type=float, default=None, help="Cooldown in seconds before each recovery round (forwarded to AgentLaboratory).")
    sp.add_argument("--g4f-switch-sleep", type=float, default=None, help="Cooldown in seconds before switching between remote g4f models.")
    sp.add_argument("--g4f-repeat-model-sleep", type=float, default=None, help="Cooldown in seconds before repeating the same remote g4f model.")
    sp.add_argument("--print-generation", action="store_true", help="Print raw planner/coder/fixer generations.")
    sp.add_argument("--print-generation-max-chars", type=int, default=None, help="Maximum chars to print per generation (0 disables clipping).")
    sp.add_argument("--worker-no-kill-process-group", action="store_true", help="Do not hard-kill the entire worker process group on timeout; only terminate the worker process itself.")
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
    sp.add_argument(
        "--models",
        dest="models",
        default="gpt-4o-mini",
        help=(
            "Comma-separated model list (passed to AgentLaboratory --models). "
            "Bare names use g4f backend (remote providers). "
            "To use local GPU inference, pass items like 'local:<hf_model_id>'."
        ),
    )
    sp.add_argument("--llm", dest="models", default=None, help=argparse.SUPPRESS)
    sp.add_argument("--agent-models", default=None, help="Optional per-agent override mapping for AgentLaboratory, e.g. 'planner=gpt-4;coder=local:Qwen/Qwen2.5-Coder-1.5B;fixer=gpt-4o-mini'.")
    sp.add_argument("--planner-models", default=None, help="Optional model list override for the planner agent.")
    sp.add_argument("--coder-models", default=None, help="Optional model list override for the coder agent.")
    sp.add_argument("--fixer-models", default=None, help="Optional model list override for the fixer agent.")
    sp.add_argument("--max-iters", type=int, default=8)
    sp.add_argument("--g4f-recovery-rounds", type=int, default=None, help="Extra recovery rounds before offline fallback (forwarded to AgentLaboratory).")
    sp.add_argument("--g4f-recovery-max-iters", type=int, default=None, help="Fixer iterations per recovery round (forwarded to AgentLaboratory).")
    sp.add_argument("--g4f-recovery-sleep", type=float, default=None, help="Cooldown in seconds before each recovery round (forwarded to AgentLaboratory).")
    sp.add_argument("--g4f-switch-sleep", type=float, default=None, help="Cooldown in seconds before switching between remote g4f models.")
    sp.add_argument("--g4f-repeat-model-sleep", type=float, default=None, help="Cooldown in seconds before repeating the same remote g4f model.")
    sp.add_argument("--print-generation", action="store_true", help="Print raw planner/coder/fixer generations.")
    sp.add_argument("--print-generation-max-chars", type=int, default=None, help="Maximum chars to print per generation (0 disables clipping).")
    sp.add_argument("--worker-no-kill-process-group", action="store_true", help="Do not hard-kill the entire worker process group on timeout; only terminate the worker process itself.")
    sp.add_argument("--allow-baseline", action="store_true")
    sp.add_argument("--no-llm", action="store_true")
    sp.add_argument("--format", default=None, help="Override llm-puzzles format slug")
    sp.add_argument("--vector-col", default=None, help="Override state column")
    sp.add_argument("--max-rows", type=int, default=None)
    sp.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    sp.add_argument("--submit", action="store_true")
    sp.add_argument("--message", default=None, help="Kaggle submission message")
    sp.add_argument("--kaggle-json", default=None, help="Path to a Kaggle credentials file (legacy kaggle.json or access_token). If set, credentials are loaded for both API and CLI submission paths.")
    sp.add_argument("--kaggle-config-dir", default=None, help="Optional directory to place a temporary kaggle.json copy")
    sp.add_argument("--submit-via", default="auto", choices=["auto","api","cli"], help="How to submit: auto (try API then CLI), api, or cli")
    sp.add_argument("--submit-competition", dest="submit_competition", default=None, help="Override Kaggle competition slug for submission")
    sp.add_argument("--schema-check", action="store_true", help="Compare output schema to bundled sample_submission.csv")
    sp.add_argument("--no-schema-check", action="store_true", help="Disable schema check (auto-enabled before --submit)")
    sp.add_argument("--no-schema-check-ids", action="store_true", help="Skip id set comparison during schema check")
    sp.add_argument("--run-log", default=None, help="Path to run_log.json (default: <output_dir>/run_log.json)")
    sp.add_argument("--no-run-log", action="store_true", help="Disable writing run_log.json")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("check-g4f-models", help="Probe g4f chat models and print the working subset")
    sp.add_argument("--models", default=None, help="Optional comma-separated candidate list. If omitted, auto-discovers text models from g4f registry or backend API.")
    sp.add_argument("--provider", default=None, help="Optional g4f provider name (same as G4F_PROVIDER).")
    sp.add_argument("--backend-api-url", default=None, help="Optional base URL of a running g4f backend API; model list is fetched from /backend-api/v2/models.")
    sp.add_argument("--timeout", type=float, default=12.0, help="Per-model probe timeout in seconds")
    sp.add_argument("--max-models", type=int, default=None, help="Optional cap on number of models to check")
    sp.add_argument("--prompt", default="ping", help="Probe user prompt")
    sp.add_argument("--system-prompt", default="Return a very short plain-text reply.", help="Probe system prompt")
    sp.add_argument("--list-only", action="store_true", help="Probe models but print only the models that returned a non-empty answer")
    sp.add_argument("--discover-only", action="store_true", help="Only list discovered candidate models without probing them")
    sp.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    sp.add_argument("--concurrency", type=int, default=5, help="Maximum number of concurrent AsyncClient probes")
    sp.set_defaults(func=cmd_check_g4f_models)

    sp = sub.add_parser("selftest", help="Offline smoke tests")
    sp.set_defaults(func=cmd_selftest)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
