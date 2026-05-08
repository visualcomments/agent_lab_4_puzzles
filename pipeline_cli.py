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
import hashlib
import itertools
import importlib
import importlib.util
import json
import py_compile
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
import re

from pipeline_registry import PipelineSpec, get_pipeline, list_pipelines
import llm_code_contract as code_contract


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
_AGENTLAB_INFERENCE_MODULE = None


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
    - GPU is only used inside this runtime for `local:*` Transformers inference.
    - Backends like `ollama:*`, `vllm:*`, `lmstudio:*`, `openai-compatible:*`, and `g4fapi:*`
      may still use GPU, but on the external/local server process rather than in this Python process.
    """
    dev = (os.getenv("AGENTLAB_DEVICE") or "").strip()
    use_gpu = (os.getenv("AGENTLAB_USE_GPU") or "").strip().lower() in {"1", "true", "yes", "on"}
    if not dev and not use_gpu:
        return

    models = [m.strip() for m in (selected_models or "").split(",") if m.strip()]
    has_local = any(m.startswith("local:") for m in models)
    has_external_local_server = any(
        m.startswith(prefix)
        for m in models
        for prefix in ("ollama:", "vllm:", "lmstudio:", "openai-compatible:", "openai_compatible:", "compat:", "g4fapi:")
    )

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

    if not has_local and not has_external_local_server:
        print(
            "[gpu] NOTE: your --models list contains no 'local:*' models. "
            "g4f/OpenAI/Claude/Gemini backends run remotely, so local GPU memory will stay near 0. "
            "If you want in-process GPU inference, pass e.g. --models 'local:Qwen/Qwen2.5-0.5B-Instruct' (and ensure torch+transformers are installed)."
        )
    elif has_external_local_server and not has_local:
        print(
            "[gpu] NOTE: your selected models use an external/local server backend (ollama/vllm/LM Studio/OpenAI-compatible/g4fapi). "
            "GPU may be used by that server, but this Python process will still appear mostly CPU-only."
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
        "r1-1776": 0,
        "gpt-4o": 1,
        "gpt-4": 2,
        "claude-3.5-sonnet": 3,
        "command-r-plus": 4,
        "deepseek-chat": 5,
        "command-r": 6,
        "aria": 7,
        "command-a": 8,
        "command-r7b": 9,
        "gpt-4o-mini": 10,
        "claude-3-haiku": 11,
    }
    deduped = _dedupe_keep_order(names)
    return sorted(deduped, key=lambda s: (preferred.get(s, 9999), s.lower()))


def _load_agentlab_inference_module():
    global _AGENTLAB_INFERENCE_MODULE
    if _AGENTLAB_INFERENCE_MODULE is not None:
        return _AGENTLAB_INFERENCE_MODULE
    module_path = ROOT / "AgentLaboratory" / "inference.py"
    if not module_path.exists():
        raise RuntimeError(f"AgentLaboratory inference module not found: {module_path}")
    agentlab_root = module_path.parent
    if str(agentlab_root) not in sys.path:
        sys.path.insert(0, str(agentlab_root))
    spec = importlib.util.spec_from_file_location("agentlab_inference_cli", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _AGENTLAB_INFERENCE_MODULE = module
    return module


def _probe_g4f_model_pipeline(
    model: str,
    timeout: float,
    prompt: str,
    system_prompt: str,
    provider_name: Optional[str] = None,
) -> Tuple[bool, str, float]:
    inference = _load_agentlab_inference_module()
    normalized = _normalize_g4f_model_name(model)
    model_str = f"g4f:{normalized}"
    started = time.time()
    old_provider = os.environ.get("G4F_PROVIDER")
    if provider_name:
        os.environ["G4F_PROVIDER"] = provider_name
    try:
        response = inference.query_model_stable(
            model_str=model_str,
            prompt=prompt,
            system_prompt=system_prompt,
            tries=1,
            timeout=float(timeout),
            print_cost=False,
        )
        elapsed = time.time() - started
        txt = str(response or "").strip()
        if not txt:
            return False, "empty response", elapsed
        return True, txt.replace("\n", " ")[:80], elapsed
    except Exception as exc:
        elapsed = time.time() - started
        return False, str(exc), elapsed
    finally:
        if provider_name:
            if old_provider is None:
                os.environ.pop("G4F_PROVIDER", None)
            else:
                os.environ["G4F_PROVIDER"] = old_provider





def _split_accidental_joined_kaggle_token(argv: Sequence[str]) -> List[str]:
    """Split tokens like ``4000kaggle competitions submit ...``.

    This catches the common shell line-continuation mistake where a trailing
    backslash joins the next line directly onto the previous token.
    """
    out: List[str] = []
    for idx, token in enumerate(argv):
        if (
            token != "kaggle"
            and token.endswith("kaggle")
            and idx + 2 < len(argv)
            and argv[idx + 1] == "competitions"
            and argv[idx + 2] == "submit"
        ):
            prefix = token[: -len("kaggle")]
            if prefix:
                out.append(prefix)
            out.append("kaggle")
            continue
        out.append(token)
    return out


def _find_option_value(argv: Sequence[str], *names: str) -> Optional[str]:
    for idx, token in enumerate(argv[:-1]):
        if token in names:
            return argv[idx + 1]
    return None


def _replace_option_value(argv: List[str], names: Sequence[str], value: str) -> List[str]:
    out = list(argv)
    for idx, token in enumerate(out[:-1]):
        if token in names:
            out[idx + 1] = value
            return out
    if names:
        out.extend([names[0], value])
    return out


def _parse_embedded_kaggle_submit_tail(tail: Sequence[str]) -> Dict[str, str]:
    if len(tail) < 3 or list(tail[:3]) != ["kaggle", "competitions", "submit"]:
        raise ValueError("not a kaggle competitions submit tail")

    competition: Optional[str] = None
    file_path: Optional[str] = None
    message: Optional[str] = None
    idx = 3
    while idx < len(tail):
        tok = tail[idx]
        if tok in {"-c", "--competition"}:
            if idx + 1 >= len(tail):
                raise ValueError("missing value after -c/--competition")
            competition = tail[idx + 1]
            idx += 2
            continue
        if tok in {"-f", "--file"}:
            if idx + 1 >= len(tail):
                raise ValueError("missing value after -f/--file")
            file_path = tail[idx + 1]
            idx += 2
            continue
        if tok in {"-m", "--message"}:
            if idx + 1 >= len(tail):
                raise ValueError("missing value after -m/--message")
            message = tail[idx + 1]
            idx += 2
            continue
        if tok.startswith("-"):
            raise ValueError(f"unsupported kaggle submit option: {tok}")
        if competition is None:
            competition = tok
            idx += 1
            continue
        raise ValueError(f"unexpected extra kaggle submit token: {tok}")

    if not competition:
        raise ValueError("missing Kaggle competition slug")
    if not file_path:
        raise ValueError("missing Kaggle submission file (-f)")
    if not message:
        raise ValueError("missing Kaggle submission message (-m)")
    return {"competition": competition, "file": file_path, "message": message}


def _rewrite_embedded_kaggle_submit(argv: Sequence[str]) -> Tuple[List[str], Optional[str]]:
    """Rewrite accidental ``... run ... kaggle competitions submit ...`` tails.

    Returns ``(argv, note)`` where ``note`` is a human-readable explanation if
    a rewrite took place.
    """
    normalized = _split_accidental_joined_kaggle_token(list(argv))
    if not normalized or normalized[0] != "run":
        return normalized, None

    start = -1
    for idx in range(1, len(normalized) - 2):
        if normalized[idx:idx + 3] == ["kaggle", "competitions", "submit"]:
            start = idx
            break
    if start < 0:
        return normalized, None

    base = normalized[:start]
    tail = normalized[start:]
    parsed = _parse_embedded_kaggle_submit_tail(tail)

    requested_output = _find_option_value(base, "--output")
    if requested_output and requested_output != parsed["file"]:
        base = _replace_option_value(base, ["--output"], parsed["file"])
    elif not requested_output:
        base.extend(["--output", parsed["file"]])

    if "--submit" not in base:
        base.append("--submit")
    if "--message" in base:
        base = _replace_option_value(base, ["--message"], parsed["message"])
    else:
        base.extend(["--message", parsed["message"]])

    if "--submit-competition" in base:
        base = _replace_option_value(base, ["--submit-competition"], parsed["competition"])
    else:
        base.extend(["--submit-competition", parsed["competition"]])

    if "--submit-via" not in base:
        base.extend(["--submit-via", "cli"])

    note = (
        "[cli] detected an embedded 'kaggle competitions submit ...' tail after "
        "'pipeline_cli.py run'. Bash line continuation joined both commands, so "
        "the CLI rewrote it to the built-in form: --submit --submit-via cli "
        f"--submit-competition {parsed['competition']} --message {parsed['message']!r}."
    )
    return base, note


def _format_unknown_args_error(unknown: Sequence[str]) -> str:
    msg = "unrecognized arguments: " + " ".join(unknown)
    if len(unknown) >= 3 and list(unknown[:3]) == ["kaggle", "competitions", "submit"]:
        msg += (
            "\nHint: you appended a raw 'kaggle competitions submit ...' command to "
            "'pipeline_cli.py run'. Use either two separate shell commands joined by '&&', "
            "or use the built-in submit flags: --submit --message '...'."
        )
    return msg

def _probe_g4f_models_sync(
    candidates: Sequence[str],
    *,
    timeout: float,
    prompt: str,
    system_prompt: str,
    provider_name: Optional[str] = None,
    on_result: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    total = len(candidates)
    results: List[Dict[str, Any]] = []
    for idx, model in enumerate(candidates, start=1):
        ok, info, elapsed = _probe_g4f_model_pipeline(
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
        results.append(result)
        if on_result is not None:
            on_result(idx, total, result)
    return results


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

    probe_mode = str(getattr(args, "probe_mode", "pipeline") or "pipeline").strip().lower()
    if probe_mode not in {"pipeline", "async"}:
        raise SystemExit(f"Unsupported --probe-mode: {probe_mode}")

    if not args.list_only:
        mode_desc = "pipeline-compatible probe" if probe_mode == "pipeline" else f"AsyncClient concurrency={args.concurrency}"
        print(
            f"[g4f-check] checking {len(candidates)} model(s) with prompt={args.prompt!r} "
            f"using {mode_desc}..."
        )

    def _on_result(idx: int, total: int, result: Dict[str, Any]) -> None:
        if args.list_only:
            return
        status = "OK" if result.get("ok") else "FAIL"
        print(f"[{idx}/{total}] {result['model']}: {status} ({result['elapsed_s']:.2f}s) {result['detail']}")

    if probe_mode == "async":
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
    else:
        results = _probe_g4f_models_sync(
            candidates,
            timeout=float(args.timeout),
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            provider_name=provider_name or None,
            on_result=_on_result,
        )
    working: List[str] = [str(r["model"]) for r in results if r.get("ok")]
    if "r1-1776" in working:
        working = ["r1-1776"] + [name for name in working if name != "r1-1776"]

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

    Important: extract into a cache path rather than overwriting the checked-in
    repository fixture. This keeps schema checks aligned with the actual
    competition bundle without mutating tracked files during selftests or CLI
    runs.
    """
    zip_path = _resolve_competition_zip(spec)
    if zip_path is None:
        return None
    target = ROOT / "_cache" / "competition_bundle_schema" / spec.key / "sample_submission.csv"
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


def _variant_prompt_path(base: Optional[Path], variant: Optional[str], *, role: str) -> Optional[Path]:
    if base is None or not variant:
        return base
    parent = base.parent
    suffix = base.suffix
    stem = base.stem

    if role == "prompt":
        candidates = [
            parent / f"{stem}_{variant}{suffix}",
            parent / f"user_prompt_{variant}{suffix}",
        ]
    else:
        candidates = [
            parent / f"custom_prompts_{variant}{suffix}",
            parent / f"{stem}_{variant}{suffix}",
        ]

    for cand in candidates:
        if cand.exists():
            return cand
    raise SystemExit(
        f"Requested --prompt-variant={variant!r}, but no matching {role} file was found near {base}. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def _resolve_prompt_bundle(spec: PipelineSpec, args: argparse.Namespace) -> tuple[Path, Optional[Path]]:
    variant = getattr(args, "prompt_variant", None)
    prompt_file = Path(args.prompt_file) if getattr(args, "prompt_file", None) else _variant_prompt_path(spec.prompt_file, variant, role="prompt")
    if prompt_file is None:
        raise SystemExit(f"No prompt_file configured for {spec.key}; pass --prompt-file.")
    custom_prompts = Path(args.custom_prompts) if getattr(args, "custom_prompts", None) else _variant_prompt_path(spec.custom_prompts_file, variant, role="custom prompts")
    return prompt_file, custom_prompts


def _safe_read_text(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return ""


def _prompt_bundle_requests_from_scratch(prompt_file: Path, custom_prompts: Optional[Path] = None) -> bool:
    text = ("\n".join([_safe_read_text(prompt_file), _safe_read_text(custom_prompts)])).lower()
    strong_markers = [
        'no_baseline_patch_bias',
        'no reference baseline will be shown',
        'do not expect any baseline section',
        'do not rely on, patch, wrap, or extend any baseline',
        'do not rely on, patch, wrap, or extend any baseline implementation',
        'write the code from scratch',
        'fresh solver from scratch',
    ]
    if any(marker in text for marker in strong_markers):
        return True

    stems = [prompt_file.stem.lower()]
    if custom_prompts is not None:
        stems.append(custom_prompts.stem.lower())
    if any(stem.endswith('_regular') or stem == 'regular' for stem in stems):
        return True

    return False


def _prompt_bundle_uses_baseline(prompt_file: Path, custom_prompts: Optional[Path] = None) -> bool:
    if _prompt_bundle_requests_from_scratch(prompt_file, custom_prompts):
        return False
    text = ("\n".join([_safe_read_text(prompt_file), _safe_read_text(custom_prompts)])).lower()
    return 'baseline' in text


def _prompt_bundle_supports_ranked_reuse(prompt_file: Path, custom_prompts: Optional[Path] = None) -> bool:
    return _prompt_bundle_uses_baseline(prompt_file, custom_prompts) or _prompt_bundle_requests_from_scratch(prompt_file, custom_prompts)


def _prompt_bundle_requires_json_code_envelope(prompt_file: Path, custom_prompts: Optional[Path] = None) -> bool:
    return code_contract.prompt_requests_code_json_envelope(
        _safe_read_text(prompt_file),
        _safe_read_text(custom_prompts),
    )


def _adaptive_baseline_bundle_slug(prompt_file: Path, custom_prompts: Optional[Path] = None) -> str:
    pieces = [prompt_file.stem]
    if custom_prompts is not None:
        pieces.append(custom_prompts.stem)
    raw = '__'.join(pieces).strip().lower()
    slug = re.sub(r'[^a-z0-9._-]+', '-', raw).strip('-._')
    return slug or 'default'


def _adaptive_baseline_paths(spec: PipelineSpec, prompt_file: Path, custom_prompts: Optional[Path] = None) -> dict[str, Path]:
    slug = _adaptive_baseline_bundle_slug(prompt_file, custom_prompts)
    root = ROOT / 'competitions' / spec.key / 'generated' / 'adaptive_baselines'
    return {
        'root': root,
        'solver': root / f'{slug}.py',
        'meta': root / f'{slug}.json',
    }


def _read_json_file(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _load_competition_self_improver(spec: PipelineSpec):
    module_path = ROOT / 'competitions' / spec.key / 'prompt_self_improver.py'
    if not module_path.exists():
        return None
    module_name = f"competition_prompt_self_improver_{re.sub(r'[^a-zA-Z0-9_]+', '_', spec.key)}"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec_obj = importlib.util.spec_from_file_location(module_name, module_path)
    if spec_obj is None or spec_obj.loader is None:
        return None
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    spec_obj.loader.exec_module(module)
    return module


def _prepare_competition_self_improvement_prompt_bundle(
    *,
    spec: PipelineSpec,
    base_prompt_file: Path,
    base_custom_prompts: Optional[Path],
    baseline_solver: Path,
    round_idx: int,
    score_history: Sequence[Dict[str, Any]],
    best_metric: Optional[Dict[str, Any]],
    prompt_history: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Optional[dict[str, Any]]:
    module = _load_competition_self_improver(spec)
    if module is None or not hasattr(module, 'build_round_prompt_bundle'):
        return None
    return module.build_round_prompt_bundle(
        base_prompt_file=base_prompt_file,
        base_custom_prompts=base_custom_prompts,
        baseline_solver=baseline_solver,
        round_idx=round_idx,
        score_history=score_history,
        best_metric=best_metric,
        prompt_history=prompt_history,
        output_dir=output_dir,
    )


def _write_prompt_evolution_history(out_path: Path, history: Sequence[Dict[str, Any]]) -> None:
    if not history:
        return
    try:
        target = out_path.with_name(out_path.stem + '_prompt_evolution.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(list(history), ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        return


def _parse_score_value(raw: Any) -> Optional[float]:
    if raw in (None, '', 'None'):
        return None
    try:
        return float(str(raw).strip())
    except Exception:
        return None


def _extract_kaggle_score_info(round_result: Optional[dict[str, Any]]) -> dict[str, Any]:
    info: dict[str, Any] = {
        'submission_id': None,
        'status': None,
        'public_score': None,
        'private_score': None,
        'leaderboard_score': None,
        'leaderboard_score_source': None,
        'error_description': None,
    }
    if not isinstance(round_result, dict):
        return info
    submit_report = round_result.get('kaggle_submit') if isinstance(round_result.get('kaggle_submit'), dict) else None
    status = submit_report.get('status') if isinstance(submit_report, dict) and isinstance(submit_report.get('status'), dict) else None
    if not isinstance(status, dict):
        return info

    public_score = _parse_score_value(status.get('public_score', status.get('publicScore')))
    private_score = _parse_score_value(status.get('private_score', status.get('privateScore')))
    leaderboard_score = public_score if public_score is not None else private_score
    leaderboard_source = 'public_score' if public_score is not None else ('private_score' if private_score is not None else None)

    info.update({
        'submission_id': status.get('id') or status.get('ref'),
        'status': status.get('status') or status.get('state'),
        'public_score': public_score,
        'private_score': private_score,
        'leaderboard_score': leaderboard_score,
        'leaderboard_score_source': leaderboard_source,
        'error_description': status.get('error_description') or status.get('errorDescription'),
    })
    return info


def _adaptive_metric_from_round(*, local_score: Optional[int], round_result: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    kaggle_info = _extract_kaggle_score_info(round_result)
    leaderboard_score = kaggle_info.get('leaderboard_score')
    if leaderboard_score is not None:
        return {
            'source': f"kaggle_{kaggle_info.get('leaderboard_score_source') or 'score'}",
            'value': float(leaderboard_score),
            'submission_id': kaggle_info.get('submission_id'),
            'status': kaggle_info.get('status'),
            'public_score': kaggle_info.get('public_score'),
            'private_score': kaggle_info.get('private_score'),
        }
    if local_score is not None:
        return {
            'source': 'local_score',
            'value': float(local_score),
        }
    return None


_ADAPTIVE_METRIC_PRIORITY = {
    'kaggle_public_score': 3,
    'kaggle_private_score': 2,
    'kaggle_score': 2,
    'local_score': 1,
}


def _is_better_metric(current_metric: Optional[dict[str, Any]], candidate_metric: Optional[dict[str, Any]]) -> bool:
    if not candidate_metric:
        return False
    if not isinstance(current_metric, dict) or not current_metric:
        return True

    current_priority = int(_ADAPTIVE_METRIC_PRIORITY.get(str(current_metric.get('source') or ''), 0))
    candidate_priority = int(_ADAPTIVE_METRIC_PRIORITY.get(str(candidate_metric.get('source') or ''), 0))
    if candidate_priority != current_priority:
        return candidate_priority > current_priority

    current_value = _parse_score_value(current_metric.get('value'))
    candidate_value = _parse_score_value(candidate_metric.get('value'))
    if current_value is None:
        return candidate_value is not None
    if candidate_value is None:
        return False
    return candidate_value < current_value


def _should_promote_adaptive_baseline(current_meta: Optional[dict[str, Any]], candidate_metric: Optional[dict[str, Any]]) -> bool:
    if not candidate_metric:
        return False
    if not isinstance(current_meta, dict):
        return True

    current_metric = current_meta.get('selection_metric') if isinstance(current_meta.get('selection_metric'), dict) else None
    return _is_better_metric(current_metric, candidate_metric)


def _resolve_effective_baseline(spec: PipelineSpec, prompt_file: Path, custom_prompts: Optional[Path] = None) -> tuple[Path, dict[str, Any]]:
    uses_baseline = _prompt_bundle_uses_baseline(prompt_file, custom_prompts)
    from_scratch = _prompt_bundle_requests_from_scratch(prompt_file, custom_prompts)
    supports_ranked_reuse = _prompt_bundle_supports_ranked_reuse(prompt_file, custom_prompts)
    paths = _adaptive_baseline_paths(spec, prompt_file, custom_prompts)
    meta = _read_json_file(paths['meta'])
    adaptive_solver = paths['solver']
    effective_baseline = spec.baseline_solver
    adaptive_valid: Optional[bool] = None
    adaptive_invalid_reason: Optional[str] = None

    if supports_ranked_reuse and adaptive_solver.exists():
        adaptive_valid, adaptive_invalid_reason = _probe_solver_validation_failure(
            adaptive_solver,
            spec.validator,
            _resolve_smoke_vectors(spec),
        )
        if adaptive_valid:
            effective_baseline = adaptive_solver

    mode = 'baseline_patch' if uses_baseline else ('reference_only' if from_scratch else 'disabled')
    info = {
        'enabled': supports_ranked_reuse,
        'from_scratch': from_scratch,
        'uses_baseline': uses_baseline,
        'mode': mode,
        'bundle_slug': _adaptive_baseline_bundle_slug(prompt_file, custom_prompts),
        'effective_baseline': str(effective_baseline),
        'default_baseline': str(spec.baseline_solver),
        'adaptive_solver': str(adaptive_solver),
        'adaptive_meta': str(paths['meta']),
        'adaptive_exists': adaptive_solver.exists(),
        'adaptive_valid': adaptive_valid,
        'adaptive_invalid_reason': adaptive_invalid_reason,
        'selection_metric': meta.get('selection_metric') if isinstance(meta, dict) else None,
    }
    return effective_baseline, info


def _persist_adaptive_baseline(
    *,
    spec: PipelineSpec,
    prompt_file: Path,
    custom_prompts: Optional[Path],
    solver_path: Path,
    round_idx: int,
    local_score: Optional[int],
    round_result: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    metric = _adaptive_metric_from_round(local_score=local_score, round_result=round_result)
    if metric is None:
        return None

    paths = _adaptive_baseline_paths(spec, prompt_file, custom_prompts)
    current_meta = _read_json_file(paths['meta'])
    if not _should_promote_adaptive_baseline(current_meta, metric):
        return None

    paths['root'].mkdir(parents=True, exist_ok=True)
    shutil.copyfile(solver_path, paths['solver'])
    kaggle_info = _extract_kaggle_score_info(round_result)
    manifest = {
        'competition': spec.key,
        'competition_slug': spec.competition,
        'prompt_bundle': _adaptive_baseline_bundle_slug(prompt_file, custom_prompts),
        'updated_at': datetime.now().isoformat(),
        'solver_path': str(paths['solver']),
        'source_solver_path': str(solver_path),
        'round': round_idx,
        'local_score': local_score,
        'selection_metric': metric,
        'kaggle': kaggle_info,
        'prompt_file': str(prompt_file),
        'custom_prompts': str(custom_prompts) if custom_prompts is not None else None,
        'default_baseline': str(spec.baseline_solver),
    }
    paths['meta'].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


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
                    raise ValueError(
                        f"Row {row_idx}: UNSOLVED is not a legal Kaggle move sequence. "
                        "Use a valid dot-separated path or fall back to a known-good baseline."
                    )
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


def _append_label_before_suffixes(path: Path, label: str) -> Path:
    suffixes = ''.join(path.suffixes)
    if suffixes:
        base_name = path.name[:-len(suffixes)]
        return path.with_name(f'{base_name}{label}{suffixes}')
    return path.with_name(path.name + label)


def _candidate_output_path(out_csv: Path) -> Path:
    return _append_label_before_suffixes(out_csv, '.candidate')


def _round_submission_output_path(out_csv: Path, round_idx: int) -> Path:
    return _append_label_before_suffixes(out_csv, f'.round{max(1, int(round_idx))}')


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
    detail = str(exc).strip()
    low = detail.lower()
    if detail and (
        'preflight failed' in low
        or 'too old for reliable competition submission' in low
        or 'kaggle cli is not installed' in low
        or 'cannot access competition submissions yet' in low
    ):
        return detail

    status_code = getattr(getattr(exc, 'response', None), 'status_code', None)
    if status_code == 401:
        return (
            f"Kaggle returned 401 Unauthorized while submitting to '{competition}'. "
            "Most often this means the runtime has no valid Kaggle credentials, the token is stale, "
            "or the competition rules have not been accepted for this account. "
            "Pass --kaggle-json /path/to/kaggle.json (or ~/.kaggle/access_token), regenerate the token if needed, "
            "and make sure the account has joined the competition."
        )
    return f"Kaggle submission failed: {detail or exc}"


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


def _print_kaggle_preflight_report(report: dict[str, Any]) -> None:
    mode = str(report.get('mode') or '?')
    version = str(report.get('client_version') or 'unknown')
    access = report.get('access') if isinstance(report.get('access'), dict) else {}
    bits = [f"client_version={version}"]
    if isinstance(access, dict):
        for key in ('can_list_files', 'can_list_submissions', 'rules_accepted_or_joined', 'file_count', 'submission_count'):
            if key in access and access.get(key) is not None:
                bits.append(f"{key}={access.get(key)}")
    print(f"[kaggle] preflight ({mode}): " + ' '.join(bits), flush=True)



def _has_kaggle_env_credentials(env: Optional[dict[str, str]] = None) -> bool:
    source = env or os.environ
    if source.get('KAGGLE_API_TOKEN') or source.get('KAGGLE_TOKEN'):
        return True
    return bool(source.get('KAGGLE_USERNAME') and source.get('KAGGLE_KEY'))



def _autodiscover_kaggle_credentials_file(*, explicit_path: str | None = None) -> Optional[dict[str, str]]:
    """Best-effort recovery for notebook/Colab uploads with arbitrary filenames.

    Common failure mode: the user uploads `kaggle_something.json`, but the notebook
    later hardcodes `~/.kaggle/kaggle.json` or `--kaggle-json ~/.kaggle/kaggle.json`.
    We only auto-recover when we can identify a *single* valid credentials file.
    """
    try:
        _ensure_llm_puzzles_on_path()
        from src.kaggle_utils import _load_kaggle_credentials  # type: ignore
    except Exception:
        return None

    resolved_explicit = str(Path(explicit_path).expanduser().resolve()) if explicit_path else None
    search_roots: list[Path] = []
    for root in [Path.cwd(), ROOT, Path('/content')]:
        try:
            rr = root.expanduser().resolve()
        except Exception:
            continue
        if rr.exists() and rr not in search_roots:
            search_roots.append(rr)

    patterns = [
        'kaggle.json',
        'kaggle*.json',
        '*kaggle*.json',
        'access_token',
        '*access_token*',
        '*.json',
        '*.token',
        '*.txt',
    ]

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in search_roots:
        for pattern in patterns:
            try:
                paths = sorted(root.glob(pattern))
            except Exception:
                continue
            for path in paths:
                try:
                    resolved = str(path.resolve())
                except Exception:
                    continue
                if resolved in seen or resolved == resolved_explicit:
                    continue
                seen.add(resolved)
                if not path.is_file():
                    continue
                try:
                    stat = path.stat()
                except Exception:
                    continue
                if stat.st_size <= 0 or stat.st_size > 128 * 1024:
                    continue
                try:
                    creds = _load_kaggle_credentials(resolved)
                except Exception:
                    continue
                name = path.name.lower()
                score = 0
                if root == Path.cwd().resolve():
                    score += 20
                if name == 'kaggle.json':
                    score += 10
                if 'kaggle' in name:
                    score += 5
                candidates.append({
                    'path': resolved,
                    'score': score,
                    'kind': str(creds.get('kind') or ''),
                    'name': path.name,
                })

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-int(item['score']), str(item['name']), str(item['path'])))
    top_score = int(candidates[0]['score'])
    top = [c for c in candidates if int(c['score']) == top_score]
    if len(top) != 1:
        return None

    winner = top[0]
    return {
        'credentials_path': str(winner['path']),
        'kind': str(winner['kind']),
        'source': 'autodiscovered_file',
    }



def _resolve_kaggle_submit_availability(
    *,
    kaggle_json: str | None,
    kaggle_config_dir: str | None,
) -> dict[str, Any]:
    explicit_path = str(Path(kaggle_json).expanduser().resolve()) if kaggle_json else None
    if kaggle_json:
        explicit_exists = Path(kaggle_json).expanduser().exists()
        if explicit_exists:
            return {
                'enabled': True,
                'source': 'explicit_file',
                'credentials_path': explicit_path,
            }
        recovered = _autodiscover_kaggle_credentials_file(explicit_path=explicit_path)
        if recovered is not None:
            return {
                'enabled': True,
                'source': str(recovered.get('source') or 'autodiscovered_file'),
                'credentials_path': str(recovered['credentials_path']),
                'recovered_from_missing_explicit': explicit_path,
            }
        return {
            'enabled': False,
            'source': 'explicit_file_missing',
            'credentials_path': explicit_path,
            'reason': f'Kaggle credentials file was not found: {explicit_path}',
            'nonfatal': False,
        }

    if _has_kaggle_env_credentials():
        return {
            'enabled': True,
            'source': 'environment',
            'credentials_path': None,
        }

    try:
        _ensure_llm_puzzles_on_path()
        from src.kaggle_utils import _discover_default_credentials_path  # type: ignore

        discovered = _discover_default_credentials_path(kaggle_config_dir)
    except Exception:
        discovered = None

    if discovered:
        return {
            'enabled': True,
            'source': 'default_file',
            'credentials_path': str(Path(discovered).expanduser().resolve()),
        }

    recovered = _autodiscover_kaggle_credentials_file(explicit_path=None)
    if recovered is not None:
        return {
            'enabled': True,
            'source': str(recovered.get('source') or 'autodiscovered_file'),
            'credentials_path': str(recovered['credentials_path']),
        }

    return {
        'enabled': False,
        'source': 'missing_credentials',
        'credentials_path': None,
        'reason': (
            'No Kaggle credentials were found. Live submission will be skipped. '
            'Pass --kaggle-json /path/to/kaggle.json (or ~/.kaggle/access_token), '
            'set KAGGLE_USERNAME/KAGGLE_KEY or KAGGLE_API_TOKEN, or place credentials under ~/.kaggle/.'
        ),
        'nonfatal': True,
    }



def _nonfatal_kaggle_submit_report(exc: BaseException, competition: str) -> dict[str, Any]:
    return {
        'mode': 'error',
        'submitted': False,
        'competition': competition,
        'error_type': type(exc).__name__,
        'error': _format_kaggle_submit_error(exc, competition),
        'nonfatal': True,
    }



def _kaggle_submit(
    *,
    competition: str,
    submission_csv: Path,
    message: str,
    kaggle_json: str | None = None,
    submit_via: str = 'auto',
    kaggle_config_dir: str | None = None,
) -> dict[str, Any]:
    """Submit to Kaggle using either the Python API or the CLI.

    submit_via: 'auto' | 'api' | 'cli'
    """
    if submit_via not in {'auto', 'api', 'cli'}:
        raise ValueError(f"Unknown submit_via={submit_via}")

    if submit_via in {'auto', 'api'}:
        try:
            _ensure_llm_puzzles_on_path()
            from src.kaggle_utils import ensure_auth, preflight_submit_via_api, submit_file

            api_preflight = preflight_submit_via_api(
                competition,
                kaggle_json_path=kaggle_json,
                config_dir=kaggle_config_dir,
            )
            _print_kaggle_preflight_report(api_preflight)

            api = ensure_auth(kaggle_json_path=kaggle_json, config_dir=kaggle_config_dir)
            print(f"[kaggle] submitting via API: competition={competition} file={submission_csv}", flush=True)
            submit_file(api, competition=competition, filepath=str(submission_csv), message=message)
            status = _poll_kaggle_submission_status(competition, kaggle_json, kaggle_config_dir)
            return {
                'mode': 'api',
                'preflight': api_preflight,
                'status': status,
            }
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

    try:
        _ensure_llm_puzzles_on_path()
        from src.kaggle_utils import preflight_submit_via_cli

        cli_preflight = preflight_submit_via_cli(
            competition,
            credentials_path=kaggle_json,
            config_dir=kaggle_config_dir,
        )
        _print_kaggle_preflight_report(cli_preflight)
    except Exception as e:
        raise SystemExit(_format_kaggle_submit_error(e, competition))

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
            status = _poll_kaggle_submission_status(competition, kaggle_json, kaggle_config_dir)
            return {
                'mode': 'cli',
                'preflight': cli_preflight,
                'status': status,
            }
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


def _solver_validation_timeout_s() -> float:
    raw = os.getenv("AGENTLAB_VALIDATOR_TIMEOUT_S", os.getenv("PIPELINE_SOLVER_TIMEOUT_S", "20"))
    try:
        return max(1.0, float(raw))
    except Exception:
        return 20.0


def _resolve_smoke_vectors(spec: PipelineSpec, extra_rows: int = 2) -> list[list[int]]:
    vectors: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def _add(vec: Sequence[int] | None) -> None:
        if vec is None:
            return
        norm = [int(x) for x in vec]
        sig = tuple(norm)
        if sig in seen:
            return
        seen.add(sig)
        vectors.append(norm)

    _add(spec.smoke_vector)

    if extra_rows <= 0:
        return vectors

    try:
        puzzles_csv = _resolve_default_puzzles(spec)
        _ensure_csv_field_size_limit()
        with puzzles_csv.open(newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    vec = _extract_state(row, spec)
                except Exception:
                    continue
                if not vec:
                    continue
                _add(vec)
                if len(vectors) >= 1 + extra_rows:
                    break
    except Exception:
        pass

    return vectors


def _normalize_smoke_vectors(smoke_vector: Sequence[int] | Sequence[Sequence[int]]) -> list[list[int]]:
    if not smoke_vector:
        return []
    first = smoke_vector[0]  # type: ignore[index]
    if isinstance(first, (list, tuple)):
        return [[int(x) for x in vec] for vec in smoke_vector]  # type: ignore[arg-type]
    return [[int(x) for x in smoke_vector]]  # type: ignore[arg-type]


def _probe_solver_validation_failure(
    solver_path: Path,
    validator_path: Path,
    smoke_vector: Sequence[int] | Sequence[Sequence[int]],
) -> tuple[bool, Optional[str]]:
    if not solver_path.exists():
        return False, f'missing solver file: {solver_path}'
    try:
        py_compile.compile(str(solver_path), doraise=True)
    except Exception as exc:
        return False, f'py_compile failed: {type(exc).__name__}: {exc}'

    try:
        solver_text = solver_path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        solver_text = ''
    if 'def solve(' not in solver_text and 'async def solve(' not in solver_text:
        return False, 'py_compile failed: missing solve() entrypoint'

    smoke_vectors = _normalize_smoke_vectors(smoke_vector)
    if not smoke_vectors:
        return True, None

    timeout_s = _solver_validation_timeout_s()
    for one_vec in smoke_vectors:
        cmd = [
            PYTHON,
            str(validator_path),
            '--solver',
            str(solver_path),
            '--vector',
            json.dumps(list(one_vec)),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            return False, f'validator timeout after {timeout_s:g}s: {type(exc).__name__}'
        if proc.returncode != 0:
            details = '\n'.join(part.strip() for part in [proc.stdout or '', proc.stderr or ''] if part.strip())
            details = details[:1200]
            return False, details or f'validator exited with status {proc.returncode}'
    return True, None


def _validate_solver(solver_path: Path, validator_path: Path, smoke_vector: Sequence[int] | Sequence[Sequence[int]]) -> None:
    smoke_vectors = _normalize_smoke_vectors(smoke_vector)
    total = len(smoke_vectors)
    if total == 0:
        raise ValueError("at least one smoke vector is required")

    timeout_s = _solver_validation_timeout_s()
    for idx, one_vec in enumerate(smoke_vectors, start=1):
        label = validator_path.name if total == 1 else f"{validator_path.name} [{idx}/{total}]"
        print(f"[validate] {label} ...")
        cmd = [
            PYTHON,
            str(validator_path),
            "--solver",
            str(solver_path),
            "--vector",
            json.dumps(list(one_vec)),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout or ""
            err = exc.stderr or ""
            if out:
                print(out, end="" if out.endswith("\n") else "\n")
            if err:
                print(err, end="" if err.endswith("\n") else "\n", file=sys.stderr)
            raise RuntimeError(
                f"Validator timed out after {timeout_s:g}s for {solver_path}. "
                "This usually means solve(vec) got stuck during smoke validation."
            ) from exc
        if proc.returncode != 0:
            if proc.stdout:
                print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
            if proc.stderr:
                print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr)
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)


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


def _poll_kaggle_submission_status(competition: str, kaggle_json: str | None, kaggle_config_dir: str | None, wait_seconds: int = 45) -> Optional[dict[str, Any]]:
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
            return None

        sid = sub.get('id') or sub.get('ref') or '?'
        status = sub.get('status') or sub.get('state') or 'unknown'
        print(f"[kaggle] latest submission right after upload: id={sid} status={status}", flush=True)

        final_sub = wait_for_submission_result(api, competition, target_ref=sid, wait_seconds=wait_seconds)
        if not final_sub:
            print('[kaggle] submission accepted by upload step; scoring is still pending.', flush=True)
            return sub

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
        return final_sub
    except Exception as e:
        print(f"[kaggle] WARNING: could not poll submission status: {e}", flush=True)
        return None


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
        env.setdefault("AGENTLAB_G4F_USE_ASYNC", os.getenv("AGENTLAB_G4F_USE_ASYNC", "1"))
        env.setdefault("AGENTLAB_G4F_REQUEST_TIMEOUT_S", os.getenv("AGENTLAB_G4F_REQUEST_TIMEOUT_S", "180"))
        env.setdefault("AGENTLAB_MAX_RESPONSE_CHARS", os.getenv("AGENTLAB_MAX_RESPONSE_CHARS", "0"))
        env.setdefault("AGENTLAB_G4F_STOP_AT_PYTHON_FENCE", os.getenv("AGENTLAB_G4F_STOP_AT_PYTHON_FENCE", "0"))
        env.setdefault("AGENTLAB_ARTIFACT_SPILL_CHARS", os.getenv("AGENTLAB_ARTIFACT_SPILL_CHARS", "8000"))
        env.setdefault("AGENTLAB_HEAVY_IMPORTS", os.getenv("AGENTLAB_HEAVY_IMPORTS", "0"))
        env.setdefault("MALLOC_ARENA_MAX", os.getenv("MALLOC_ARENA_MAX", "2"))
    return env


def _agent_model_cli_args(
    *,
    agent_models: str | None = None,
    planner_models: str | None = None,
    coder_models: str | None = None,
    fixer_models: str | None = None,
    search_mode: str | None = None,
    plan_beam_width: int | None = None,
    frontier_width: int | None = None,
    archive_size: int | None = None,
    refine_rounds: int | None = None,
) -> list[str]:
    args: list[str] = []
    if agent_models:
        args.extend(["--agent-models", agent_models])
    if planner_models:
        args.extend(["--planner-models", planner_models])
    if coder_models:
        args.extend(["--coder-models", coder_models])
    if fixer_models:
        args.extend(["--fixer-models", fixer_models])
    if search_mode:
        args.extend(["--search-mode", search_mode])
    if plan_beam_width is not None:
        args.extend(["--plan-beam-width", str(max(1, int(plan_beam_width)))])
    if frontier_width is not None:
        args.extend(["--frontier-width", str(max(1, int(frontier_width)))])
    if archive_size is not None:
        args.extend(["--archive-size", str(max(1, int(archive_size)))])
    if refine_rounds is not None:
        args.extend(["--refine-rounds", str(max(0, int(refine_rounds)))])
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
    search_mode: str | None = None,
    plan_beam_width: int | None = None,
    frontier_width: int | None = None,
    archive_size: int | None = None,
    refine_rounds: int | None = None,
    max_iters: int = 8,
    no_llm: bool = False,
    allow_baseline: bool = True,
    g4f_recovery_rounds: int | None = None,
    baseline_patch_max_iters: int | None = None,
    g4f_recovery_max_iters: int | None = None,
    g4f_recovery_sleep: float | None = None,
    worker_no_kill_process_group: bool = False,
    print_generation: bool = False,
    print_generation_max_chars: int | None = None,
    g4f_async: Optional[bool] = None,
    max_response_chars: int | None = None,
    g4f_request_timeout: float | None = None,
    g4f_stop_at_python_fence: Optional[bool] = None,
    strict_codegen: bool = False,
    attempt_archive_dir: Optional[Path] = None,
) -> None:
    """Run AgentLaboratory perm_pipeline to generate/repair a solver.

    strict_codegen is intentionally separate from allow_baseline: during
    self-improvement a provider/backend failure must be treated as a failed
    candidate, not as a valid baseline-sized improvement attempt.
    """

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
            search_mode=search_mode,
            plan_beam_width=plan_beam_width,
            frontier_width=frontier_width,
            archive_size=archive_size,
            refine_rounds=refine_rounds,
        )
    )

    # Basic self-improvement scenario: fail loud on codegen/provider failure.
    # Without --strict, run_perm_pipeline.py writes the baseline solver and exits 0,
    # which makes the outer loop spend many rounds scoring identical candidates.
    if strict_codegen:
        cmd.append("--strict")
    if no_llm:
        cmd.append("--no-llm")
    if custom_prompts:
        cmd.extend(["--custom-prompts", str(custom_prompts)])
    if attempt_archive_dir is not None:
        attempt_archive_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--attempt-archive-dir", str(attempt_archive_dir)])

    env = os.environ.copy()
    effective_codegen_env = _memory_env_for_codegen(llm)
    env.update(effective_codegen_env)
    if g4f_recovery_rounds is not None:
        env["AGENTLAB_G4F_RECOVERY_ROUNDS"] = str(max(0, int(g4f_recovery_rounds)))
        effective_codegen_env["AGENTLAB_G4F_RECOVERY_ROUNDS"] = env["AGENTLAB_G4F_RECOVERY_ROUNDS"]
    if baseline_patch_max_iters is not None:
        env["AGENTLAB_BASELINE_PATCH_MAX_ITERS"] = str(max(1, int(baseline_patch_max_iters)))
        effective_codegen_env["AGENTLAB_BASELINE_PATCH_MAX_ITERS"] = env["AGENTLAB_BASELINE_PATCH_MAX_ITERS"]
    if g4f_recovery_max_iters is not None:
        env["AGENTLAB_G4F_RECOVERY_MAX_ITERS"] = str(max(1, int(g4f_recovery_max_iters)))
        effective_codegen_env["AGENTLAB_G4F_RECOVERY_MAX_ITERS"] = env["AGENTLAB_G4F_RECOVERY_MAX_ITERS"]
    if g4f_recovery_sleep is not None:
        env["AGENTLAB_G4F_RECOVERY_SLEEP_S"] = str(max(0.0, float(g4f_recovery_sleep)))
        effective_codegen_env["AGENTLAB_G4F_RECOVERY_SLEEP_S"] = env["AGENTLAB_G4F_RECOVERY_SLEEP_S"]
    if worker_no_kill_process_group:
        env["AGENTLAB_WORKER_KILL_PROCESS_GROUP"] = "0"
        effective_codegen_env["AGENTLAB_WORKER_KILL_PROCESS_GROUP"] = "0"
    if print_generation:
        env["AGENTLAB_PRINT_GENERATION"] = "1"
        effective_codegen_env["AGENTLAB_PRINT_GENERATION"] = "1"
    if print_generation_max_chars is not None:
        env["AGENTLAB_PRINT_GENERATION_MAX_CHARS"] = str(max(0, int(print_generation_max_chars)))
        effective_codegen_env["AGENTLAB_PRINT_GENERATION_MAX_CHARS"] = env["AGENTLAB_PRINT_GENERATION_MAX_CHARS"]
    if g4f_async is not None:
        env["AGENTLAB_G4F_USE_ASYNC"] = "1" if g4f_async else "0"
        effective_codegen_env["AGENTLAB_G4F_USE_ASYNC"] = env["AGENTLAB_G4F_USE_ASYNC"]
    if max_response_chars is not None:
        env["AGENTLAB_MAX_RESPONSE_CHARS"] = str(int(max_response_chars))
        effective_codegen_env["AGENTLAB_MAX_RESPONSE_CHARS"] = env["AGENTLAB_MAX_RESPONSE_CHARS"]
    if g4f_request_timeout is not None:
        env["AGENTLAB_G4F_REQUEST_TIMEOUT_S"] = str(max(0.0, float(g4f_request_timeout)))
        effective_codegen_env["AGENTLAB_G4F_REQUEST_TIMEOUT_S"] = env["AGENTLAB_G4F_REQUEST_TIMEOUT_S"]
    if g4f_stop_at_python_fence is not None:
        env["AGENTLAB_G4F_STOP_AT_PYTHON_FENCE"] = "1" if g4f_stop_at_python_fence else "0"
        effective_codegen_env["AGENTLAB_G4F_STOP_AT_PYTHON_FENCE"] = env["AGENTLAB_G4F_STOP_AT_PYTHON_FENCE"]
    print("[agentlab] " + " ".join(cmd))
    if effective_codegen_env:
        print("[agentlab] low-RAM env: " + ", ".join(f"{k}={effective_codegen_env[k]}" for k in sorted(effective_codegen_env.keys())))
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)



def _file_sha256(path: Optional[Path]) -> Optional[str]:
    """Return a stable full-file SHA256 digest for guard/lineage metadata."""
    if path is None or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _infer_submission_columns(submission_csv: Path) -> tuple[str, str]:
    """Infer (id_column, move_column) from a submission-like CSV."""
    _ensure_csv_field_size_limit()
    with submission_csv.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError(f'Could not read header from {submission_csv}')
    id_field = fieldnames[0]
    move_field: Optional[str] = None
    for candidate in ('path', 'moves', 'solution'):
        if candidate in fieldnames:
            move_field = candidate
            break
    if move_field is None:
        for name in fieldnames:
            if str(name).strip().lower() not in {'id', 'initial_state_id'}:
                move_field = name
                break
    if move_field is None:
        raise ValueError(f'Could not infer moves column in {submission_csv}')
    return id_field, move_field


def _split_move_path(value: Any) -> list[str]:
    """Split a Megaminx path using the actual dot-delimited format, with whitespace fallback."""
    text = str(value or '').strip()
    if not text or text.upper() == 'UNSOLVED':
        return []
    if '.' in text:
        return [part for part in text.split('.') if part]
    return [part for part in text.split() if part]


def _submission_path_digest(submission_csv: Optional[Path]) -> Optional[str]:
    """Hash normalized id/path rows so identical submissions cannot masquerade as new work."""
    if submission_csv is None or not submission_csv.exists():
        return None
    id_field, move_field = _infer_submission_columns(submission_csv)
    h = hashlib.sha256()
    with submission_csv.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get(id_field, '')).strip()
            path_value = str(row.get(move_field, '')).strip()
            h.update(rid.encode('utf-8'))
            h.update(b'\x1f')
            h.update(path_value.encode('utf-8'))
            h.update(b'\n')
    return h.hexdigest()


def _submission_guard_stats(submission_csv: Optional[Path]) -> dict[str, Any]:
    """Return row/move stats used by self-improvement acceptance and run reports."""
    out: dict[str, Any] = {
        'exists': bool(submission_csv and submission_csv.exists()),
        'row_count': 0,
        'empty_rows': 0,
        'unsolved_rows': 0,
        'total_move_tokens': 0,
        'mean_move_tokens': None,
        'max_move_tokens': 0,
        'digest': None,
    }
    if submission_csv is None or not submission_csv.exists():
        return out
    try:
        id_field, move_field = _infer_submission_columns(submission_csv)
        lengths: list[int] = []
        with submission_csv.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = str(row.get(move_field) or '').strip()
                out['row_count'] += 1
                if not value:
                    out['empty_rows'] += 1
                if value.upper() == 'UNSOLVED':
                    out['unsolved_rows'] += 1
                n = len(_split_move_path(value))
                lengths.append(n)
                out['total_move_tokens'] += n
        if lengths:
            out['mean_move_tokens'] = out['total_move_tokens'] / len(lengths)
            out['max_move_tokens'] = max(lengths)
        out['digest'] = _submission_path_digest(submission_csv)
        out['id_column'] = id_field
        out['move_column'] = move_field
    except Exception as exc:
        out['csv_error'] = f'{type(exc).__name__}: {exc}'
    return out


def _write_per_row_delta(
    *,
    baseline_csv: Optional[Path],
    candidate_csv: Optional[Path],
    out_csv: Path,
    out_json: Optional[Path] = None,
) -> dict[str, Any]:
    """Write per-row move-count deltas for candidate-vs-current-best acceptance debugging.

    Delta is candidate_len - baseline_len, so negative values are improvements.
    """
    summary: dict[str, Any] = {
        'baseline_csv': str(baseline_csv) if baseline_csv else None,
        'candidate_csv': str(candidate_csv) if candidate_csv else None,
        'delta_csv': str(out_csv),
        'rows_compared': 0,
        'improved_rows': 0,
        'regressed_rows': 0,
        'unchanged_rows': 0,
        'total_saved_moves': 0,
        'total_added_moves': 0,
        'net_delta_moves': 0,
        'ok': False,
    }
    if baseline_csv is None or candidate_csv is None or not baseline_csv.exists() or not candidate_csv.exists():
        summary['error'] = 'missing baseline or candidate CSV'
        return summary

    base_id, base_move = _infer_submission_columns(baseline_csv)
    cand_id, cand_move = _infer_submission_columns(candidate_csv)
    baseline: dict[str, tuple[int, str]] = {}
    candidate: dict[str, tuple[int, str]] = {}
    with baseline_csv.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rid = str(row.get(base_id, '')).strip()
            path_value = str(row.get(base_move, '')).strip()
            baseline[rid] = (len(_split_move_path(path_value)), path_value)
    with candidate_csv.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rid = str(row.get(cand_id, '')).strip()
            path_value = str(row.get(cand_move, '')).strip()
            candidate[rid] = (len(_split_move_path(path_value)), path_value)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'baseline_len', 'candidate_len', 'delta', 'status'])
        writer.writeheader()
        for rid in sorted(set(baseline) | set(candidate)):
            b = baseline.get(rid)
            c = candidate.get(rid)
            if b is None or c is None:
                status = 'missing_candidate' if c is None else 'missing_baseline'
                delta = None
                b_len = b[0] if b is not None else None
                c_len = c[0] if c is not None else None
            else:
                b_len, c_len = b[0], c[0]
                delta = c_len - b_len
                summary['rows_compared'] += 1
                summary['net_delta_moves'] += delta
                if delta < 0:
                    summary['improved_rows'] += 1
                    summary['total_saved_moves'] += -delta
                    status = 'improved'
                elif delta > 0:
                    summary['regressed_rows'] += 1
                    summary['total_added_moves'] += delta
                    status = 'regressed'
                else:
                    summary['unchanged_rows'] += 1
                    status = 'unchanged'
            writer.writerow({'id': rid, 'baseline_len': b_len, 'candidate_len': c_len, 'delta': delta, 'status': status})
    summary['ok'] = True
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary

def _submission_path_score(submission_csv: Path) -> int:
    stats = _submission_guard_stats(submission_csv)
    if stats.get('csv_error'):
        raise ValueError(f"Could not score submission {submission_csv}: {stats.get('csv_error')}")
    return int(stats.get('total_move_tokens') or 0)


def _score_solver_with_submission(
    *,
    spec: PipelineSpec,
    solver_path: Path,
    puzzles_csv: Path,
    competition_format_slug: str,
    vector_col_override: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> int:
    score, _ = _score_solver_with_submission_artifact(
        spec=spec,
        solver_path=solver_path,
        puzzles_csv=puzzles_csv,
        competition_format_slug=competition_format_slug,
        vector_col_override=vector_col_override,
        max_rows=max_rows,
    )
    return score


def _score_solver_with_submission_artifact(
    *,
    spec: PipelineSpec,
    solver_path: Path,
    puzzles_csv: Path,
    competition_format_slug: str,
    vector_col_override: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> tuple[int, Path]:
    temp_csv = solver_path.parent / f'.score_{solver_path.stem}.csv'
    _build_submission(
        puzzles_csv=puzzles_csv,
        out_csv=temp_csv,
        competition_format_slug=competition_format_slug,
        solver_path=solver_path,
        spec=spec,
        vector_col_override=vector_col_override,
        max_rows=max_rows,
        no_progress=True,
    )
    return _submission_path_score(temp_csv), temp_csv


def _score_solver_with_optional_artifact(
    *,
    spec: PipelineSpec,
    solver_path: Path,
    puzzles_csv: Path,
    competition_format_slug: str,
    vector_col_override: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> tuple[int, Optional[Path]]:
    """Score a solver and return the submission artifact when available.

    Normal production runs use the artifact-producing scorer. Some older unit tests
    monkeypatch only _score_solver_with_submission, so this compatibility wrapper
    falls back to that function when artifact generation is unavailable.
    """
    artifact_error: Optional[Exception] = None
    try:
        return _score_solver_with_submission_artifact(
            spec=spec,
            solver_path=solver_path,
            puzzles_csv=puzzles_csv,
            competition_format_slug=competition_format_slug,
            vector_col_override=vector_col_override,
            max_rows=max_rows,
        )
    except Exception as exc:
        artifact_error = exc
    try:
        score = _score_solver_with_submission(
            spec=spec,
            solver_path=solver_path,
            puzzles_csv=puzzles_csv,
            competition_format_slug=competition_format_slug,
            vector_col_override=vector_col_override,
            max_rows=max_rows,
        )
        return score, None
    except Exception:
        if artifact_error is not None:
            raise artifact_error
        raise


def _generate_solver_with_optional_improvement(
    *,
    spec: PipelineSpec,
    out_path: Path,
    prompt_file: Path,
    custom_prompts: Optional[Path],
    llm: str,
    agent_models: str | None,
    planner_models: str | None,
    coder_models: str | None,
    fixer_models: str | None,
    search_mode: str | None,
    plan_beam_width: int | None,
    frontier_width: int | None,
    archive_size: int | None,
    refine_rounds: int | None,
    max_iters: int,
    allow_baseline: bool,
    g4f_recovery_rounds: int | None,
    baseline_patch_max_iters: int | None = None,
    g4f_recovery_max_iters: int | None,
    g4f_recovery_sleep: float | None,
    worker_no_kill_process_group: bool,
    print_generation: bool,
    print_generation_max_chars: int | None,
    g4f_async: Optional[bool],
    max_response_chars: int | None,
    g4f_request_timeout: float | None,
    g4f_stop_at_python_fence: Optional[bool],
    keep_improving: bool,
    improvement_rounds: int,
    puzzles_csv_for_score: Optional[Path],
    competition_format_slug: str,
    vector_col_override: Optional[str] = None,
    max_rows: Optional[int] = None,
    validated_round_hook: Optional[Callable[[int, Path], Optional[Dict[str, Any]]]] = None,
    initial_baseline: Optional[Path] = None,
    candidate_round_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    self_improve_prompts: bool = False,
    reject_identical_candidates: bool = True,
    write_per_row_delta: bool = True,
) -> dict[str, Any]:
    smoke_vectors = _resolve_smoke_vectors(spec)
    scoring_enabled = bool(keep_improving and puzzles_csv_for_score is not None)
    rounds = max(1, int(improvement_rounds) if keep_improving else 1)
    baseline_origin = initial_baseline or spec.baseline_solver
    baseline_for_round = baseline_origin
    best_score: Optional[int] = None
    best_metric: Optional[dict[str, Any]] = None
    best_submission_csv: Optional[Path] = None
    best_solver_digest: Optional[str] = _file_sha256(baseline_origin)
    best_submission_digest: Optional[str] = None
    score_history: list[dict[str, Any]] = []
    generation_failure_streak = 0
    live_candidate_seen = False
    try:
        max_codegen_failures_before_live_candidate = int(os.getenv('SELF_IMPROVE_MAX_CODEGEN_FAILURES_BEFORE_LIVE', '3') or '3')
    except Exception:
        max_codegen_failures_before_live_candidate = 3

    if scoring_enabled:
        try:
            best_score, best_submission_csv = _score_solver_with_optional_artifact(
                spec=spec,
                solver_path=baseline_origin,
                puzzles_csv=puzzles_csv_for_score,
                competition_format_slug=competition_format_slug,
                vector_col_override=vector_col_override,
                max_rows=max_rows,
            )
            best_submission_digest = _submission_path_digest(best_submission_csv)
            best_metric = _adaptive_metric_from_round(local_score=best_score, round_result=None)
            print(f'[improve] baseline local score = {best_score}', flush=True)
        except Exception as e:
            scoring_enabled = False
            print(f'[improve] WARNING: could not score baseline locally; continuing without score-guided selection: {e}', flush=True)

    best_candidate_path: Optional[Path] = None
    best_round: Optional[int] = None
    submitted_rounds: list[int] = []
    prompt_evolution_history: list[dict[str, Any]] = []

    for round_idx in range(1, rounds + 1):
        candidate_path = out_path if (not keep_improving and round_idx == 1) else out_path.parent / f'{out_path.stem}.round{round_idx}{out_path.suffix}'
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'[improve] round {round_idx}/{rounds}: baseline={baseline_for_round} -> candidate={candidate_path}', flush=True)

        try:
            prompt_file_for_round = prompt_file
            custom_prompts_for_round = custom_prompts
            round_archive_dir = out_path.parent / f'{out_path.stem}_candidate_archive' / f'round_{round_idx:04d}'
            if self_improve_prompts:
                prompt_bundle_dir = out_path.parent / f'{out_path.stem}_prompt_rounds'
                prepared_prompt = _prepare_competition_self_improvement_prompt_bundle(
                    spec=spec,
                    base_prompt_file=prompt_file,
                    base_custom_prompts=custom_prompts,
                    baseline_solver=baseline_for_round,
                    round_idx=round_idx,
                    score_history=score_history,
                    best_metric=best_metric,
                    prompt_history=prompt_evolution_history,
                    output_dir=prompt_bundle_dir,
                )
                if prepared_prompt is not None:
                    prompt_file_for_round = Path(prepared_prompt['prompt_file'])
                    custom_prompts_for_round = Path(prepared_prompt['custom_prompts_file']) if prepared_prompt.get('custom_prompts_file') else None
                    meta = prepared_prompt.get('meta') if isinstance(prepared_prompt.get('meta'), dict) else None
                    if meta is not None:
                        prompt_evolution_history.append(meta)
            _run_agent_laboratory(
                prompt_file=prompt_file_for_round,
                out_path=candidate_path,
                validator=spec.validator,
                baseline=baseline_for_round,
                custom_prompts=custom_prompts_for_round,
                llm=llm,
                agent_models=agent_models,
                planner_models=planner_models,
                coder_models=coder_models,
                fixer_models=fixer_models,
                search_mode=search_mode,
                plan_beam_width=plan_beam_width,
                frontier_width=frontier_width,
                archive_size=archive_size,
                refine_rounds=refine_rounds,
                max_iters=max_iters,
                no_llm=False,
                allow_baseline=allow_baseline,
                g4f_recovery_rounds=g4f_recovery_rounds,
                baseline_patch_max_iters=baseline_patch_max_iters,
                g4f_recovery_max_iters=g4f_recovery_max_iters,
                g4f_recovery_sleep=g4f_recovery_sleep,
                worker_no_kill_process_group=worker_no_kill_process_group,
                print_generation=print_generation,
                print_generation_max_chars=print_generation_max_chars,
                g4f_async=g4f_async,
                max_response_chars=max_response_chars,
                g4f_request_timeout=g4f_request_timeout,
                g4f_stop_at_python_fence=g4f_stop_at_python_fence,
                strict_codegen=bool(keep_improving or self_improve_prompts),
                attempt_archive_dir=round_archive_dir if keep_improving else None,
            )

            candidate_solver_digest = _file_sha256(candidate_path)
            if reject_identical_candidates and keep_improving and candidate_solver_digest and best_solver_digest and candidate_solver_digest == best_solver_digest:
                generation_failure_streak += 1
                rejection_reasons = ['no_novelty_identical_solver_pre_score']
                score_history.append({
                    'round': round_idx,
                    'score': None,
                    'metric': None,
                    'accepted': False,
                    'path': str(candidate_path),
                    'candidate_solver_sha256': candidate_solver_digest,
                    'baseline_solver_sha256': best_solver_digest,
                    'rejection_reasons': rejection_reasons,
                    'failure_kind': rejection_reasons[0],
                    'candidate_archive_dir': str(round_archive_dir),
                })
                print(f'[improve] round {round_idx} rejected before scoring: candidate solver is byte-identical to current best.', flush=True)
                if (not live_candidate_seen) and max_codegen_failures_before_live_candidate > 0 and generation_failure_streak >= max_codegen_failures_before_live_candidate:
                    print(f'[improve] stopping early after {generation_failure_streak} consecutive codegen/no-novelty failures before any live candidate.', flush=True)
                    break
                continue

            live_candidate_seen = True
            generation_failure_streak = 0
            _validate_solver(candidate_path, spec.validator, smoke_vectors)

            round_hook_result: Optional[Dict[str, Any]] = None
            if validated_round_hook is not None:
                round_hook_result = validated_round_hook(round_idx, candidate_path)
                if isinstance(round_hook_result, dict) and round_hook_result.get('submitted'):
                    submitted_rounds.append(round_idx)

            candidate_score: Optional[int] = None
            candidate_submission_csv: Optional[Path] = None
            candidate_submission_digest: Optional[str] = None
            candidate_solver_digest = _file_sha256(candidate_path)
            per_row_delta_summary: Optional[dict[str, Any]] = None
            if scoring_enabled and puzzles_csv_for_score is not None:
                candidate_score, candidate_submission_csv = _score_solver_with_optional_artifact(
                    spec=spec,
                    solver_path=candidate_path,
                    puzzles_csv=puzzles_csv_for_score,
                    competition_format_slug=competition_format_slug,
                    vector_col_override=vector_col_override,
                    max_rows=max_rows,
                )
                candidate_submission_digest = _submission_path_digest(candidate_submission_csv)
                print(f'[improve] round {round_idx}: local score = {candidate_score}', flush=True)
                if write_per_row_delta and best_submission_csv is not None and candidate_submission_csv is not None:
                    delta_csv = candidate_path.parent / f'{candidate_path.stem}.round{round_idx}.per_row_delta.csv'
                    delta_json = candidate_path.parent / f'{candidate_path.stem}.round{round_idx}.per_row_delta.summary.json'
                    per_row_delta_summary = _write_per_row_delta(
                        baseline_csv=best_submission_csv,
                        candidate_csv=candidate_submission_csv,
                        out_csv=delta_csv,
                        out_json=delta_json,
                    )
                    if per_row_delta_summary.get('ok'):
                        print(
                            f"[improve] round {round_idx}: per-row delta improved={per_row_delta_summary.get('improved_rows')} "
                            f"regressed={per_row_delta_summary.get('regressed_rows')} net_delta={per_row_delta_summary.get('net_delta_moves')}",
                            flush=True,
                        )

            novelty_report = {
                'reject_identical_candidates': bool(reject_identical_candidates),
                'candidate_solver_sha256': candidate_solver_digest,
                'baseline_solver_sha256': best_solver_digest,
                'candidate_submission_digest': candidate_submission_digest,
                'baseline_submission_digest': best_submission_digest,
                'identical_solver': bool(candidate_solver_digest and best_solver_digest and candidate_solver_digest == best_solver_digest),
                'identical_submission': bool(candidate_submission_digest and best_submission_digest and candidate_submission_digest == best_submission_digest),
                'per_row_delta': per_row_delta_summary,
            }

            if candidate_round_hook is not None:
                try:
                    candidate_round_hook({
                        'round': round_idx,
                        'candidate_path': str(candidate_path),
                        'candidate_score': candidate_score,
                        'post_validation': round_hook_result,
                        'baseline_path': str(baseline_for_round),
                        'candidate_submission_csv': str(candidate_submission_csv) if candidate_submission_csv else None,
                        'candidate_solver_sha256': candidate_solver_digest,
                        'candidate_submission_digest': candidate_submission_digest,
                        'novelty_report': novelty_report,
                    })
                except KeyboardInterrupt:
                    raise
                except Exception as hook_exc:
                    print(f'[improve] WARNING: candidate_round_hook failed for round {round_idx}: {type(hook_exc).__name__}: {hook_exc}', flush=True)

            candidate_metric = _adaptive_metric_from_round(local_score=candidate_score, round_result=round_hook_result)

            accepted = False
            rejection_reasons: list[str] = []
            if reject_identical_candidates and keep_improving:
                if novelty_report.get('identical_solver'):
                    rejection_reasons.append('no_novelty_identical_solver')
                if novelty_report.get('identical_submission'):
                    rejection_reasons.append('no_novelty_identical_submission')
                if per_row_delta_summary and per_row_delta_summary.get('ok') and int(per_row_delta_summary.get('improved_rows') or 0) <= 0:
                    rejection_reasons.append('no_per_row_improvement')
            if not keep_improving:
                accepted = True
            elif scoring_enabled:
                accepted = _is_better_metric(best_metric, candidate_metric)
            else:
                accepted = not rejection_reasons
            if rejection_reasons:
                accepted = False

            score_history.append({
                'round': round_idx,
                'score': candidate_score,
                'metric': candidate_metric,
                'accepted': accepted,
                'path': str(candidate_path),
                'post_validation': round_hook_result,
                'novelty_report': novelty_report,
                'rejection_reasons': rejection_reasons,
                'failure_kind': rejection_reasons[0] if rejection_reasons else (None if accepted else 'score_not_improved'),
                'candidate_archive_dir': str(round_archive_dir),
            })

            if accepted:
                generation_failure_streak = 0
                if candidate_path != out_path:
                    shutil.copyfile(candidate_path, out_path)
                baseline_for_round = out_path
                best_candidate_path = out_path
                best_round = round_idx
                if scoring_enabled:
                    best_score = candidate_score
                    best_metric = candidate_metric
                    best_submission_csv = candidate_submission_csv
                    best_solver_digest = candidate_solver_digest
                    best_submission_digest = candidate_submission_digest
                    metric_source = (candidate_metric or {}).get('source')
                    metric_value = (candidate_metric or {}).get('value')
                    print(f'[improve] accepted round {round_idx} as new best {metric_source}={metric_value}.', flush=True)
                elif keep_improving:
                    print(f'[improve] accepted round {round_idx} as the latest validated solver.', flush=True)
            elif scoring_enabled:
                current_metric_source = (best_metric or {}).get('source')
                current_metric_value = (best_metric or {}).get('value')
                reason_text = ';'.join(rejection_reasons) if rejection_reasons else 'score_not_improved'
                print(f'[improve] round {round_idx} rejected ({reason_text}); current best {current_metric_source}={current_metric_value}; keeping round {best_round} output and continuing.', flush=True)
        except KeyboardInterrupt:
            raise
        except SystemExit as e:
            if not keep_improving:
                raise
            score_history.append({
                'round': round_idx,
                'score': None,
                'accepted': False,
                'path': str(candidate_path),
                'error': str(e),
            })
            print(f'[improve] round {round_idx} failed with SystemExit; continuing from best known solver: {e}', flush=True)
            continue
        except Exception as e:
            if not keep_improving:
                raise
            generation_failure_streak += 1
            score_history.append({
                'round': round_idx,
                'score': None,
                'accepted': False,
                'path': str(candidate_path),
                'error': f'{type(e).__name__}: {e}',
                'failure_kind': 'generation_or_validation_exception',
                'candidate_archive_dir': str(round_archive_dir),
            })
            print(f'[improve] round {round_idx} failed; continuing from best known solver: {type(e).__name__}: {e}', flush=True)
            if (not live_candidate_seen) and max_codegen_failures_before_live_candidate > 0 and generation_failure_streak >= max_codegen_failures_before_live_candidate:
                print(f'[improve] stopping early after {generation_failure_streak} consecutive failures before any live candidate.', flush=True)
                break
            continue

    if best_candidate_path is None:
        shutil.copyfile(baseline_origin, out_path)
        best_candidate_path = out_path

    _write_prompt_evolution_history(out_path, prompt_evolution_history)

    return {
        'best_score': best_score,
        'best_metric': best_metric,
        'best_round': best_round,
        'scoring_enabled': scoring_enabled,
        'history': score_history,
        'rounds_requested': rounds,
        'submitted_rounds': submitted_rounds,
        'selected_round_already_submitted': best_round in submitted_rounds if best_round is not None else False,
        'initial_baseline': str(baseline_origin),
        'best_solver_sha256': best_solver_digest,
        'best_submission_digest': best_submission_digest,
        'best_submission_csv': str(best_submission_csv) if best_submission_csv else None,
        'strict_guards': {
            'reject_identical_candidates': bool(reject_identical_candidates),
            'write_per_row_delta': bool(write_per_row_delta),
            'strict_codegen': bool(keep_improving or self_improve_prompts),
            'max_codegen_failures_before_live_candidate': int(max_codegen_failures_before_live_candidate),
        },
    }


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


def _normalize_explicit_baseline_path(spec: PipelineSpec, baseline_value: Optional[str], *, command_name: str) -> Optional[Path]:
    if not baseline_value:
        return None
    explicit_baseline = Path(baseline_value).resolve()
    if not explicit_baseline.exists():
        raise SystemExit(f"Explicit --baseline file does not exist: {explicit_baseline}")

    if explicit_baseline.suffix.lower() == '.csv':
        print(
            f"[baseline] WARNING: {command_name} received a CSV via --baseline ({explicit_baseline}). "
            f"This flag expects a solver .py file, so the CSV override will be ignored and the default solver baseline "
            f"{spec.baseline_solver} will be used for prompt grounding and fallback.",
            flush=True,
        )
        return None

    try:
        _validate_solver(explicit_baseline, spec.validator, _resolve_smoke_vectors(spec))
    except Exception as exc:
        print(
            f"[baseline] WARNING: explicit override {explicit_baseline} failed preflight ({exc}). "
            f"Falling back to default baseline {spec.baseline_solver}.",
            flush=True,
        )
        return None

    return explicit_baseline


def cmd_generate_solver(args: argparse.Namespace) -> None:
    spec = get_pipeline(args.competition) if args.competition else None
    if spec is None:
        raise SystemExit(
            "Unknown --competition. Run `python pipeline_cli.py list-pipelines` to see supported pipelines."
        )

    explicit_baseline = _normalize_explicit_baseline_path(spec, getattr(args, 'baseline', None), command_name='generate-solver')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Select prompt bundle (can be overridden, or switched via --prompt-variant)
    prompt_file, custom_prompts = _resolve_prompt_bundle(spec, args)
    if explicit_baseline is not None:
        effective_baseline = explicit_baseline
        adaptive_baseline_info = {
            'enabled': False,
            'uses_baseline': True,
            'from_scratch': _prompt_bundle_requests_from_scratch(prompt_file, custom_prompts),
            'supports_ranked_reuse': False,
            'bundle_slug': _adaptive_baseline_bundle_slug(prompt_file, custom_prompts),
            'effective_baseline': str(explicit_baseline),
            'default_baseline': str(spec.baseline_solver),
            'source': 'explicit_override',
        }
        print(f"[baseline] using explicit override: {explicit_baseline}", flush=True)
    else:
        effective_baseline, adaptive_baseline_info = _resolve_effective_baseline(spec, prompt_file, custom_prompts)

        if adaptive_baseline_info.get('enabled'):
            if adaptive_baseline_info.get('adaptive_exists') and adaptive_baseline_info.get('adaptive_valid') is False:
                print(
                    f"[adaptive-baseline] ignoring invalid saved baseline {adaptive_baseline_info.get('adaptive_solver')}: "
                    f"{adaptive_baseline_info.get('adaptive_invalid_reason')}",
                    flush=True,
                )
            print(
                f"[adaptive-baseline] using {adaptive_baseline_info.get('effective_baseline')} "
                f"for bundle={adaptive_baseline_info.get('bundle_slug')}",
                flush=True,
            )

    if args.no_llm:
        shutil.copyfile(effective_baseline, out_path)
        print(f"[generate-solver] --no-llm: copied baseline -> {out_path}")
        _validate_solver(out_path, spec.validator, _resolve_smoke_vectors(spec))
        return

    _gpu_diag_hint(args.models)

    adaptive_updates: list[dict[str, Any]] = []

    def _candidate_round_hook(payload: dict[str, Any]) -> None:
        if not adaptive_baseline_info.get('enabled'):
            return
        solver_candidate = Path(str(payload['candidate_path']))
        novelty = payload.get('novelty_report') if isinstance(payload.get('novelty_report'), dict) else {}
        if novelty.get('identical_solver') or novelty.get('identical_submission'):
            print('[adaptive-baseline] not promoting: identical solver/submission fingerprint', flush=True)
            return
        delta = novelty.get('per_row_delta') if isinstance(novelty.get('per_row_delta'), dict) else {}
        if delta.get('ok') and int(delta.get('improved_rows') or 0) <= 0:
            print('[adaptive-baseline] not promoting: no per-row improvement', flush=True)
            return
        manifest = _persist_adaptive_baseline(
            spec=spec,
            prompt_file=prompt_file,
            custom_prompts=custom_prompts,
            solver_path=solver_candidate,
            round_idx=int(payload['round']),
            local_score=payload.get('candidate_score'),
            round_result=payload.get('post_validation'),
        )
        if manifest is not None:
            adaptive_updates.append(manifest)
            metric = manifest.get('selection_metric') if isinstance(manifest.get('selection_metric'), dict) else {}
            print(
                f"[adaptive-baseline] promoted round {manifest.get('round')} -> {manifest.get('solver_path')} "
                f"via {metric.get('source')}={metric.get('value')}",
                flush=True,
            )

    puzzles_for_score = _resolve_default_puzzles(spec) if args.keep_improving else None
    result = _generate_solver_with_optional_improvement(
        spec=spec,
        out_path=out_path,
        prompt_file=prompt_file,
        custom_prompts=custom_prompts,
        llm=args.models,
        agent_models=args.agent_models,
        planner_models=args.planner_models,
        coder_models=args.coder_models,
        fixer_models=args.fixer_models,
        search_mode=args.search_mode,
        plan_beam_width=args.plan_beam_width,
        frontier_width=args.frontier_width,
        archive_size=args.archive_size,
        refine_rounds=args.refine_rounds,
        max_iters=args.max_iters,
        allow_baseline=args.allow_baseline,
        g4f_recovery_rounds=args.g4f_recovery_rounds,
        baseline_patch_max_iters=args.baseline_patch_max_iters,
        g4f_recovery_max_iters=args.g4f_recovery_max_iters,
        g4f_recovery_sleep=args.g4f_recovery_sleep,
        worker_no_kill_process_group=args.worker_no_kill_process_group,
        print_generation=args.print_generation,
        print_generation_max_chars=args.print_generation_max_chars,
        g4f_async=args.g4f_async,
        max_response_chars=args.max_response_chars,
        g4f_request_timeout=args.g4f_request_timeout,
        g4f_stop_at_python_fence=args.g4f_stop_at_python_fence,
        keep_improving=args.keep_improving,
        improvement_rounds=args.improvement_rounds,
        puzzles_csv_for_score=puzzles_for_score,
        competition_format_slug=spec.format_slug,
        initial_baseline=effective_baseline,
        candidate_round_hook=_candidate_round_hook if adaptive_baseline_info.get('enabled') else None,
        self_improve_prompts=getattr(args, 'self_improve_prompts', False),
        reject_identical_candidates=getattr(args, 'reject_identical_candidates', True),
        write_per_row_delta=getattr(args, 'write_per_row_delta', True),
    )
    if args.keep_improving:
        print(f"[improve] best_round={result.get('best_round')} best_score={result.get('best_score')}", flush=True)
    if adaptive_updates:
        latest_metric = adaptive_updates[-1].get('selection_metric') if isinstance(adaptive_updates[-1].get('selection_metric'), dict) else {}
        print(
            f"[adaptive-baseline] saved reusable baseline {adaptive_updates[-1].get('solver_path')} "
            f"with {latest_metric.get('source')}={latest_metric.get('value')}",
            flush=True,
        )


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

    allowed_moves = _load_allowed_moves_from_validator(spec.validator)

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


def cmd_kaggle_preflight(args: argparse.Namespace) -> None:
    spec = get_pipeline(args.competition)
    competition = args.submit_competition or (spec.competition if spec is not None else args.competition)

    _ensure_llm_puzzles_on_path()
    from src.kaggle_utils import preflight_submit_via_api, preflight_submit_via_cli  # type: ignore

    payload: dict[str, Any] = {
        'competition': competition,
        'requested_via': args.submit_via,
        'results': {},
    }
    ok = False

    if args.submit_via in {'auto', 'api'}:
        try:
            report = preflight_submit_via_api(
                competition,
                kaggle_json_path=args.kaggle_json,
                config_dir=args.kaggle_config_dir,
            )
            payload['results']['api'] = report
            ok = True
            if not args.json:
                _print_kaggle_preflight_report(report)
        except Exception as exc:
            payload['results']['api_error'] = str(exc)
            if args.submit_via == 'api':
                raise SystemExit(str(exc))
            if not args.json:
                print(f"[kaggle] preflight (api) failed: {exc}", flush=True)

    if args.submit_via in {'auto', 'cli'}:
        try:
            report = preflight_submit_via_cli(
                competition,
                credentials_path=args.kaggle_json,
                config_dir=args.kaggle_config_dir,
            )
            payload['results']['cli'] = report
            ok = True
            if not args.json:
                _print_kaggle_preflight_report(report)
        except Exception as exc:
            payload['results']['cli_error'] = str(exc)
            if args.submit_via == 'cli':
                raise SystemExit(str(exc))
            if not args.json:
                print(f"[kaggle] preflight (cli) failed: {exc}", flush=True)

    payload['ok'] = ok
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif ok:
        print(f"[kaggle] preflight OK for '{competition}'.", flush=True)
    else:
        raise SystemExit(f"Kaggle preflight failed for '{competition}'.")


def cmd_run(args: argparse.Namespace) -> None:
    # Pipeline selection by competition slug
    spec = get_pipeline(args.competition)
    if spec is None:
        raise SystemExit(
            f"Unknown competition/pipeline '{args.competition}'. Run `python pipeline_cli.py list-pipelines`."
        )

    explicit_baseline = _normalize_explicit_baseline_path(spec, getattr(args, 'baseline', None), command_name='run')

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
            "search_mode": args.search_mode,
            "plan_beam_width": args.plan_beam_width,
            "frontier_width": args.frontier_width,
            "archive_size": args.archive_size,
            "refine_rounds": args.refine_rounds,
            "max_iters": args.max_iters,
            "print_generation": bool(args.print_generation),
            "print_generation_max_chars": args.print_generation_max_chars,
            "g4f_async": args.g4f_async,
            "max_response_chars": args.max_response_chars,
            "g4f_request_timeout": args.g4f_request_timeout,
            "g4f_stop_at_python_fence": args.g4f_stop_at_python_fence,
            "keep_improving": bool(getattr(args, 'keep_improving', False)),
            "improvement_rounds": getattr(args, 'improvement_rounds', 1),
            "baseline_patch_max_iters": getattr(args, 'baseline_patch_max_iters', None),
            "g4f_recovery_max_iters": getattr(args, 'g4f_recovery_max_iters', None),
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

    if args.submit and not args.message:
        raise SystemExit("--submit requires --message")

    submit_availability = {
        'enabled': False,
        'source': 'submit_disabled',
        'credentials_path': None,
    }
    if args.submit:
        submit_availability = _resolve_kaggle_submit_availability(
            kaggle_json=args.kaggle_json,
            kaggle_config_dir=args.kaggle_config_dir,
        )
        if submit_availability.get('enabled'):
            source = submit_availability.get('source')
            cred_path = submit_availability.get('credentials_path')
            suffix = f" ({cred_path})" if cred_path else ''
            recovered_from = submit_availability.get('recovered_from_missing_explicit')
            print(f"[kaggle] live submit enabled via {source}{suffix}", flush=True)
            if recovered_from:
                print(
                    f"[kaggle] recovered missing credentials path {recovered_from} using {cred_path}",
                    flush=True,
                )
        else:
            reason = str(submit_availability.get('reason') or 'Kaggle live submit is unavailable.')
            if submit_availability.get('nonfatal'):
                print(f"[kaggle] WARNING: {reason}", flush=True)
            else:
                raise SystemExit(reason)
    effective_kaggle_json = (
        str(submit_availability.get('credentials_path'))
        if submit_availability.get('enabled') and submit_availability.get('credentials_path')
        else args.kaggle_json
    )

    improvement_result: dict[str, Any] | None = None
    per_round_submit_reports: list[dict[str, Any]] = []
    do_schema_check = (args.submit or args.schema_check) and (not args.no_schema_check)
    allowed_moves = _load_allowed_moves_from_validator(spec.validator)
    move_column, joiner = _resolve_submission_move_column(args.format or spec.competition)
    schema_sample: Path | None = None
    if do_schema_check:
        try:
            schema_sample = _resolve_sample_submission(spec)
        except Exception:
            schema_sample = None
    submit_during_improvement = bool(args.submit and submit_availability.get('enabled') and getattr(args, 'keep_improving', False) and not args.no_llm)
    callback_state = {'schema_warning_emitted': False}
    prompt_file: Path | None = None
    custom_prompts: Path | None = None
    effective_baseline = explicit_baseline or spec.baseline_solver
    adaptive_baseline_info: dict[str, Any] = {'enabled': False}
    adaptive_updates: list[dict[str, Any]] = []

    if not args.no_llm:
        prompt_file, custom_prompts = _resolve_prompt_bundle(spec, args)
        if explicit_baseline is not None:
            adaptive_baseline_info = {
                'enabled': False,
                'uses_baseline': True,
                'from_scratch': _prompt_bundle_requests_from_scratch(prompt_file, custom_prompts),
                'supports_ranked_reuse': False,
                'bundle_slug': _adaptive_baseline_bundle_slug(prompt_file, custom_prompts),
                'effective_baseline': str(explicit_baseline),
                'default_baseline': str(spec.baseline_solver),
                'source': 'explicit_override',
            }
            print(f"[baseline] using explicit override: {explicit_baseline}", flush=True)
        else:
            effective_baseline, adaptive_baseline_info = _resolve_effective_baseline(spec, prompt_file, custom_prompts)
            if adaptive_baseline_info.get('enabled'):
                print(
                    f"[adaptive-baseline] using {adaptive_baseline_info.get('effective_baseline')} "
                    f"for bundle={adaptive_baseline_info.get('bundle_slug')}",
                    flush=True,
                )
        report['adaptive_baseline'] = adaptive_baseline_info

    def _candidate_round_hook(payload: dict[str, Any]) -> None:
        if not adaptive_baseline_info.get('enabled') or prompt_file is None:
            return
        novelty = payload.get('novelty_report') if isinstance(payload.get('novelty_report'), dict) else {}
        if novelty.get('identical_solver') or novelty.get('identical_submission'):
            print('[adaptive-baseline] not promoting: identical solver/submission fingerprint', flush=True)
            return
        delta = novelty.get('per_row_delta') if isinstance(novelty.get('per_row_delta'), dict) else {}
        if delta.get('ok') and int(delta.get('improved_rows') or 0) <= 0:
            print('[adaptive-baseline] not promoting: no per-row improvement', flush=True)
            return
        manifest = _persist_adaptive_baseline(
            spec=spec,
            prompt_file=prompt_file,
            custom_prompts=custom_prompts,
            solver_path=Path(str(payload['candidate_path'])),
            round_idx=int(payload['round']),
            local_score=payload.get('candidate_score'),
            round_result=payload.get('post_validation'),
        )
        if manifest is not None:
            adaptive_updates.append(manifest)
            metric = manifest.get('selection_metric') if isinstance(manifest.get('selection_metric'), dict) else {}
            print(
                f"[adaptive-baseline] promoted round {manifest.get('round')} -> {manifest.get('solver_path')} "
                f"via {metric.get('source')}={metric.get('value')}",
                flush=True,
            )

    def _validated_round_hook(round_idx: int, candidate_solver_path: Path) -> dict[str, Any]:
        round_submission_csv = _round_submission_output_path(out_csv, round_idx)
        if round_submission_csv.exists():
            round_submission_csv.unlink()

        print(f"[improve] round {round_idx}: build submission before next iteration", flush=True)
        _build_submission(
            puzzles_csv=puzzles_csv,
            out_csv=round_submission_csv,
            competition_format_slug=args.format or spec.format_slug,
            solver_path=candidate_solver_path,
            spec=spec,
            vector_col_override=args.vector_col,
            max_rows=args.max_rows,
            no_progress=True,
        )

        round_result: dict[str, Any] = {
            'round': round_idx,
            'submission_csv': str(round_submission_csv),
            'submitted': False,
        }

        if allowed_moves:
            round_result['move_validation'] = _validate_submission_move_tokens(
                submission_csv=round_submission_csv,
                move_column=move_column,
                allowed_moves=allowed_moves,
                joiner=joiner,
            )

        if do_schema_check:
            if schema_sample is None:
                if not callback_state['schema_warning_emitted']:
                    print(f"[schema] WARNING: no bundled sample_submission.csv found for '{spec.key}'. Skipping.")
                    callback_state['schema_warning_emitted'] = True
            else:
                round_result['schema'] = {
                    'sample': str(schema_sample),
                    **_validate_submission_schema(
                        submission_csv=round_submission_csv,
                        sample_submission_csv=schema_sample,
                        check_ids=(not args.no_schema_check_ids),
                    ),
                }

        if submit_during_improvement:
            submit_comp = args.submit_competition or spec.competition
            _improvement_rounds = max(1, int(getattr(args, 'improvement_rounds', 1)))
            round_message = args.message if _improvement_rounds <= 1 else f"{args.message} [round {round_idx}/{_improvement_rounds}]"
            print(f"[improve] round {round_idx}: submit to Kaggle before next iteration", flush=True)
            try:
                round_result['kaggle_submit'] = _kaggle_submit(
                    competition=submit_comp,
                    submission_csv=round_submission_csv,
                    message=round_message,
                    kaggle_json=effective_kaggle_json,
                    submit_via=args.submit_via,
                    kaggle_config_dir=args.kaggle_config_dir,
                )
                round_result['submitted'] = bool(round_result['kaggle_submit'].get('status') or round_result['kaggle_submit'].get('submitted', True))
                if round_result['submitted']:
                    per_round_submit_reports.append(round_result)
            except KeyboardInterrupt:
                raise
            except SystemExit as submit_exc:
                round_result['kaggle_submit'] = _nonfatal_kaggle_submit_report(submit_exc, submit_comp)
                print(f"[kaggle] WARNING: round {round_idx} live submit failed; keeping validated solver and continuing: {submit_exc}", flush=True)
            except Exception as submit_exc:
                round_result['kaggle_submit'] = _nonfatal_kaggle_submit_report(submit_exc, submit_comp)
                print(f"[kaggle] WARNING: round {round_idx} live submit failed; keeping validated solver and continuing: {type(submit_exc).__name__}: {submit_exc}", flush=True)

        return round_result

    try:
        t0 = _stage("generate solver")
        report["stages"]["generate_solver"] = {"start": time.time()}
        if args.no_llm:
            shutil.copyfile(effective_baseline, solver_path)
            print(f"[run] --no-llm: copied baseline solver -> {solver_path}")
        else:
            assert prompt_file is not None
            improvement_result = _generate_solver_with_optional_improvement(
                spec=spec,
                out_path=solver_path,
                prompt_file=prompt_file,
                custom_prompts=custom_prompts,
                llm=args.models,
                agent_models=args.agent_models,
                planner_models=args.planner_models,
                coder_models=args.coder_models,
                fixer_models=args.fixer_models,
                search_mode=args.search_mode,
                plan_beam_width=args.plan_beam_width,
                frontier_width=args.frontier_width,
                archive_size=args.archive_size,
                refine_rounds=args.refine_rounds,
                max_iters=args.max_iters,
                allow_baseline=args.allow_baseline,
                g4f_recovery_rounds=args.g4f_recovery_rounds,
                baseline_patch_max_iters=args.baseline_patch_max_iters,
                g4f_recovery_max_iters=args.g4f_recovery_max_iters,
                g4f_recovery_sleep=args.g4f_recovery_sleep,
                worker_no_kill_process_group=args.worker_no_kill_process_group,
                print_generation=args.print_generation,
                print_generation_max_chars=args.print_generation_max_chars,
                g4f_async=args.g4f_async,
                max_response_chars=args.max_response_chars,
                g4f_request_timeout=args.g4f_request_timeout,
                g4f_stop_at_python_fence=args.g4f_stop_at_python_fence,
                keep_improving=getattr(args, 'keep_improving', False),
                improvement_rounds=getattr(args, 'improvement_rounds', 1),
                puzzles_csv_for_score=puzzles_csv if getattr(args, 'keep_improving', False) else None,
                competition_format_slug=args.format or spec.format_slug,
                vector_col_override=args.vector_col,
                max_rows=args.max_rows,
                validated_round_hook=_validated_round_hook if submit_during_improvement else None,
                initial_baseline=effective_baseline,
                candidate_round_hook=_candidate_round_hook if adaptive_baseline_info.get('enabled') else None,
                self_improve_prompts=getattr(args, 'self_improve_prompts', False),
                reject_identical_candidates=getattr(args, 'reject_identical_candidates', True),
                write_per_row_delta=getattr(args, 'write_per_row_delta', True),
            )
            report['stages']['generate_solver']['improvement'] = improvement_result
            if per_round_submit_reports:
                report['round_kaggle_submissions'] = per_round_submit_reports
            if adaptive_updates:
                report['adaptive_baseline_updates'] = adaptive_updates

        report["stages"]["generate_solver"]["end"] = time.time()
        report["stages"]["generate_solver"]["seconds"] = report["stages"]["generate_solver"]["end"] - report["stages"]["generate_solver"]["start"]
        _stage_done("generate solver", t0)

        # Smoke validate (generation helper already validated in-loop; keep a final explicit check for the selected artifact)
        t1 = _stage("validate solver")
        report["stages"]["validate_solver"] = {"start": time.time()}
        _validate_solver(solver_path, spec.validator, _resolve_smoke_vectors(spec))
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
        if do_schema_check:
            if schema_sample is None:
                print(f"[schema] WARNING: no bundled sample_submission.csv found for '{spec.key}'. Skipping.")
            else:
                tsc = _stage("schema check")
                report["stages"]["schema_check"] = {"start": time.time()}
                stats = _validate_submission_schema(
                    submission_csv=candidate_out,
                    sample_submission_csv=schema_sample,
                    check_ids=(not args.no_schema_check_ids),
                )
                report["schema"] = {"sample": str(schema_sample), **stats}
                report["stages"]["schema_check"]["end"] = time.time()
                report["stages"]["schema_check"]["seconds"] = report["stages"]["schema_check"]["end"] - report["stages"]["schema_check"]["start"]
                _stage_done("schema check", tsc)

        _finalize_submission_output(candidate_out, out_csv)

        # Optionally submit to Kaggle
        if args.submit:
            if not submit_availability.get('enabled'):
                report["kaggle_submit"] = {
                    "mode": "skipped",
                    "submitted": False,
                    **submit_availability,
                }
                print(f"[kaggle] skipping final live submit: {submit_availability.get('reason')}", flush=True)
                if getattr(args, 'require_submit_success', True):
                    raise RuntimeError(f"Kaggle submit was required but unavailable: {submit_availability.get('reason')}")
            elif submit_during_improvement and improvement_result is not None and improvement_result.get('selected_round_already_submitted'):
                report["kaggle_submit"] = {
                    "mode": "already_submitted_in_round",
                    "selected_round": improvement_result.get('best_round'),
                    "submitted_rounds": improvement_result.get('submitted_rounds', []),
                }
                print(
                    f"[kaggle] final selected solver was already submitted during improvement round {improvement_result.get('best_round')}; skipping duplicate final submit.",
                    flush=True,
                )
            else:
                t3 = _stage("submit to Kaggle")
                report["stages"]["submit_kaggle"] = {"start": time.time()}
                submit_comp = args.submit_competition or spec.competition
                try:
                    kaggle_submit_report = _kaggle_submit(
                        competition=submit_comp,
                        submission_csv=out_csv,
                        message=args.message,
                        kaggle_json=effective_kaggle_json,
                        submit_via=args.submit_via,
                        kaggle_config_dir=args.kaggle_config_dir,
                    )
                except KeyboardInterrupt:
                    raise
                except SystemExit as submit_exc:
                    kaggle_submit_report = _nonfatal_kaggle_submit_report(submit_exc, submit_comp)
                    print(f"[kaggle] WARNING: final live submit failed non-fatally: {submit_exc}", flush=True)
                except Exception as submit_exc:
                    kaggle_submit_report = _nonfatal_kaggle_submit_report(submit_exc, submit_comp)
                    print(f"[kaggle] WARNING: final live submit failed non-fatally: {type(submit_exc).__name__}: {submit_exc}", flush=True)
                report["kaggle_submit"] = kaggle_submit_report
                report["stages"]["submit_kaggle"]["end"] = time.time()
                report["stages"]["submit_kaggle"]["seconds"] = report["stages"]["submit_kaggle"]["end"] - report["stages"]["submit_kaggle"]["start"]
                _stage_done("submit to Kaggle", t3)
                if getattr(args, 'require_submit_success', True) and not bool(kaggle_submit_report.get('submitted') or kaggle_submit_report.get('status')):
                    raise RuntimeError(f"Kaggle submit was required but not confirmed: {kaggle_submit_report.get('error') or kaggle_submit_report}")

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
        _validate_solver(solver_path, spec.validator, _resolve_smoke_vectors(spec))

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
            # id,n,permutation
            with puzzles_csv.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "n", "permutation"])
                w.writeheader()
                w.writerow({"id": "0", "n": "4", "permutation": "3,1,2,0"})
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
    sp.set_defaults(func=cmd_show_pipeline)

    sp = sub.add_parser("generate-solver", help="Generate/repair a solver with AgentLaboratory")
    sp.add_argument("--competition", required=True, help="Competition slug / pipeline key")
    sp.add_argument("--out", required=True, help="Output path for generated solver")
    sp.add_argument("--prompt-file", default=None, help="Override user prompt file")
    sp.add_argument("--prompt-variant", default=None, choices=["regular", "improved", "dataset_adapted", "structured", "heuristic_boosted", "master_hybrid", "neighbour_model_hybrid", "score_guarded", "algorithmic_population", "portfolio_orchestrated", "hard_row_routed", "exact_score_population", "strict_self_improvement", "failure_aware_self_improvement"], help="Select a competition prompt bundle variant when available")
    sp.add_argument("--custom-prompts", default=None, help="Override AgentLaboratory custom prompts JSON")
    sp.add_argument(
        "--models",
        dest="models",
        default="gpt-4o-mini",
        help=(
            "Comma-separated model list (passed to AgentLaboratory --models). "
            "Bare names use g4f backend (remote providers). "
            "You can also pass explicit backends like 'local:<hf_model_id>', 'ollama:<model>', 'vllm:<model>', "
            "'lmstudio:<model>', 'g4fapi:<model>', or plain g4f model names like 'gpt-4o-mini'."
        ),
    )
    sp.add_argument("--llm", dest="models", default=None, help=argparse.SUPPRESS)
    sp.add_argument("--agent-models", default=None, help="Optional per-agent override mapping for AgentLaboratory, e.g. 'planner=claude-3.5-sonnet;coder=deepseek-chat,qwen2.5-coder;fixer=gpt-4o-mini'.")
    sp.add_argument("--planner-models", default=None, help="Optional model list override for the planner agent.")
    sp.add_argument("--coder-models", default=None, help="Optional model list override for the coder agent.")
    sp.add_argument("--fixer-models", default=None, help="Optional model list override for the fixer agent.")
    sp.add_argument("--search-mode", default="hybrid", choices=["classic", "hybrid"], help="classic = single-plan linear search; hybrid = multi-plan frontier search with experiment-memory refinement.")
    sp.add_argument("--plan-beam-width", type=int, default=3, help="Planner beam width in hybrid mode.")
    sp.add_argument("--frontier-width", type=int, default=6, help="Planner/coder frontier width per hybrid round.")
    sp.add_argument("--archive-size", type=int, default=6, help="How many failed attempts to retain as hybrid experiment memory.")
    sp.add_argument("--refine-rounds", type=int, default=1, help="How many planner refinement rounds to run in hybrid mode.")
    sp.add_argument("--max-iters", type=int, default=100000)
    sp.add_argument("--g4f-recovery-rounds", type=int, default=None, help="Extra recovery rounds before offline fallback (forwarded to AgentLaboratory).")
    sp.add_argument("--baseline-patch-max-iters", type=int, default=None, help="Optional fixer iterations for the baseline-patcher stage (forwarded as AGENTLAB_BASELINE_PATCH_MAX_ITERS).")
    sp.add_argument("--g4f-recovery-max-iters", type=int, default=None, help="Fixer iterations per recovery round (forwarded to AgentLaboratory).")
    sp.add_argument("--g4f-recovery-sleep", type=float, default=None, help="Cooldown in seconds before each recovery round (forwarded to AgentLaboratory).")
    sp.add_argument("--worker-no-kill-process-group", action="store_true", help="Do not hard-kill the entire worker process group on timeout; only terminate the worker process itself.")
    sp.add_argument("--print-generation", action="store_true", help="Print raw planner/coder/fixer generations.")
    sp.add_argument("--print-generation-max-chars", type=int, default=None, help="Maximum number of characters to print per generation.")
    sp.add_argument("--g4f-async", dest="g4f_async", action="store_true", help="Use g4f AsyncClient inside the pipeline worker path.")
    sp.add_argument("--no-g4f-async", dest="g4f_async", action="store_false", help="Disable g4f AsyncClient and fall back to ChatCompletion.create.")
    sp.add_argument("--max-response-chars", type=int, default=None, help="Optional hard cap on captured g4f response size. 0 disables clipping.")
    sp.add_argument("--g4f-request-timeout", type=float, default=None, help="Optional timeout passed through to g4f requests.")
    sp.add_argument("--g4f-stop-at-python-fence", dest="g4f_stop_at_python_fence", action="store_true", help="Trim g4f output after the first complete ```python``` fence.")
    sp.add_argument("--no-g4f-stop-at-python-fence", dest="g4f_stop_at_python_fence", action="store_false", help="Do not trim g4f output at the first python fence.")
    sp.set_defaults(g4f_async=None, g4f_stop_at_python_fence=None)
    sp.add_argument("--keep-improving", action="store_true", help="Do not stop after the first validated solver; keep running additional locally scored improvement rounds.")
    sp.add_argument("--improvement-rounds", type=int, default=3, help="How many validated generation rounds to run when --keep-improving is enabled.")
    sp.add_argument("--self-improve-prompts", action="store_true", help="For competition-specific pipelines that support it, synthesize a stronger round-specific prompt bundle from the previous accepted solver before each new improvement round.")
    sp.add_argument("--reject-identical-candidates", action=argparse.BooleanOptionalAction, default=True, help="Reject keep-improving candidates whose solver/submission fingerprint is identical to the current best artifact.")
    sp.add_argument("--write-per-row-delta", action=argparse.BooleanOptionalAction, default=True, help="Write per-row candidate-vs-best move-count deltas for every locally scored improvement round.")
    sp.add_argument("--allow-baseline", action="store_true")
    sp.add_argument("--baseline", default=None, help="Optional explicit baseline solve_module.py override used for prompt grounding, fallback, and --no-llm.")
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


    sp = sub.add_parser("kaggle-preflight", help="Check Kaggle submission prerequisites (version + competition access) without uploading")
    sp.add_argument("--competition", required=True, help="Competition slug / pipeline key")
    sp.add_argument("--kaggle-json", default=None, help="Path to a Kaggle credentials file (legacy kaggle.json or access_token)")
    sp.add_argument("--kaggle-config-dir", default=None, help="Optional directory to place a temporary kaggle.json copy")
    sp.add_argument("--submit-via", default="auto", choices=["auto","api","cli"], help="Which submission path to preflight: auto (check both), api, or cli")
    sp.add_argument("--submit-competition", dest="submit_competition", default=None, help="Override Kaggle competition slug used for preflight")
    sp.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    sp.set_defaults(func=cmd_kaggle_preflight)

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
    sp.add_argument("--prompt-variant", default=None, choices=["regular", "improved", "dataset_adapted", "structured", "heuristic_boosted", "master_hybrid", "neighbour_model_hybrid", "score_guarded", "algorithmic_population", "portfolio_orchestrated", "hard_row_routed", "exact_score_population", "strict_self_improvement", "failure_aware_self_improvement"], help="Select a competition prompt bundle variant when available")
    sp.add_argument("--custom-prompts", default=None, help="Override custom prompts JSON")
    sp.add_argument(
        "--models",
        dest="models",
        default="gpt-4o-mini",
        help=(
            "Comma-separated model list (passed to AgentLaboratory --models). "
            "Bare names use g4f backend (remote providers). "
            "You can also pass explicit backends like 'local:<hf_model_id>', 'ollama:<model>', 'vllm:<model>', "
            "'lmstudio:<model>', 'g4fapi:<model>', or plain g4f model names like 'gpt-4o-mini'."
        ),
    )
    sp.add_argument("--llm", dest="models", default=None, help=argparse.SUPPRESS)
    sp.add_argument("--agent-models", default=None, help="Optional per-agent override mapping for AgentLaboratory, e.g. 'planner=claude-3.5-sonnet;coder=deepseek-chat,qwen2.5-coder;fixer=gpt-4o-mini'.")
    sp.add_argument("--planner-models", default=None, help="Optional model list override for the planner agent.")
    sp.add_argument("--coder-models", default=None, help="Optional model list override for the coder agent.")
    sp.add_argument("--fixer-models", default=None, help="Optional model list override for the fixer agent.")
    sp.add_argument("--search-mode", default="hybrid", choices=["classic", "hybrid"], help="classic = single-plan linear search; hybrid = multi-plan frontier search with experiment-memory refinement.")
    sp.add_argument("--plan-beam-width", type=int, default=3, help="Planner beam width in hybrid mode.")
    sp.add_argument("--frontier-width", type=int, default=6, help="Planner/coder frontier width per hybrid round.")
    sp.add_argument("--archive-size", type=int, default=6, help="How many failed attempts to retain as hybrid experiment memory.")
    sp.add_argument("--refine-rounds", type=int, default=1, help="How many planner refinement rounds to run in hybrid mode.")
    sp.add_argument("--max-iters", type=int, default=100000)
    sp.add_argument("--g4f-recovery-rounds", type=int, default=None, help="Extra recovery rounds before offline fallback (forwarded to AgentLaboratory).")
    sp.add_argument("--baseline-patch-max-iters", type=int, default=None, help="Optional fixer iterations for the baseline-patcher stage (forwarded as AGENTLAB_BASELINE_PATCH_MAX_ITERS).")
    sp.add_argument("--g4f-recovery-max-iters", type=int, default=None, help="Fixer iterations per recovery round (forwarded to AgentLaboratory).")
    sp.add_argument("--g4f-recovery-sleep", type=float, default=None, help="Cooldown in seconds before each recovery round (forwarded to AgentLaboratory).")
    sp.add_argument("--worker-no-kill-process-group", action="store_true", help="Do not hard-kill the entire worker process group on timeout; only terminate the worker process itself.")
    sp.add_argument("--print-generation", action="store_true", help="Print raw planner/coder/fixer generations.")
    sp.add_argument("--print-generation-max-chars", type=int, default=None, help="Maximum number of characters to print per generation.")
    sp.add_argument("--g4f-async", dest="g4f_async", action="store_true", help="Use g4f AsyncClient inside the pipeline worker path.")
    sp.add_argument("--no-g4f-async", dest="g4f_async", action="store_false", help="Disable g4f AsyncClient and fall back to ChatCompletion.create.")
    sp.add_argument("--max-response-chars", type=int, default=None, help="Optional hard cap on captured g4f response size. 0 disables clipping.")
    sp.add_argument("--g4f-request-timeout", type=float, default=None, help="Optional timeout passed through to g4f requests.")
    sp.add_argument("--g4f-stop-at-python-fence", dest="g4f_stop_at_python_fence", action="store_true", help="Trim g4f output after the first complete ```python``` fence.")
    sp.add_argument("--no-g4f-stop-at-python-fence", dest="g4f_stop_at_python_fence", action="store_false", help="Do not trim g4f output at the first python fence.")
    sp.set_defaults(g4f_async=None, g4f_stop_at_python_fence=None)
    sp.add_argument("--keep-improving", action="store_true", help="Do not stop after the first validated solver; keep running additional locally scored improvement rounds.")
    sp.add_argument("--improvement-rounds", type=int, default=3, help="How many validated generation rounds to run when --keep-improving is enabled.")
    sp.add_argument("--self-improve-prompts", action="store_true", help="For competition-specific pipelines that support it, synthesize a stronger round-specific prompt bundle from the previous accepted solver before each new improvement round.")
    sp.add_argument("--reject-identical-candidates", action=argparse.BooleanOptionalAction, default=True, help="Reject keep-improving candidates whose solver/submission fingerprint is identical to the current best artifact.")
    sp.add_argument("--write-per-row-delta", action=argparse.BooleanOptionalAction, default=True, help="Write per-row candidate-vs-best move-count deltas for every locally scored improvement round.")
    sp.add_argument("--allow-baseline", action="store_true")
    sp.add_argument("--baseline", default=None, help="Optional explicit baseline solve_module.py override used for prompt grounding, fallback, and --no-llm.")
    sp.add_argument("--no-llm", action="store_true")
    sp.add_argument("--format", default=None, help="Override llm-puzzles format slug")
    sp.add_argument("--vector-col", default=None, help="Override state column")
    sp.add_argument("--max-rows", type=int, default=None)
    sp.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    sp.add_argument("--submit", action="store_true")
    sp.add_argument("--require-submit-success", action=argparse.BooleanOptionalAction, default=True, help="When --submit is enabled, fail the run unless Kaggle upload is confirmed.")
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
    sp.add_argument("--probe-mode", choices=["pipeline", "async"], default="pipeline", help="How to probe candidates: pipeline-compatible worker path or AsyncClient")
    sp.add_argument("--concurrency", type=int, default=5, help="Maximum number of concurrent AsyncClient probes")
    sp.set_defaults(func=cmd_check_g4f_models)

    sp = sub.add_parser("selftest", help="Offline smoke tests")
    sp.set_defaults(func=cmd_selftest)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    rewritten_argv, rewrite_note = _rewrite_embedded_kaggle_submit(raw_argv)
    args, unknown = parser.parse_known_args(rewritten_argv)
    if unknown:
        parser.error(_format_unknown_args_error(unknown))
    if rewrite_note:
        print(rewrite_note, flush=True)
    args.func(args)


if __name__ == "__main__":
    main()
