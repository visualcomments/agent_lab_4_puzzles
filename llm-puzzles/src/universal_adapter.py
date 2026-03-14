from __future__ import annotations

import csv
import os
import sys
from typing import Callable, Dict, Any, List, Union

from .comp_registry import get_config, CompConfig

SolverRet = Union[str, List[str], Dict[str, Any]]


def _normalize_moves(out: SolverRet, cfg: CompConfig) -> str:
    # Accepts: string -> use as is; list[str] -> join; dict -> try cfg.moves_key or "moves"
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)):
        return cfg.move_joiner.join(map(str, out))
    if isinstance(out, dict):
        moves = out.get(cfg.moves_key, out.get("moves", ""))
        if isinstance(moves, (list, tuple)):
            return cfg.move_joiner.join(map(str, moves))
        return str(moves or "")
    return ""


def _progress_print(i: int, total: int, desc: str) -> None:
    """Lightweight progress bar without external deps (stderr)."""
    if total <= 0:
        return
    pct = (i / total) * 100.0
    bar_w = 24
    filled = int(bar_w * pct / 100.0)
    bar = "=" * filled + "." * (bar_w - filled)
    sys.stderr.write(f"\r[{bar}] {pct:6.2f}%  {desc} ({i}/{total})")
    sys.stderr.flush()


def build_submission(
    puzzles_csv: str,
    output_csv: str,
    competition: str,
    solver: Callable[[Dict[str, str], CompConfig], SolverRet],
    max_rows: int | None = None,
    progress: bool = False,
    progress_desc: str = "building submission",
) -> None:
    cfg = get_config(competition)
    with open(puzzles_csv, newline="") as f:
        reader = csv.DictReader(f)
        if cfg.puzzles_id_field not in reader.fieldnames:
            raise ValueError(
                f"'{cfg.puzzles_id_field}' column not found in {puzzles_csv}. Fields: {reader.fieldnames}"
            )
        rows = list(reader)

    if max_rows is not None:
        rows = rows[:max_rows]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as w:
        writer = csv.writer(w)
        writer.writerow(cfg.submission_headers)

        total = len(rows)
        step = max(1, total // 200)  # ~200 updates max

        for i, row in enumerate(rows, 1):
            if progress and (i == 1 or i == total or i % step == 0):
                _progress_print(i, total, progress_desc)

            rid = row[cfg.puzzles_id_field]
            result = solver(row, cfg)
            moves = _normalize_moves(result, cfg)

            # Include original row fields as well, so comp_registry can map extra columns
            record = dict(row)
            record["id"] = rid
            record["moves"] = moves

            writer.writerow([record.get(k) for k in cfg.header_keys])

        if progress:
            sys.stderr.write("\n")
            sys.stderr.flush()
