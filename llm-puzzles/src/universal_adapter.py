from __future__ import annotations
import csv, os
from typing import Callable, Dict, Any, Iterable, List, Union
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

def build_submission(puzzles_csv: str, output_csv: str, competition: str, solver: Callable[[Dict[str,str], CompConfig], SolverRet]) -> None:
    cfg = get_config(competition)
    with open(puzzles_csv, newline="") as f:
        reader = csv.DictReader(f)
        if cfg.puzzles_id_field not in reader.fieldnames:
            raise ValueError(f"'{cfg.puzzles_id_field}' column not found in {puzzles_csv}. Fields: {reader.fieldnames}")
        rows = list(reader)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as w:
        writer = csv.writer(w)
        writer.writerow(cfg.submission_headers)
        for row in rows:
            rid = row[cfg.puzzles_id_field]
            result = solver(row, cfg)
            moves = _normalize_moves(result, cfg)
            record = {"id": rid, "moves": moves}
            # map to headers
            writer.writerow([record.get(k) for k in cfg.header_keys])
