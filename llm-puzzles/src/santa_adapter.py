from __future__ import annotations
import csv, os
from typing import Callable, List, Dict

# Santa 2023 expects a CSV with header: id,moves
# - id: row id (string/integer from puzzles.csv)
# - moves: dot-separated tokens like: f0.r1.-d2 ... (possibly empty for no moves)
# See: competition evaluation page and public sample submissions.

def solve_dataset(puzzles_csv: str, output_csv: str, solver: Callable[[Dict[str,str]], str]) -> None:
    """Read puzzles.csv (as downloaded from Kaggle), call solver(row)->moves string,
    and write submission CSV with columns id,moves."""
    with open(puzzles_csv, newline="") as f:
        reader = csv.DictReader(f)
        if "id" not in reader.fieldnames:
            raise ValueError("puzzles.csv must contain an 'id' column")
        rows = list(reader)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as w:
        writer = csv.writer(w)
        writer.writerow(["id", "moves"])  # required header
        for row in rows:
            rid = row["id"]
            moves = solver(row)  # must be a dot-joined string per Santa format
            if moves is None:
                moves = ""
            writer.writerow([rid, moves])
