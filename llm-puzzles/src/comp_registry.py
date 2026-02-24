from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

@dataclass(init=False)
class CompConfig:
    slug: str
    submission_headers: List[str] | None = None
    header_keys: List[str] | None = None
    puzzles_id_field: str = "id"
    moves_key: str = "moves"
    move_joiner: str = "."

    def __init__(
        self,
        slug: str,
        submission_headers: List[str] | None = None,
        header_keys: List[str] | None = None,
        puzzles_id_field: str = "id",
        moves_key: str = "moves",
        move_joiner: str = ".",
        **kwargs
    ):
        if "id_col" in kwargs and puzzles_id_field == "id":
            puzzles_id_field = kwargs.pop("id_col")
        if "move_col" in kwargs and moves_key == "moves":
            moves_key = kwargs.pop("move_col")
        if "joiner" in kwargs and move_joiner == ".":
            move_joiner = kwargs.pop("joiner")

        self.slug = slug
        self.submission_headers = submission_headers if submission_headers is not None else ["id", "moves"]
        self.header_keys = header_keys if header_keys is not None else ["id", "moves"]
        self.puzzles_id_field = puzzles_id_field
        self.moves_key = moves_key
        self.move_joiner = move_joiner

DEFAULT = CompConfig(slug="generic-id-moves")

REGISTRY: Dict[str, CompConfig] = {
    "cayley-py-444-cube": CompConfig(
        slug="cayley-py-444-cube",
        submission_headers=["initial_state_id","path"],
        header_keys=["id","moves"],
        puzzles_id_field="initial_state_id",
        moves_key="path",
        move_joiner="."
    ),
    "format/initial_state_id+path": CompConfig(
        slug="format/initial_state_id+path",
        submission_headers=["initial_state_id","path"],
        header_keys=["id","moves"],
        puzzles_id_field="initial_state_id",
        moves_key="path",
        move_joiner="."
    ),
    "format/moves-dot": CompConfig(
        slug="format/moves-dot",
        submission_headers=["id","moves"],
        header_keys=["id","moves"],
        id_col="id",
        move_col="moves",
        joiner="."
    ),
    "format/moves-space": CompConfig(
        slug="format/moves-space",
        submission_headers=["id","moves"],
        header_keys=["id","moves"],
        id_col="id",
        move_col="moves",
        joiner=" "
    ),
}

def get_config(slug: str) -> CompConfig:
    return REGISTRY.get(slug, DEFAULT)
