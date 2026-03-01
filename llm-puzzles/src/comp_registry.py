from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(init=False)
class CompConfig:
    """Formatting rules for a Kaggle competition submission.

    Notes
    -----
    llm-puzzles uses a generic adapter that always produces an internal record
    with keys:
      - id
      - moves
    and (optionally) any original fields from the puzzles CSV row.

    `submission_headers` are the column names to write.
    `header_keys` are the keys to pull from the internal record for each column.

    Example:
        submission_headers=["initial_state_id","path"]
        header_keys=["id","moves"]

    will write the internal record's 'id' under 'initial_state_id' and
    'moves' under 'path'.
    """

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
        **kwargs,
    ):
        # Back-compat aliases
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

# Common formats
FORMAT_INITIAL_STATE_ID_PATH = CompConfig(
    slug="format/initial_state_id+path",
    submission_headers=["initial_state_id", "path"],
    header_keys=["id", "moves"],
    puzzles_id_field="initial_state_id",
    moves_key="path",
    move_joiner=".",
)


REGISTRY: Dict[str, CompConfig] = {
    # Generic format slugs
    "format/initial_state_id+path": FORMAT_INITIAL_STATE_ID_PATH,
    "format/moves-dot": CompConfig(
        slug="format/moves-dot",
        submission_headers=["id", "moves"],
        header_keys=["id", "moves"],
        id_col="id",
        move_col="moves",
        joiner=".",
    ),
    "format/moves-space": CompConfig(
        slug="format/moves-space",
        submission_headers=["id", "moves"],
        header_keys=["id", "moves"],
        id_col="id",
        move_col="moves",
        joiner=" ",
    ),



    # LRX puzzle series (submission schema differs per competition)
    # lrx-discover-math-gods-algorithm: submission echoes the permutation string
    "lrx-discover-math-gods-algorithm": CompConfig(
        slug="lrx-discover-math-gods-algorithm",
        submission_headers=["permutation", "solution"],
        header_keys=["permutation", "moves"],
        puzzles_id_field="permutation",
        moves_key="solution",
        move_joiner=".",
    ),

    # lrx-oeis-a-186783-brainstorm-math-conjecture: submission keyed by n, moves concatenated (no delimiter)
    "lrx-oeis-a-186783-brainstorm-math-conjecture": CompConfig(
        slug="lrx-oeis-a-186783-brainstorm-math-conjecture",
        submission_headers=["n", "solution"],
        header_keys=["n", "moves"],
        puzzles_id_field="n",
        moves_key="solution",
        move_joiner="",
    ),
    # CayleyPy series (most are initial_state_id + path)
    "cayley-py-444-cube": FORMAT_INITIAL_STATE_ID_PATH,
    "cayley-py-professor-tetraminx-solve-optimally": FORMAT_INITIAL_STATE_ID_PATH,
    "cayley-py-megaminx": FORMAT_INITIAL_STATE_ID_PATH,
    "cayleypy-ihes-cube": FORMAT_INITIAL_STATE_ID_PATH,
    "cayleypy-christophers-jewel": FORMAT_INITIAL_STATE_ID_PATH,
    "cayleypy-reversals": FORMAT_INITIAL_STATE_ID_PATH,
    "cayleypy-transposons": FORMAT_INITIAL_STATE_ID_PATH,
    "cayleypy-glushkov": FORMAT_INITIAL_STATE_ID_PATH,

    # Pancake has a capitalized Kaggle slug; keep both keys
    "CayleyPy-pancake": FORMAT_INITIAL_STATE_ID_PATH,
    "cayleypy-pancake": FORMAT_INITIAL_STATE_ID_PATH,

    # RapaportM2 has a different submission schema (echoes the permutation)
    "cayleypy-rapapport-m2": CompConfig(
        slug="cayleypy-rapapport-m2",
        submission_headers=["id", "permutation", "solution"],
        header_keys=["id", "permutation", "moves"],
        puzzles_id_field="id",
        moves_key="solution",
        move_joiner=".",
    ),
}


def get_config(slug: str) -> CompConfig:
    return REGISTRY.get(slug, DEFAULT)
