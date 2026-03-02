"""Pipeline registry.

This repository started as a single-competition template.
We extend it with per-competition baselines, validators, and prompt bundles.

Each pipeline is keyed by Kaggle competition slug (case-insensitive).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineSpec:
    # Canonical ID (mostly for humans/docs)
    key: str

    # Kaggle competition slug (used for `kaggle competitions submit -c ...`)
    competition: str

    # llm-puzzles comp_registry slug used to format the submission
    # (often identical to `competition`, but kept separate for flexibility)
    format_slug: str

    baseline_solver: Path
    validator: Path

    # Optional prompt bundle for AgentLaboratory generation
    prompt_file: Optional[Path] = None
    custom_prompts_file: Optional[Path] = None

    # CSV column candidates to extract the puzzle state
    # (first existing one wins)
    state_columns: List[str] = None  # type: ignore[assignment]

    # A small vector to use for smoke-validation of the solver
    smoke_vector: List[int] = None  # type: ignore[assignment]


def _norm(s: str) -> str:
    return s.strip().lower()


def _p(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


# Canonical pipeline definitions
_PIPELINES: List[PipelineSpec] = [
    PipelineSpec(
        key="lrx-sort",
        competition="lrx-sort",  # not a real Kaggle slug; kept for backwards-compat demos
        format_slug="format/moves-dot",
        baseline_solver=_p("competitions", "lrx-sort", "solve_module.py"),
        validator=_p("competitions", "lrx-sort", "validate_solve_output.py"),
        prompt_file=_p("competitions", "lrx-sort", "prompts", "example_user_prompt.txt"),
        custom_prompts_file=_p("competitions", "lrx-sort", "prompts", "custom_prompts_template.json"),
        state_columns=["vector"],
        smoke_vector=[3, 1, 2, 5, 4],
    ),
PipelineSpec(
        key="lrx-discover-math-gods-algorithm",
        competition="lrx-discover-math-gods-algorithm",
        format_slug="lrx-discover-math-gods-algorithm",
        baseline_solver=_p("competitions", "lrx-discover-math-gods-algorithm", "solve_module.py"),
        validator=_p("competitions", "lrx-discover-math-gods-algorithm", "validate_solve_output.py"),
        prompt_file=_p("competitions", "lrx-discover-math-gods-algorithm", "prompts", "user_prompt.txt"),
        custom_prompts_file=_p("competitions", "lrx-discover-math-gods-algorithm", "prompts", "custom_prompts_template.json"),
        state_columns=["permutation"],
        smoke_vector=[7, 4, 0, 5, 3, 6, 2, 1],
    ),
    PipelineSpec(
        key="lrx-oeis-a-186783-brainstorm-math-conjecture",
        competition="lrx-oeis-a-186783-brainstorm-math-conjecture",
        format_slug="lrx-oeis-a-186783-brainstorm-math-conjecture",
        baseline_solver=_p("competitions", "lrx-oeis-a-186783-brainstorm-math-conjecture", "solve_module.py"),
        validator=_p("competitions", "lrx-oeis-a-186783-brainstorm-math-conjecture", "validate_solve_output.py"),
        prompt_file=_p("competitions", "lrx-oeis-a-186783-brainstorm-math-conjecture", "prompts", "user_prompt.txt"),
        custom_prompts_file=_p("competitions", "lrx-oeis-a-186783-brainstorm-math-conjecture", "prompts", "custom_prompts_template.json"),
        state_columns=["permutation"],
        smoke_vector=[1, 0, 4, 3, 2],
    ),
        PipelineSpec(
        key="cayleypy-rapapport-m2",
        competition="cayleypy-rapapport-m2",
        format_slug="cayleypy-rapapport-m2",
        baseline_solver=_p("competitions", "cayleypy-rapapport-m2", "solve_module.py"),
        validator=_p("competitions", "cayleypy-rapapport-m2", "validate_solve_output.py"),
        prompt_file=_p("competitions", "cayleypy-rapapport-m2", "prompts", "user_prompt.txt"),
        custom_prompts_file=_p("competitions", "cayleypy-rapapport-m2", "prompts", "custom_prompts.json"),
        state_columns=["permutation", "initial_state", "vector"],
        smoke_vector=[3, 0, 1, 4, 2],
    ),
    PipelineSpec(
        key="cayleypy-pancake",
        competition="CayleyPy-pancake",
        format_slug="CayleyPy-pancake",
        baseline_solver=_p("competitions", "cayleypy-pancake", "solve_module.py"),
        validator=_p("competitions", "cayleypy-pancake", "validate_solve_output.py"),
        prompt_file=_p("competitions", "cayleypy-pancake", "prompts", "user_prompt.txt"),
        custom_prompts_file=_p("competitions", "cayleypy-pancake", "prompts", "custom_prompts.json"),
        state_columns=["initial_state", "permutation", "vector"],
        smoke_vector=[3, 1, 2, 0],
    ),
    PipelineSpec(
        key="cayleypy-glushkov",
        competition="cayleypy-glushkov",
        format_slug="cayleypy-glushkov",
        baseline_solver=_p("competitions", "cayleypy-glushkov", "solve_module.py"),
        validator=_p("competitions", "cayleypy-glushkov", "validate_solve_output.py"),
        prompt_file=_p("competitions", "cayleypy-glushkov", "prompts", "user_prompt.txt"),
        state_columns=["initial_state", "permutation", "vector"],
        smoke_vector=[3, 0, 2, 1],
    ),
    PipelineSpec(
        key="cayleypy-reversals",
        competition="cayleypy-reversals",
        format_slug="cayleypy-reversals",
        baseline_solver=_p("competitions", "cayleypy-reversals", "solve_module.py"),
        validator=_p("competitions", "cayleypy-reversals", "validate_solve_output.py"),
        prompt_file=_p("competitions", "cayleypy-reversals", "prompts", "user_prompt.txt"),
        state_columns=["initial_state", "permutation", "vector"],
        smoke_vector=[3, 0, 2, 1],
    ),
    # Template-only / placeholder pipelines (baseline returns UNSOLVED)
    PipelineSpec(
        key="cayleypy-transposons",
        competition="cayleypy-transposons",
        format_slug="cayleypy-transposons",
        baseline_solver=_p("competitions", "cayleypy-transposons", "solve_module.py"),
        validator=_p("competitions", "cayleypy-transposons", "validate_solve_output.py"),
        prompt_file=_p("competitions", "cayleypy-transposons", "prompts", "user_prompt.txt"),
        custom_prompts_file=_p(
            "competitions",
            "cayleypy-transposons",
            "prompts",
            "custom_prompts_template.json",
        ),
        state_columns=["initial_state", "permutation", "vector"],
        smoke_vector=[0, 1, 2, 3],
    ),
    PipelineSpec(
        key="cayleypy-christophers-jewel",
        competition="cayleypy-christophers-jewel",
        format_slug="cayleypy-christophers-jewel",
        baseline_solver=_p("competitions", "cayleypy-christophers-jewel", "solve_module.py"),
        validator=_p("competitions", "cayleypy-christophers-jewel", "validate_solve_output.py"),
        prompt_file=_p(
            "competitions",
            "cayleypy-christophers-jewel",
            "prompts",
            "user_prompt.txt",
        ),
        custom_prompts_file=_p(
            "competitions",
            "cayleypy-christophers-jewel",
            "prompts",
            "custom_prompts_template.json",
        ),
        state_columns=["initial_state", "permutation", "vector"],
        smoke_vector=[0, 1, 2, 3],
    ),
    PipelineSpec(
        key="cayley-py-444-cube",
        competition="cayley-py-444-cube",
        format_slug="cayley-py-444-cube",
        baseline_solver=_p("competitions", "cayley-py-444-cube", "solve_module.py"),
        validator=_p("competitions", "cayley-py-444-cube", "validate_solve_output.py"),
        prompt_file=_p("competitions", "cayley-py-444-cube", "prompts", "user_prompt.txt"),
        custom_prompts_file=_p(
            "competitions",
            "cayley-py-444-cube",
            "prompts",
            "custom_prompts_template.json",
        ),
        state_columns=["initial_state", "state", "permutation", "vector"],
        smoke_vector=[0, 1, 2, 3],
    ),
    PipelineSpec(
        key="cayley-py-professor-tetraminx-solve-optimally",
        competition="cayley-py-professor-tetraminx-solve-optimally",
        format_slug="cayley-py-professor-tetraminx-solve-optimally",
        baseline_solver=_p("competitions", "cayley-py-professor-tetraminx-solve-optimally", "solve_module.py"),
        validator=_p("competitions", "cayley-py-professor-tetraminx-solve-optimally", "validate_solve_output.py"),
        prompt_file=_p(
            "competitions",
            "cayley-py-professor-tetraminx-solve-optimally",
            "prompts",
            "user_prompt.txt",
        ),
        custom_prompts_file=_p(
            "competitions",
            "cayley-py-professor-tetraminx-solve-optimally",
            "prompts",
            "custom_prompts_template.json",
        ),
        state_columns=["initial_state", "state", "permutation", "vector"],
        smoke_vector=[0, 1, 2, 3],
    ),
    PipelineSpec(
        key="cayley-py-megaminx",
        competition="cayley-py-megaminx",
        format_slug="cayley-py-megaminx",
        baseline_solver=_p("competitions", "cayley-py-megaminx", "solve_module.py"),
        validator=_p("competitions", "cayley-py-megaminx", "validate_solve_output.py"),
        prompt_file=_p("competitions", "cayley-py-megaminx", "prompts", "user_prompt.txt"),
        custom_prompts_file=_p(
            "competitions",
            "cayley-py-megaminx",
            "prompts",
            "custom_prompts_template.json",
        ),
        state_columns=["initial_state", "state", "permutation", "vector"],
        smoke_vector=[0, 1, 2, 3],
    ),
    PipelineSpec(
        key="cayleypy-ihes-cube",
        competition="cayleypy-ihes-cube",
        format_slug="cayleypy-ihes-cube",
        baseline_solver=_p("competitions", "cayleypy-ihes-cube", "solve_module.py"),
        validator=_p("competitions", "cayleypy-ihes-cube", "validate_solve_output.py"),
        prompt_file=_p("competitions", "cayleypy-ihes-cube", "prompts", "user_prompt.txt"),
        custom_prompts_file=_p(
            "competitions",
            "cayleypy-ihes-cube",
            "prompts",
            "custom_prompts_template.json",
        ),
        state_columns=["initial_state", "state", "permutation", "vector"],
        smoke_vector=[0, 1, 2, 3],
    ),
]


# Build lookup with aliases
_REGISTRY: Dict[str, PipelineSpec] = {}
for spec in _PIPELINES:
    _REGISTRY[_norm(spec.key)] = spec
    _REGISTRY[_norm(spec.competition)] = spec

    # Common alias: lowercase competition slug
    _REGISTRY[_norm(spec.competition.lower())] = spec


def get_pipeline(competition_or_key: str) -> Optional[PipelineSpec]:
    return _REGISTRY.get(_norm(competition_or_key))


def list_pipelines() -> List[PipelineSpec]:
    # return unique canonical specs (by key)
    seen = set()
    out: List[PipelineSpec] = []
    for spec in _PIPELINES:
        if spec.key not in seen:
            out.append(spec)
            seen.add(spec.key)
    return out
