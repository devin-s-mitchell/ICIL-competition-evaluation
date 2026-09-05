"""Scoring and the crown rule. Pure functions over unit verdict dicts.

Scores are fractions in [0, 1]. `score_margin` arrives in percentage points and
is divided by 100 here and nowhere else.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..spec import AXES

SIDES = ("challenger", "king")
SCORE_EPSILON = 1e-9


def paired_outcome(king_success: bool | None, challenger_success: bool | None) -> str:
    if king_success is None or challenger_success is None:
        return "tie"
    if king_success == challenger_success:
        return "tie"
    return "challenger" if challenger_success else "king"


def side_success(unit: dict[str, Any], side: str) -> bool | None:
    value = unit.get(f"{side}_success")
    return value if isinstance(value, bool) else None


def axis_rate(units: Iterable[dict[str, Any]], side: str, axis: str) -> float | None:
    scored = 0
    successes = 0
    for u in units:
        if u.get("axis") != axis or u.get("void"):
            continue
        s = side_success(u, side)
        if s is None:
            continue
        scored += 1
        successes += int(s)
    return successes / scored if scored else None


def average(per_axis: dict[str, float | None]) -> float | None:
    present = [per_axis[a] for a in AXES if per_axis.get(a) is not None]
    return sum(present) / len(present) if present else None


def axis_scores(units: Iterable[dict[str, Any]], side: str) -> dict[str, float | None]:
    units = list(units)
    per = {a: axis_rate(units, side, a) for a in AXES}
    per["average"] = average(per)
    return per


def empty_scores() -> dict[str, float | None]:
    return {**{a: None for a in AXES}, "average": None}


def crown_moves(
    king_average: float | None, challenger_average: float | None, margin_points: float
) -> bool:
    if king_average is None or challenger_average is None:
        return False
    return challenger_average >= king_average + margin_points / 100.0 - SCORE_EPSILON


@dataclass
class Tally:
    units: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    decided: int = 0
    void: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "decided": self.decided,
            "void": self.void,
        }


def tally(units: Iterable[dict[str, Any]]) -> Tally:
    t = Tally()
    for u in units:
        t.units += 1
        if u.get("void"):
            t.void += 1
            continue
        outcome = u.get("outcome") or paired_outcome(
            side_success(u, "king"), side_success(u, "challenger")
        )
        if outcome == "challenger":
            t.wins += 1
            t.decided += 1
        elif outcome == "king":
            t.losses += 1
            t.decided += 1
        else:
            t.ties += 1
    return t


@dataclass
class Verdict:
    king_scores: dict[str, float | None]
    challenger_scores: dict[str, float | None]
    score_margin: float
    dethroned: bool
    reason: str
    tally: Tally = field(default_factory=Tally)

    @property
    def delta_points(self) -> float | None:
        k, c = self.king_scores["average"], self.challenger_scores["average"]
        return None if k is None or c is None else (c - k) * 100.0


def verdict(units: Iterable[dict[str, Any]], score_margin: float) -> Verdict:
    units = list(units)
    king = axis_scores(units, "king")
    challenger = axis_scores(units, "challenger")
    moves = crown_moves(king["average"], challenger["average"], score_margin)
    if not units:
        reason = "no-units"
    elif king["average"] is None or challenger["average"] is None:
        reason = "unscored"
    else:
        reason = "margin-met" if moves else "short-of-margin"
    return Verdict(king, challenger, score_margin, moves, reason, tally(units))


def void_fraction(units: Iterable[dict[str, Any]]) -> float:
    units = list(units)
    return sum(1 for u in units if u.get("void")) / len(units) if units else 0.0
