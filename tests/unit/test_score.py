from icilval.duel.score import (
    SCORE_EPSILON,
    axis_scores,
    crown_moves,
    paired_outcome,
    tally,
    verdict,
    void_fraction,
)
from icilval.spec import AXES


def unit(axis, k, c, void=False, i=0):
    return {
        "unit_id": f"{axis[:2]}-{i:03d}",
        "axis": axis,
        "king_success": k,
        "challenger_success": c,
        "outcome": paired_outcome(k, c),
        "void": void,
    }


def test_paired_outcome():
    assert paired_outcome(True, True) == "tie"
    assert paired_outcome(False, False) == "tie"
    assert paired_outcome(False, True) == "challenger"
    assert paired_outcome(True, False) == "king"
    assert paired_outcome(None, True) == "tie"


def test_axis_scores_exclude_void_and_unscored():
    units = [
        unit("spatial", True, False),
        unit("spatial", True, True, void=True),
        unit("spatial", None, True),
    ]
    k = axis_scores(units, "king")
    c = axis_scores(units, "challenger")
    assert k["spatial"] == 1.0 and c["spatial"] == 0.5
    assert k["environment"] is None
    assert k["average"] == 1.0 and c["average"] == 0.5


def test_crown_rule_boundary():
    # 0.65+0.60+0.65+0.50 = 2.40/4 = 0.60 ; +0.03 each -> 0.63 exactly at the margin
    assert crown_moves(0.60, 0.63, 3.0)
    assert not crown_moves(0.60, 0.63 - 1e-6, 3.0)
    assert crown_moves(0.5, 0.5, 0.0)
    assert not crown_moves(None, 0.9, 3.0)
    assert not crown_moves(0.9, None, 3.0)
    assert SCORE_EPSILON < 1e-6


def test_verdict_average_and_margin():
    units = []
    i = 0
    for axis, (kw, cw) in zip(AXES, [(6, 8), (7, 7), (5, 7), (3, 4)], strict=True):
        for n in range(10):
            units.append(unit(axis, n < kw, n < cw, i=i))
            i += 1
    v = verdict(units, 3.0)
    assert v.king_scores["average"] == (0.6 + 0.7 + 0.5 + 0.3) / 4
    assert v.challenger_scores["average"] == (0.8 + 0.7 + 0.7 + 0.4) / 4
    assert v.dethroned and v.reason == "margin-met"
    assert round(v.delta_points, 6) == 12.5
    assert (
        v.tally.wins == 2 + 0 + 2 + 1
        and v.tally.losses == 0
        and v.tally.decided == 5
        and v.tally.ties == 35
    )


def test_copy_of_king_never_moves_crown():
    units = [unit(a, n % 2 == 0, n % 2 == 0, i=n) for a in AXES for n in range(6)]
    v = verdict(units, 3.0)
    assert v.delta_points == 0.0 and not v.dethroned and v.reason == "short-of-margin"
    assert v.tally.decided == 0
    v0 = verdict(units, 0.0)
    assert (
        v0.dethroned
    )  # margin 0 means >= ; the copy ties and moves the crown, which is why margin > 0


def test_verdict_edge_cases():
    assert verdict([], 3.0).reason == "no-units"
    v = verdict([unit("spatial", None, None)], 3.0)
    assert v.reason == "unscored" and not v.dethroned
    t = tally([unit("spatial", True, False, void=True), unit("spatial", False, True)])
    assert t.void == 1 and t.wins == 1 and t.decided == 1
    assert (
        void_fraction([unit("spatial", True, True, void=True), unit("spatial", True, True)]) == 0.5
    )
