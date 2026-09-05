"""Pure geometry shared by unit derivation and the runtime perturbation."""

from __future__ import annotations

import math


def direction_delta(direction_index: int, directions: int, radius_m: float) -> list[float]:
    angle = 2.0 * math.pi * direction_index / directions
    return [round(radius_m * math.cos(angle), 4), round(radius_m * math.sin(angle), 4)]
