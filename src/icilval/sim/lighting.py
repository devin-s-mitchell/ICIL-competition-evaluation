"""Lighting draws for the environment axis. Pure: sampling only, no simulator.

The draw is part of the unit definition, so it is made at derivation time from
the unit's own generator and recorded verbatim in the published verdict.
`apply_lighting` in `sim/perturb.py` maps the record onto a MuJoCo model.
"""

from __future__ import annotations

from typing import Any

from ..rng import HashRng


def sample_lighting(rng: HashRng, cfg: dict[str, Any], n_lights: int = 4) -> dict[str, Any]:
    lo_d, hi_d = cfg["diffuse_scale"]
    lo_a, hi_a = cfg["ambient_add"]
    lo_s, hi_s = cfg["specular_scale"]
    lo_h, hi_h = cfg["headlight_scale"]
    lights = []
    for _ in range(n_lights):
        lights.append(
            {
                "active": not rng.chance(float(cfg["light_drop_prob"])),
                "diffuse_scale": round(rng.uniform(lo_d, hi_d), 4),
                "ambient_add": round(rng.uniform(lo_a, hi_a), 4),
                "specular_scale": round(rng.uniform(lo_s, hi_s), 4),
                "pos_jitter": [
                    round(rng.uniform(-1, 1) * float(cfg["pos_jitter_m"]), 4) for _ in range(3)
                ],
                "dir_jitter": [
                    round(rng.uniform(-1, 1) * float(cfg["dir_jitter_rad"]), 4) for _ in range(3)
                ],
            }
        )
    if all(not light["active"] for light in lights):
        lights[0]["active"] = True
    return {"lights": lights, "headlight_scale": round(rng.uniform(lo_h, hi_h), 4)}
