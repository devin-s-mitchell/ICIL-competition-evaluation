# Protocol

The competition scores a submission on one track: for every skill, the model is shown **one
demonstration** of a task and **no language**, then must perform the task under a change the
demonstration did not show. Every constant below comes from `spec.json`; nothing here is
normative on its own.

## Skills

A **skill** is what is scored. Each has its own simulator, its own Behavior Prompting Policy
architecture (one checkpoint per skill in a submission) and its own perturbations. A skill's score
is one success rate over its units; the **final score** is the mean over skills.

| Skill | What the model does | Simulator / architecture | Perturbations a unit carries |
|---|---|---|---|
| `pick_and_place` | Grasp one object and place it at a destination: BPP's LIBERO pick-and-place domain, one Grasp stage then one Place stage. Tasks with an open / close / turn-on / push stage or two placements are excluded (`skills.pick_and_place.task_filter`). | LIBERO (MuJoCo) / `bpp_libero_v1` | `spatial` (LIBERO-PRO swap and pose perturbations plus the displacement ladder L1..L5), `environment` (another LIBERO table and room, lights drawn per unit), `object` (LIBERO-Gen novel object pairings; the demonstration shows the new pairing) |
| `draw_anything` | Reproduce a drawing shown once: BPP's DrawAnything-Sim domain. The demonstration is a person drawing a shape on a square whiteboard; the model draws it again on a blank board. | DrawAnything-Sim (pygame/pymunk) / `bpp_draw_v1` | `rotation` (the board is turned to a different angle than the demonstration's, by at least `min_delta_rad`; the pen starts elsewhere) |

A **unit** is one skill + one perturbation group + one task (or variant) + one initial state + one
prompt demonstration + one seed. A skill's units are spread evenly over its perturbation groups
(`pools/units.py`, `group_sizes`), and within a group over the eligible tasks or variants. The
group and the perturbation's details are published on every unit and are never scored
separately. Both sides of a duel run the identical unit list. The demonstration never starts from
the scored initial state: on the drawing skill that means its board angle differs from the unit's
by at least `min_delta_rad`.

## Success

- `pick_and_place`: LIBERO's goal predicate holds at any step before the cap
  (`skills.pick_and_place.max_steps`, or the task's own smaller limit).
- `draw_anything`: the symmetric Chamfer distance, in canvas pixels, between the demonstrated
  strokes (turned upright) and the drawn strokes (turned upright by the unit's board angle) is at
  most `skills.draw_anything.success.threshold` at the best step of the episode. This is BPP's
  own reward, negated; the best value is published on the unit as `*_metric`. An episode also ends
  when the predicted pen positions stay within `idle_stop_px` for `idle_stop_steps` steps, as in
  BPP's runner. The threshold was calibrated on the genesis checkpoint (see `docs/pools.md`).

## Scoring

Per skill, `success rate = successes / scored units` (void units excluded). The **final score** is
the mean of the skill rates. Scores are stored as fractions in `[0, 1]`.

## Crown rule

The challenger takes the crown iff `challenger.average >= king.average + duel.score_margin / 100`.
Paired per-unit outcomes (challenger / king / tie) are published as a diagnostic and never decide
the crown. A copy of the reigning model scores identically, so it never clears the margin.

## Determinism

`duel_id = sha256(spec_version|track|challenger.key|challenger.revision|king.key|king.revision)`.
Unit lists are derived from the pool and the duel id with a sha256-counter generator, so a third
party holding the pool can regenerate them from the published record. Per-episode seeds are
`int(sha256(duel_id|skill|index)[:8], 16)`; the diffusion sampler is seeded with them. A drawing
instance's board angle and cursor start are a pure function of the task id and instance index
(`pools/units.py`, `draw_instance`), so the pool carries no initial-state files for that skill.

## What is published

For every duel: the signed index record, the full event (every unit with both outcomes, the
perturbation, the prompt length and, on the drawing skill, both sides' Chamfer distances) and three
clips per unit: the prompt demonstration, the reigning model's rollout and the challenger's
rollout. Drawing clips show the board with the demonstrated strokes overlaid in red and the
model's in blue. See `store-schema.json`.

## Adding a skill

A skill is an entry in `spec.json` `skills` (code, title, architecture, simulator, environment,
perturbations, success rule), an architecture template under `arch/`, a pool stage that imports
its tasks, a simulator wrapper and episode loop under `sim/`, and a policy under `model/`. Adding
one bumps `spec_version`, since it changes every duel id and every submission's layout.
