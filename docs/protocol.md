# Protocol

The competition scores a Behavior Prompting Policy (BPP) checkpoint on one track: the model is shown
**one demonstration** of a task and **no language**, then must perform the task under four kinds of
change. Every constant below comes from `spec.json`; nothing here is normative on its own.

## Axes

| Axis | What changes between the demonstration and the scored episode | Source of variants |
|---|---|---|
| spatial | Object (and robot) start positions. | LIBERO-PRO `swap` scenes and its initial-pose deltas, plus a level ladder L1..L5 (radius / minimum displacement from `axes.spatial.levels`) applied at reset. |
| environment | The table and room the scene sits in, and the lights. | LIBERO's kitchen / living-room / study tables (initial states regenerated) and per-unit light draws applied to the MuJoCo model. |
| object | The target object is swapped for another one affording the same operation; the demonstration shows the new pairing. | LIBERO-Gen novel pairings (spatial combinations, first-step novelties) and organizer-generated tasks from `affordance.yaml`. |
| composition | Two single-step tasks in sequence; the demonstration is one continuous recording of both. | LIBERO-Gen chains and LIBERO-10. |

A **unit** is one axis + one task (or variant) + one initial state + one prompt demonstration + one seed.
Both sides of a duel run the identical unit list. The demonstration never starts from the scored
initial state.

## Scoring

Per axis, `success rate = successes / scored units` (void units excluded). The **final score** is the
mean of the four axis rates. Scores are stored as fractions in `[0, 1]`.

## Crown rule

The challenger takes the crown iff `challenger.average >= king.average + duel.score_margin / 100`.
Paired per-unit outcomes (challenger / king / tie) are published as a diagnostic and never decide the
crown. A copy of the reigning model scores identically, so it never clears the margin.

## Determinism

`duel_id = sha256(spec_version|track|challenger.key|challenger.revision|king.key|king.revision)`.
Unit lists are derived from the pool and the duel id with a sha256-counter generator, so a third
party holding the pool can regenerate them from the published record. Per-episode seeds are
`int(sha256(duel_id|axis|index)[:8], 16)`; the diffusion sampler is seeded with them.

## What is published

For every duel: the signed index record, the full event (every unit with both outcomes, perturbation
details, prompt length) and three clips per unit: the prompt demonstration, the reigning model's
rollout and the challenger's rollout. See `store-schema.json`.
