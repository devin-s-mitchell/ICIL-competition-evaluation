import pytest

from icilval.ids import ModelRef, duel_id
from icilval.pools.schema import Pool, PoolTask, PoolVariant
from icilval.pools.units import derive_units
from icilval.spec import AXES


def make_pool(n_tasks=3, n_variants=2, n_init=5, n_demos=4):
    tasks, variants, axes = {}, {}, {a: {"eligible": []} for a in AXES}
    for axis in AXES:
        for t in range(n_tasks):
            tid = f"{axis}/task{t}"
            demos = [f"{tid}/demo_{d:02d}" for d in range(n_demos)]
            tasks[tid] = PoolTask(
                task_id=tid,
                axis=axis,
                suite="s",
                bddl=f"bddl/{axis}_{t}.bddl",
                language=f"do task {t}",
                init=f"init/{axis}_{t}.npz",
                n_init=n_init,
                goal=[["On", "a", "b"]],
                demos=demos,
                max_steps=300,
                demo_init_index={demos[0]: 0},
                perturbation={"kind": "chain"} if axis == "composition" else {},
            )
            if axis in ("spatial", "environment"):
                for v in range(n_variants):
                    vid = f"{tid}#v{v}"
                    variants[vid] = PoolVariant(
                        variant_id=vid,
                        axis=axis,
                        base_task=tid,
                        kind="pro_swap" if axis == "spatial" else "table",
                        params={"level": f"L{v + 1}"}
                        if axis == "spatial"
                        else {"table": "kitchen_table"},
                        bddl=f"bddl/{axis}_{t}_v{v}.bddl",
                        init=f"init/{axis}_{t}_v{v}.npz",
                        n_init=n_init,
                    )
                    axes[axis]["eligible"].append(vid)
            else:
                axes[axis]["eligible"].append(tid)
    pool = Pool(
        schema=1,
        pool_version="test",
        spec_version=1,
        sources={},
        tasks=tasks,
        variants=variants,
        axes=axes,
    )
    pool.seal()
    return pool


def test_pool_roundtrip(tmp_path):
    pool = make_pool()
    pool.save(tmp_path)
    again = Pool.load(tmp_path)
    assert again.pool_id == pool.pool_id
    assert again.to_dict() == pool.to_dict()
    (tmp_path / "pool.json").write_text(
        (tmp_path / "pool.json").read_text().replace('"L1"', '"L9"')
    )
    with pytest.raises(ValueError):
        Pool.load(tmp_path)


def test_units_deterministic_and_stratified(spec):
    pool = make_pool()
    did = duel_id(1, spec.track_id, ModelRef.make("a/b", "1" * 40), ModelRef.make("c/d", "2" * 40))
    units = derive_units(pool, spec, did, "heavy")
    again = derive_units(pool, spec, did, "heavy")
    assert [u.as_dict() for u in units] == [u.as_dict() for u in again]
    per = spec.units_per_axis("heavy")
    assert len(units) == 4 * per
    for axis in AXES:
        axis_units = [u for u in units if u.axis == axis]
        assert len(axis_units) == per
        assert [u.index for u in axis_units] == list(range(per))
        counts = {}
        for u in axis_units:
            key = u.variant or u.task
            counts[key] = counts.get(key, 0) + 1
        assert max(counts.values()) - min(counts.values()) <= 1
    ids = [u.unit_id for u in units]
    assert len(set(ids)) == len(ids)
    other = derive_units(
        pool, spec, duel_id(1, spec.track_id, ModelRef.make("a/b", "3" * 40), None), "heavy"
    )
    assert [u.seed for u in other] != [u.seed for u in units]


def test_units_prompt_disjoint_and_perturbation(spec):
    pool = make_pool()
    did = duel_id(1, spec.track_id, ModelRef.make("a/b", "1" * 40), None)
    for u in derive_units(pool, spec, did, "heavy"):
        task = pool.tasks[u.task]
        assert task.demo_init_index.get(u.demo) != u.instance
        assert 0 <= u.instance < 5
        assert u.perturbation["kind"]
        assert u.max_steps <= spec.max_steps(u.axis)
        if u.axis == "environment":
            assert "lighting" in u.perturbation and len(u.perturbation["lighting"]["lights"]) == 4
            assert any(light["active"] for light in u.perturbation["lighting"]["lights"])
        if u.axis == "spatial":
            assert u.perturbation["kind"] == "pro_swap" and u.variant
        if u.axis == "object":
            assert u.perturbation["kind"] == "object_swap" and u.variant is None
        if u.axis == "composition":
            assert u.perturbation["kind"] == "chain"
