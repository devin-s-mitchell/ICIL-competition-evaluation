import pytest

from icilval.ids import ModelRef, duel_id
from icilval.pools.schema import Pool, PoolTask, PoolVariant
from icilval.pools.units import derive_units, draw_instance, group_sizes

PP, DA = "pick_and_place", "draw_anything"


def make_pool(n_tasks=3, n_variants=2, n_init=5, n_demos=4):
    tasks, variants = {}, {}
    skills = {
        PP: {g: {"eligible": []} for g in ("spatial", "environment", "object")},
        DA: {"rotation": {"eligible": []}},
    }
    for t in range(n_tasks):
        tid = f"libero_spatial/task{t}"
        demos = [f"{tid}/demo_{d:02d}" for d in range(n_demos)]
        tasks[tid] = PoolTask(
            task_id=tid,
            skill=PP,
            kind="base",
            suite="libero_spatial",
            bddl=f"bddl/{t}.bddl",
            language=f"pick task {t}",
            init=f"init/{t}.npz",
            n_init=n_init,
            goal=[["On", "a", "b"]],
            demos=demos,
            max_steps=300,
            demo_init_index={demos[0]: 0},
        )
        for kind, group in (
            ("level", "spatial"),
            ("pro_swap", "spatial"),
            ("table", "environment"),
            ("lighting", "environment"),
        ):
            for v in range(n_variants):
                vid = f"{tid}#{kind}{v}"
                params = (
                    {
                        "level": f"L{v + 1}",
                        "radius_m": 0.1,
                        "min_delta_m": 0.05,
                        "directions": 8,
                        "target": "a",
                        "yaw_max_rad": 0.5,
                    }
                    if kind == "level"
                    else {"table": "kitchen_table"}
                    if kind == "table"
                    else {}
                )
                variants[vid] = PoolVariant(
                    variant_id=vid,
                    skill=PP,
                    base_task=tid,
                    kind=kind,
                    params=params,
                    bddl=f"bddl/{t}_{kind}{v}.bddl",
                    init=f"init/{t}_{kind}{v}.npz",
                    n_init=n_init * (8 if kind == "level" else 1),
                )
                skills[PP][group]["eligible"].append(vid)
        oid = f"libero_gen/swap{t}"
        odemos = [f"{oid}/demo_{d:02d}" for d in range(n_demos)]
        tasks[oid] = PoolTask(
            task_id=oid,
            skill=PP,
            kind="object_swap",
            suite="gen",
            bddl=f"bddl/swap{t}.bddl",
            language=f"put thing {t}",
            init=f"init/swap{t}.npz",
            n_init=n_init,
            goal=[["On", "c", "d"]],
            demos=odemos,
            max_steps=400,
            perturbation={"kind": "object_swap", "object": "c", "target": "d"},
        )
        skills[PP]["object"]["eligible"].append(oid)
        did_ = f"drawanything_handmade/draw_{t}"
        ddemos = [f"{did_}/demo_{d:02d}" for d in range(n_demos)]
        tasks[did_] = PoolTask(
            task_id=did_,
            skill=DA,
            kind="drawing",
            suite="drawanything_handmade",
            language=f"draw {t}",
            n_init=n_init,
            demos=ddemos,
            max_steps=400,
            meta={"demo_angles": {d: (-0.7 + 0.4 * i) for i, d in enumerate(ddemos)}},
        )
        skills[DA]["rotation"]["eligible"].append(did_)
    pool = Pool(
        schema=2,
        pool_version="test",
        spec_version=2,
        sources={},
        tasks=tasks,
        variants=variants,
        skills=skills,
    )
    pool.seal()
    return pool


def test_pool_roundtrip(tmp_path):
    pool = make_pool()
    pool.save(tmp_path)
    again = Pool.load(tmp_path)
    assert again.pool_id == pool.pool_id
    assert again.to_dict() == pool.to_dict()
    assert again.tasks["drawanything_handmade/draw_0"].bddl is None
    (tmp_path / "pool.json").write_text(
        (tmp_path / "pool.json").read_text().replace('"L1"', '"L9"')
    )
    with pytest.raises(ValueError):
        Pool.load(tmp_path)


def test_group_sizes():
    assert group_sizes(3, ["a", "b", "c"]) == {"a": 1, "b": 1, "c": 1}
    assert group_sizes(4, ["a", "b", "c"]) == {"a": 2, "b": 1, "c": 1}
    assert group_sizes(42, ["a", "b", "c"]) == {"a": 14, "b": 14, "c": 14}
    assert group_sizes(2, ["a", "b", "c"]) == {"a": 1, "b": 1, "c": 0}


def test_units_deterministic_and_stratified(spec):
    pool = make_pool()
    did = duel_id(2, spec.track_id, ModelRef.make("a/b", "1" * 40), ModelRef.make("c/d", "2" * 40))
    units = derive_units(pool, spec, did, "heavy")
    again = derive_units(pool, spec, did, "heavy")
    assert [u.as_dict() for u in units] == [u.as_dict() for u in again]
    per = spec.units_per_skill("heavy")
    assert len(units) == 2 * per
    for skill in spec.skills:
        skill_units = [u for u in units if u.skill == skill]
        assert len(skill_units) == per
        assert [u.index for u in skill_units] == list(range(per))
        groups = list(spec.perturbations(skill))
        sizes = group_sizes(per, groups)
        for g in groups:
            g_units = [u for u in skill_units if u.kind == g]
            assert len(g_units) == sizes[g]
            counts = {}
            for u in g_units:
                key = u.variant or u.task
                counts[key] = counts.get(key, 0) + 1
            assert max(counts.values()) - min(counts.values()) <= 1
        assert all(u.unit_id.startswith(spec.skill_code(skill) + "-") for u in skill_units)
    ids = [u.unit_id for u in units]
    assert len(set(ids)) == len(ids)
    other = derive_units(
        pool, spec, duel_id(2, spec.track_id, ModelRef.make("a/b", "3" * 40), None), "heavy"
    )
    assert [u.seed for u in other] != [u.seed for u in units]


def test_units_prompt_disjoint_and_perturbation(spec):
    pool = make_pool()
    did = duel_id(2, spec.track_id, ModelRef.make("a/b", "1" * 40), None)
    for u in derive_units(pool, spec, did, "heavy"):
        task = pool.tasks[u.task]
        assert u.perturbation["kind"]
        assert u.max_steps <= spec.max_steps(u.skill)
        if u.skill == "pick_and_place":
            assert task.demo_init_index.get(u.demo) != u.instance
            assert 0 <= u.instance < 5
            assert u.bddl and u.init
        if u.kind == "environment":
            assert "lighting" in u.perturbation and len(u.perturbation["lighting"]["lights"]) == 4
            assert any(light["active"] for light in u.perturbation["lighting"]["lights"])
            assert u.perturbation["kind"] in ("table", "lighting")
        if u.kind == "spatial":
            assert u.perturbation["kind"] in ("pro_swap", "level") and u.variant
            if u.perturbation["kind"] == "level":
                assert "delta_xy" in u.perturbation and "directions" not in u.perturbation
        if u.kind == "object":
            assert u.perturbation["kind"] == "object_swap" and u.variant is None
        if u.kind == "rotation":
            p = u.perturbation
            cfg = spec.perturbation("draw_anything", "rotation")
            lo, hi = cfg["angle_range_rad"]
            assert lo <= p["angle_rad"] <= hi
            assert p["delta_rad"] >= cfg["min_delta_rad"] - 1e-9
            assert task.meta["demo_angles"][u.demo] == pytest.approx(p["demo_angle_rad"], abs=1e-5)
            c_lo, c_hi = spec.env("draw_anything")["cursor_start_range_px"]
            assert all(c_lo <= c <= c_hi for c in p["cursor_px"])
            assert u.bddl is None and u.init is None
            state = draw_instance(u.task, u.instance, cfg, spec.env("draw_anything"))
            assert state["angle_rad"] == p["angle_rad"] and state["cursor_px"] == p["cursor_px"]


def test_instances_roundtrip(tmp_path):
    pool = make_pool()
    vid = next(iter(pool.variants))
    pool.variants[vid].instances = [0, 2, 4]
    tid = next(iter(pool.tasks))
    pool.tasks[tid].instances = [1]
    pool.seal()
    pool.save(tmp_path)
    again = Pool.load(tmp_path)
    assert again.variants[vid].instances == [0, 2, 4] and again.variants[vid].valid_instances == [
        0,
        2,
        4,
    ]
    assert again.tasks[tid].instances == [1]
    assert again.pool_id == pool.pool_id


def test_old_pool_schema_is_refused(tmp_path):
    (tmp_path / "pool.json").write_text('{"schema": 1, "pool_version": "x", "spec_version": 1}')
    with pytest.raises(ValueError, match="upgrade"):
        Pool.load(tmp_path)
