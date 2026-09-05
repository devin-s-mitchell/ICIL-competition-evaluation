from icilval.ids import ModelRef, duel_id, event_id, model_key, unit_id, unit_seed
from icilval.rng import HashRng


def test_ids_are_stable():
    ref = ModelRef.make("owner/model", "0123456789abcdef0123456789abcdef01234567")
    assert ref.key == model_key("owner/model", "0123456789abcdef0123456789abcdef01234567")
    assert len(ref.key) == 16
    king = ModelRef.make("org/king", "abcdef0123456789abcdef0123456789abcdef01")
    did = duel_id(1, "icil_1demo", ref, king)
    assert len(did) == 64 and did == duel_id(1, "icil_1demo", ref, king)
    assert did != duel_id(2, "icil_1demo", ref, king)
    assert did != duel_id(1, "icil_1demo", ref, None)
    assert unit_seed(did, "spatial", 0) != unit_seed(did, "spatial", 1)
    assert 0 <= unit_seed(did, "object", 3) < 2**32
    assert unit_id("composition", 7) == "co-007"
    assert event_id("duel", "icil_1demo", 4, did) != event_id("duel", "icil_1demo", 5, did)


def test_golden_values():
    # Frozen: a change here changes every published id.
    assert model_key("a/b", "c") == "fdd11077ff89f6bf"
    assert unit_seed("d", "spatial", 0) == 1941450640


def test_hash_rng_determinism_and_range():
    a, b = HashRng("x", 1), HashRng("x", 1)
    assert [a.below(10) for _ in range(20)] == [b.below(10) for _ in range(20)]
    c = HashRng("x", 2)
    assert [a.below(10) for _ in range(20)] != [c.below(10) for _ in range(20)]
    r = HashRng("s")
    assert all(0 <= r.below(7) < 7 for _ in range(500))
    assert all(0.0 <= r.uniform() < 1.0 for _ in range(500))
    perm = HashRng("p").shuffled(list(range(50)))
    assert sorted(perm) == list(range(50)) and perm != list(range(50))
