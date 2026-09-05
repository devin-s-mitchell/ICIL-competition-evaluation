import json

import pytest

from icilval.spec import AXES, load_spec_file, validate_spec


def test_spec_loads_and_fingerprints(spec):
    assert spec.version == 1
    assert spec.track_id == "icil_1demo"
    assert spec.axes == AXES
    assert spec.units_per_duel("smoke") == 4 * spec.units_per_axis("smoke")
    assert spec.size_of("bogus") == spec.default_size
    assert len(spec.fingerprint) == 64
    assert 0 <= spec.score_margin <= 100


def test_validate_rejects_bad_specs(spec, tmp_path):
    doc = json.loads(json.dumps(spec.raw))
    doc["duel"]["score_margin"] = 101
    assert any("score_margin" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    del doc["axes"]["object"]
    assert any("axes keys" in e for e in validate_spec(doc))
    doc = json.loads(json.dumps(spec.raw))
    doc["duel"]["default_size"] = "gigantic"
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_spec_file(p)
