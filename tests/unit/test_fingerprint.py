import json
import struct

import yaml

from icilval.model.fingerprint import (
    check_submission,
    diff_model_cfg,
    read_safetensors_header,
    targets_in,
)

TEMPLATE_MODEL = {
    "_target_": "pkg.Policy",
    "name": "prompt_diffusion_unet",
    "num_inference_steps": 16,
    "obs_encoder": {
        "_target_": "pkg.Encoder",
        "n_emb": 768,
        "obs_encoder": {"_target_": "pkg.Tokenizer", "pretrained": False},
    },
}


def write_safetensors(path, tensors):
    header = {k: {"dtype": d, "shape": s, "data_offsets": [0, 0]} for k, (s, d) in tensors.items()}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def make_arch(tmp_path):
    arch = tmp_path / "arch"
    arch.mkdir()
    (arch / "bpp_libero_v1.cfg.json").write_text(
        json.dumps({"architecture": "bpp_libero_v1", "model": TEMPLATE_MODEL})
    )
    (arch / "bpp_libero_v1.tensors.json").write_text(
        json.dumps(
            {"a.weight": {"shape": [4, 3], "dtype": "F32"}, "b": {"shape": [2], "dtype": "F32"}}
        )
    )
    return arch


def make_submission(tmp_path, model=None, tensors=None, extra=None):
    d = tmp_path / "sub"
    d.mkdir(exist_ok=True)
    (d / "config.yaml").write_text(
        yaml.safe_dump({"architecture": "bpp_libero_v1", "model": model or TEMPLATE_MODEL})
    )
    write_safetensors(
        d / "model.safetensors", tensors or {"a.weight": ([4, 3], "F32"), "b": ([2], "F32")}
    )
    for name, content in (extra or {}).items():
        (d / name).write_bytes(content)
    return d


def test_diff_and_targets():
    changed = json.loads(json.dumps(TEMPLATE_MODEL))
    changed["name"] = "mine"
    changed["obs_encoder"]["n_emb"] = 512
    changed["obs_encoder"]["obs_encoder"]["_target_"] = "evil.Loader"
    diffs = diff_model_cfg(changed, TEMPLATE_MODEL, ["name"])
    assert any(d.startswith("obs_encoder.n_emb") for d in diffs)
    assert any("_target_" in d for d in diffs)
    assert not any(d.startswith("name") for d in diffs)
    assert "evil.Loader" in targets_in(changed)


def test_header_and_accept(spec, tmp_path):
    arch = make_arch(tmp_path)
    sub = make_submission(tmp_path)
    assert read_safetensors_header(sub / "model.safetensors")["a.weight"]["shape"] == [4, 3]
    report = check_submission(sub, spec, arch)
    assert report.ok, report.errors
    assert report.param_count == 14 and report.model_sha256 and report.config_sha256


def test_rejections(spec, tmp_path):
    arch = make_arch(tmp_path)
    m = json.loads(json.dumps(TEMPLATE_MODEL))
    m["obs_encoder"]["obs_encoder"]["_target_"] = "evil.Loader"
    r = check_submission(make_submission(tmp_path, model=m), spec, arch)
    assert any("allow-listed" in e for e in r.errors)

    r = check_submission(
        make_submission(tmp_path, tensors={"a.weight": ([4, 4], "F32"), "b": ([2], "F32")}),
        spec,
        arch,
    )
    assert any("shape" in e for e in r.errors)
    r = check_submission(
        make_submission(tmp_path, tensors={"a.weight": ([4, 3], "F64"), "b": ([2], "F32")}),
        spec,
        arch,
    )
    assert any("dtype" in e for e in r.errors)
    r = check_submission(
        make_submission(tmp_path, tensors={"a.weight": ([4, 3], "F32")}), spec, arch
    )
    assert any("missing" in e for e in r.errors)
    r = check_submission(make_submission(tmp_path, extra={"weights.ckpt": b"\x80\x04"}), spec, arch)
    assert any("extension" in e for e in r.errors)
    (tmp_path / "sub" / "weights.ckpt").unlink()
    (tmp_path / "sub" / "config.yaml").write_text("architecture: other\nmodel: {}\n")
    r = check_submission(tmp_path / "sub", spec, arch)
    assert any("architecture" in e for e in r.errors)
    (tmp_path / "sub" / "config.yaml").unlink()
    r = check_submission(tmp_path / "sub", spec, arch)
    assert any("config.yaml missing" in e for e in r.errors)
