"""BPP training checkpoint (torch+dill pickle) -> model.safetensors + config.yaml.

Organizer and entrant tool. Runs where behavior_prompting is importable. This is the only
place a pickle is ever opened, and only on files the caller owns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

SPLIT_INFO_KEY = "_extra_training_split_info"

# BPP's package was called `umi_day` when some public checkpoints were trained; the classes
# are the same, only the import path moved. Applied to every `_target_` in the model config.
TARGET_RENAMES = {"umi_day.": "behavior_prompting."}

# Training-only settings neutralised for inference. `pretrained` stays True: BPP's encoder only
# builds through timm's pretrained path (its own init path rejects the patch-embed conv), so the
# CLIP ViT-B/16 weights must be in the local timm/HF cache; they are overwritten by the strict
# load right after construction. Train-time image transforms are unused in eval mode.
INFERENCE_OVERRIDES: dict[str, Any] = {
    "obs_encoder.obs_encoder.obs_encoder.train_image_transforms": None,
    "obs_encoder.obs_encoder.obs_encoder.non_prompt_train_image_transforms": None,
    "obs_encoder.obs_encoder.use_pool_modality_pos_embed": False,
}


def _rename_targets(node: Any) -> Any:
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "_target_" and isinstance(v, str):
                for old, new in TARGET_RENAMES.items():
                    if v.startswith(old):
                        v = new + v[len(old) :]
            out[k] = _rename_targets(v)
        return out
    if isinstance(node, list):
        return [_rename_targets(v) for v in node]
    return node


def _set_path(d: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _resolve_model_cfg(cfg: Any) -> tuple[dict[str, Any], dict[str, Any], int]:
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("hydra"):
        OmegaConf.register_new_resolver("hydra", lambda *a: "hydra")
    model = _rename_targets(OmegaConf.to_container(cfg.model, resolve=True))
    shape_meta = OmegaConf.to_container(
        cfg.shape_meta if "shape_meta" in cfg else cfg.task.shape_meta, resolve=True
    )
    exec_horizon = int(OmegaConf.select(cfg, "task.env_runner.exec_action_horizon", default=12))
    for path, value in INFERENCE_OVERRIDES.items():
        _set_path(model, path, value)
    return model, shape_meta, exec_horizon


def convert_checkpoint(
    ckpt_path: str | Path,
    out_dir: str | Path,
    *,
    arch_name: str,
    emit_arch: str | Path | None = None,
) -> dict[str, Any]:
    import dill
    import torch
    from safetensors.torch import save_file

    ckpt_path = Path(ckpt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, "rb") as fh:
        payload = torch.load(fh, pickle_module=dill, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    state = payload["state_dicts"]["model"]
    split_info = state.pop(SPLIT_INFO_KEY, None)
    tensors = {k: v.detach().contiguous().cpu() for k, v in state.items() if hasattr(v, "detach")}
    non_tensors = sorted(k for k in state if not hasattr(state[k], "detach"))
    model_cfg, shape_meta, exec_horizon = _resolve_model_cfg(cfg)

    metadata = {"format": "pt", "architecture": arch_name, "source": ckpt_path.name}
    save_file(tensors, str(out_dir / "model.safetensors"), metadata=metadata)
    config = {
        "architecture": arch_name,
        "model": model_cfg,
        "shape_meta": shape_meta,
        "exec_action_horizon": exec_horizon,
        "source": {
            "checkpoint": ckpt_path.name,
            "training_split_info": bool(split_info),
            "dropped_keys": non_tensors,
        },
    }
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        k: {"shape": list(v.shape), "dtype": _dtype_name(v)} for k, v in sorted(tensors.items())
    }
    param_count = sum(int(v.numel()) for v in tensors.values())
    if emit_arch:
        arch_dir = Path(emit_arch)
        arch_dir.mkdir(parents=True, exist_ok=True)
        (arch_dir / f"{arch_name}.cfg.json").write_text(
            json.dumps(
                {
                    "architecture": arch_name,
                    "model": model_cfg,
                    "shape_meta": shape_meta,
                    "exec_action_horizon": exec_horizon,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (arch_dir / f"{arch_name}.tensors.json").write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n"
        )
    return {
        "tensors": len(tensors),
        "param_count": param_count,
        "dropped_keys": non_tensors,
        "exec_action_horizon": exec_horizon,
    }


def _dtype_name(t: Any) -> str:
    import torch

    return {
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.bool: "BOOL",
    }.get(t.dtype, str(t.dtype))
