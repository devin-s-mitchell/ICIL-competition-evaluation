# Submissions

A submission is a **Hugging Face model repository at a pinned revision** holding:

- `model.safetensors` — the weights of a `bpp_libero_v1` policy (see `arch/`).
- `config.yaml` — produced by `icilval convert-ckpt`; its `model` block must equal the
  architecture template except for `model.mutable_keys` from `spec.json`.
- optionally `README.md`, `.json`, `.txt`, `.yaml` files.

Anything else (in particular `.ckpt` / pickles) is ignored on download and rejected if it is the
only weights file. No participant code runs anywhere.

## Converting a BPP training checkpoint

```bash
icilval convert-ckpt --ckpt path/to/your.ckpt --out ./submission
huggingface-cli upload <owner>/<name> ./submission
```

Then queue `owner/name@<commit sha>` through the organizer form or `icilval queue add`.

## What the validator checks

1. Allow-listed file extensions and total size (`model.max_repo_bytes`).
2. `config.yaml` parses with `yaml.safe_load`; `architecture == spec.model.architecture`;
   every `_target_` appears in the template; every other value equals the template
   except `mutable_keys`.
3. `model.safetensors` header: exactly the template's tensor names and shapes; dtypes in
   `model.allowed_dtypes`; parameter count ≤ `model.max_params`.
4. Weights load strictly into the template architecture inside a container with no network.

The check is itemised in the duel log and in the failure message the organizer form shows.
