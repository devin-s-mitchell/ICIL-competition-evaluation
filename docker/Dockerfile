# icilval model-side image: BPP's environment + LIBERO + icilval. Model sides run here with
# `--network none`; the CLIP ViT-B/16 weights BPP's encoder constructs from are baked in.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive MAMBA_ROOT_PREFIX=/opt/micromamba PATH=/opt/micromamba/envs/bp/bin:/opt/micromamba/bin:$PATH \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    LIBERO_CONFIG_PATH=/etc/libero HF_HOME=/opt/hf
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates git bzip2 build-essential cmake linux-libc-dev \
      libegl1 libegl1-mesa-dev libgl1 libgl1-mesa-dev libglvnd-dev libglib2.0-0 libosmesa6 libglew-dev libxrender1 libxext6 libsm6 patchelf \
    && rm -rf /var/lib/apt/lists/* \
    && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local bin/micromamba \
    && /usr/local/bin/micromamba --version
COPY vendor/behavior_prompting /opt/bpp
WORKDIR /opt/bpp
RUN /usr/local/bin/micromamba create -y -n bp -f environment.yml \
    && /usr/local/bin/micromamba clean -a -y
# CLIP backbone weights for construction (overwritten by the submission's weights at load)
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "import timm; timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True, global_pool='', num_classes=0); print('clip cached')" \
    && chmod -R a+rX /opt/hf
COPY docker/libero_config.yaml /etc/libero/config.yaml
COPY dist/*.whl /tmp/
RUN pip install --no-deps /tmp/*.whl && pip install "pynacl>=1.5" "safetensors>=0.4" "imageio-ffmpeg>=0.5" "huggingface_hub>=0.25" && rm /tmp/*.whl
RUN useradd -m -u 1000 runner && mkdir -p /model /pool /arch /run && chown -R runner /run
USER runner
COPY --chown=runner docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["icilval", "--help"]
