# syntax=docker/dockerfile:1.19

ARG BASE_IMAGE

FROM ghcr.io/astral-sh/uv:0.9.25@sha256:13e233d08517abdafac4ead26c16d881cd77504a2c40c38c905cf3a0d70131a6 AS uv
FROM ${BASE_IMAGE} AS cap-system

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg libglib2.0-0 tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CAP_ROOT=/workspace/isaaclab_arena \
    CAP_SYSTEM_PYTHON=/isaac-sim/kit/python/bin/python3 \
    ARENA_GAP_ROOT=/workspace/isaaclab_arena/isaaclab_arena_gap \
    GAP_ROOT=/workspace/isaaclab_arena/external/gap/graph-as-policy \
    SKILLS_ROOT=/workspace/isaaclab_arena/external/gap/open-robot-skills \
    ARENA_SOURCE_ROOT=/workspace/isaaclab_arena/external/IsaacLab-Arena \
    AUTOLAB_VABAR_ROOT=/workspace/isaaclab_arena/external/gap/autolab-vabar \
    AUTOLAB_VABAR_VENV=/workspace/isaaclab_arena/external/gap/autolab-vabar/.venv \
    ARENA_ENV_ROOT=/workspace/isaaclab_arena \
    ROBOT_DESCRIPTIONS_CACHE=/opt/cap/robot-descriptions \
    HF_HOME=/opt/cap/models/huggingface \
    HF_HUB_CACHE=/opt/cap/models/huggingface/hub \
    CAP_IMAGE_MANIFEST=/opt/cap/image-manifest.json \
    CAP_IMAGE_PROVENANCE=/opt/cap/provenance \
    UV_CONCURRENT_DOWNLOADS=4 \
    UV_HTTP_TIMEOUT=3600 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

FROM cap-system AS environment-builder

COPY --from=uv /uv /uvx /usr/local/bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential cmake git git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${CAP_ROOT}
# Keep the expensive environment closure independent of ordinary repository
# source. Only immutable external pins, installation contracts, and dependency
# manifests are inputs to this stage.
COPY external/ ${CAP_ROOT}/external/
# AUTOLab v1.0's committed sources, which docker/build.sh exports in either layout: the
# external/gap/autolab-vabar gitlink, or the VABAR repository this tree is a folder of.
COPY --from=autolab . ${AUTOLAB_VABAR_ROOT}/
COPY isaaclab_arena_gap/scripts/bootstrap.sh ${ARENA_GAP_ROOT}/scripts/bootstrap.sh
COPY isaaclab_arena_gap/scripts/skill_bundles.sh ${ARENA_GAP_ROOT}/scripts/skill_bundles.sh
COPY isaaclab_arena_gap/scripts/gitlinks.sh ${ARENA_GAP_ROOT}/scripts/gitlinks.sh
COPY isaaclab_arena_gap/scripts/autolab_layout.sh ${ARENA_GAP_ROOT}/scripts/autolab_layout.sh
COPY isaaclab_arena_gap/scripts/arena_environment_consistency.sh \
    ${ARENA_GAP_ROOT}/scripts/arena_environment_consistency.sh
COPY isaaclab_arena_gap/pyproject.toml /opt/cap/overlay-manifests/isaaclab_arena_gap.toml
COPY isaaclab_arena_gap/pyproject.toml ${ARENA_GAP_ROOT}/pyproject.toml
COPY isaaclab_arena_vabar/pyproject.toml \
    /opt/cap/overlay-manifests/isaaclab_arena_vabar.toml
COPY docker/install_overlay_dependencies.py \
    ${CAP_ROOT}/docker/install_overlay_dependencies.py

RUN --mount=type=cache,target=/opt/uv-cache,sharing=locked \
    --mount=type=secret,id=github_token,required=true \
    github_auth="$(printf 'x-access-token:%s' "$(cat /run/secrets/github_token)" | base64 -w0)" \
    && export GIT_CONFIG_COUNT=1 \
       GIT_CONFIG_KEY_0=http.https://github.com/.extraheader \
       GIT_CONFIG_VALUE_0="Authorization: Basic ${github_auth}" \
       UV_CACHE_DIR=/opt/uv-cache \
    && ./isaaclab_arena_gap/scripts/bootstrap.sh --materialized-source --external-only \
    && "${GAP_ROOT}/.venv/bin/python" -c \
      'from robot_descriptions.loaders.yourdfpy import load_robot_description; load_robot_description("panda_description")' \
    && "${GAP_ROOT}/.venv/bin/python" -c 'import gap' \
    && "${ARENA_ENV_ROOT}/.venv/bin/python" -c 'import isaaclab' \
    && "${CAP_SYSTEM_PYTHON}" \
      ./docker/install_overlay_dependencies.py \
      --python "${ARENA_ENV_ROOT}/.venv/bin/python" \
      "${ARENA_SOURCE_ROOT}/pyproject.toml" \
      /opt/cap/overlay-manifests/isaaclab_arena_gap.toml \
      /opt/cap/overlay-manifests/isaaclab_arena_vabar.toml \
    && uv pip install --offline --no-deps --no-build-isolation \
      --python "${ARENA_ENV_ROOT}/.venv/bin/python" \
      --editable "${ARENA_SOURCE_ROOT}" \
    && "${ARENA_ENV_ROOT}/.venv/bin/python" -c 'import isaaclab_arena'

FROM environment-builder AS model-builder

COPY docker/bake_models.py ${CAP_ROOT}/docker/bake_models.py
COPY docker/runtime_dependencies.json \
    ${CAP_ROOT}/docker/runtime_dependencies.json
RUN --mount=type=secret,id=hf_token,required=true \
    HF_TOKEN="$(cat /run/secrets/hf_token)" \
    "${SKILLS_ROOT}/tools/grounding-dino/.venv/bin/python" \
      ./docker/bake_models.py \
      --manifest ./docker/runtime_dependencies.json \
      --cache-dir "${HF_HUB_CACHE}" \
      --notice-dir /opt/cap/model-notices \
    && test ! -e "${HF_HOME}/token" \
    && test ! -e "${HF_HOME}/stored_tokens"

FROM model-builder AS manifest-builder

COPY docker/compose_image_manifest.py \
    ${CAP_ROOT}/docker/compose_image_manifest.py
COPY --from=source-metadata source-manifest.json /opt/cap/provenance/source-manifest.json
COPY --from=source-metadata dirty.patch /opt/cap/provenance/dirty.patch
ARG BASE_IMAGE
RUN "${CAP_SYSTEM_PYTHON}" ./docker/compose_image_manifest.py \
      --source-root "${CAP_ROOT}" \
      --source-manifest /opt/cap/provenance/source-manifest.json \
      --model-manifest ./docker/runtime_dependencies.json \
      --base-image "${BASE_IMAGE}" \
      --output /opt/cap/image-manifest.json

FROM model-builder AS runtime-payload-builder

COPY docker/deduplicate_payloads.py /opt/cap/deduplicate_payloads.py
RUN "${CAP_SYSTEM_PYTHON}" /opt/cap/deduplicate_payloads.py \
      "${GAP_ROOT}/.venv/lib/python3.11/site-packages/torch" \
      "${GAP_ROOT}/.venv/lib/python3.11/site-packages/nvidia" \
      "${GAP_ROOT}/.venv/lib/python3.11/site-packages/triton" \
      "${SKILLS_ROOT}/tools/curobo/.venv/lib/python3.11/site-packages/torch" \
      "${SKILLS_ROOT}/tools/curobo/.venv/lib/python3.11/site-packages/nvidia" \
      "${SKILLS_ROOT}/tools/curobo/.venv/lib/python3.11/site-packages/triton" \
      "${SKILLS_ROOT}/tools/grounding-dino/.venv/lib/python3.11/site-packages/torch" \
      "${SKILLS_ROOT}/tools/grounding-dino/.venv/lib/python3.11/site-packages/nvidia" \
      "${SKILLS_ROOT}/tools/grounding-dino/.venv/lib/python3.11/site-packages/triton" \
      "${SKILLS_ROOT}/tools/sam3/.venv/lib/python3.11/site-packages/torch" \
      "${SKILLS_ROOT}/tools/sam3/.venv/lib/python3.11/site-packages/nvidia" \
      "${SKILLS_ROOT}/tools/sam3/.venv/lib/python3.11/site-packages/triton"

FROM cap-system AS runtime

ENV CAP_BAKED_RUNTIME=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    UV_OFFLINE=1 \
    UV_LOCKED=1 \
    UV_NO_SYNC=1 \
    HUB__ARGS__DETECT_ONLY=false \
    HOME=/home/cap \
    XDG_CACHE_HOME=/cache/xdg \
    PATH=/workspace/isaaclab_arena/.venv/bin:/usr/local/bin:/usr/bin:/bin

WORKDIR ${CAP_ROOT}
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=environment-builder /opt/uv/python/ /opt/uv/python/
# One layer preserves cross-environment hardlinks and includes the immutable
# source used by the editable GaP/tool environments.
COPY --chmod=go+rX --from=runtime-payload-builder ${CAP_ROOT}/external/ ${CAP_ROOT}/external/
COPY --from=environment-builder ${CAP_ROOT}/.venv/ ${CAP_ROOT}/.venv/
COPY --from=model-builder /opt/cap/models/ /opt/cap/models/
COPY --from=model-builder /opt/cap/model-notices/ /opt/cap/model-notices/
COPY --from=environment-builder /opt/cap/robot-descriptions/ /opt/cap/robot-descriptions/

# Runtime tools are independent of source and provenance, so ordinary adapter
# edits do not repeat package-index work.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git zstd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Keep the 4.8 GB asset payload independent of mutable benchmark Python and
# configuration. The second copy overlays everything except those payload
# directories, preserving the complete package without copying the assets twice.
COPY --chmod=go+rX --exclude=*.py --exclude=README.md --exclude=registrations \
    isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets/ \
    ${CAP_ROOT}/isaaclab_arena_vabar/src/isaaclab_arena_vabar/assets/
COPY --chmod=go+rX --exclude=src/isaaclab_arena_vabar/assets/vabar__* \
    --exclude=src/isaaclab_arena_vabar/assets/vabar_* \
    isaaclab_arena_vabar/ ${CAP_ROOT}/isaaclab_arena_vabar/
COPY --chmod=go+rX --from=manifest-builder /opt/cap/provenance/ /opt/cap/provenance/
COPY --chmod=go+rX --from=manifest-builder /opt/cap/image-manifest.json /opt/cap/image-manifest.json
COPY --chmod=go+rX isaaclab_arena_gap/ ${ARENA_GAP_ROOT}/
COPY --chmod=go+rX pyproject.toml ${CAP_ROOT}/
# Runtime helpers from the root docker/ folder: source verification, model
# verification and the entrypoint.
COPY --chmod=go+rX docker/entrypoint.sh docker/source_manifest.py docker/bake_models.py \
    docker/runtime_dependencies.json ${CAP_ROOT}/docker/

ARG CAP_SOURCE_REVISION=unknown
ARG CAP_SOURCE_FINGERPRINT=unknown
ARG CAP_SOURCE_DIRTY=unknown
LABEL org.opencontainers.image.revision="${CAP_SOURCE_REVISION}" \
      vabar.source-fingerprint="${CAP_SOURCE_FINGERPRINT}" \
      vabar.source-dirty="${CAP_SOURCE_DIRTY}"
ENV CAP_IMAGE_SOURCE_REVISION="${CAP_SOURCE_REVISION}" \
    CAP_IMAGE_SOURCE_FINGERPRINT="${CAP_SOURCE_FINGERPRINT}" \
    CAP_IMAGE_SOURCE_DIRTY="${CAP_SOURCE_DIRTY}"

RUN "${CAP_SYSTEM_PYTHON}" ./docker/source_manifest.py verify \
      --source-root "${CAP_ROOT}" \
      --manifest /opt/cap/provenance/source-manifest.json \
    && UV_CACHE_DIR=/tmp/cap-build-uv-cache uv pip install --offline --no-deps --no-build-isolation \
      --python "${ARENA_ENV_ROOT}/.venv/bin/python" \
      --editable "${CAP_ROOT}/isaaclab_arena_vabar" \
      --editable "${ARENA_GAP_ROOT}" \
    && rm -rf /tmp/cap-build-uv-cache \
    && source ./isaaclab_arena_gap/scripts/arena_environment_consistency.sh \
    && if check_arena_environment_consistency "${ARENA_ENV_ROOT}/.venv/bin/python"; then \
         :; \
       elif [[ "$?" == "${ARENA_PIP_CHECK_KNOWN_CONFLICTS}" ]]; then \
         printf 'Arena environment carries only the 5 pinned Arena source conflicts.\n'; \
       else \
         printf 'Arena environment consistency check failed:\n%s\n' \
           "${ARENA_PIP_CHECK_OUTPUT}" >&2; \
         exit 1; \
       fi \
    && /usr/sbin/groupadd --non-unique --gid 1000 cap \
    && /usr/sbin/useradd --non-unique --uid 1000 --gid 1000 --create-home \
      --home-dir /home/cap --shell /bin/bash cap \
    && mkdir -p /cache/kit /cache/xdg /cache/ov /cache/nvidia/ComputeCache \
      /cache/nvidia/GLCache /home/cap/.cache /outputs /var/cache/hub \
    && ln -s /cache/kit \
      "${ARENA_ENV_ROOT}/.venv/lib/python3.12/site-packages/isaacsim/kit/cache" \
    && ln -s /cache/nvidia /home/cap/.nv \
    && ln -s /cache/ov /home/cap/.cache/ov \
    && ln -s "${SKILLS_ROOT}/tools/vlm/.venv" \
      "${ARENA_GAP_ROOT}/skill_overrides/tools/vlm/.venv" \
    && chown -R cap:cap /home/cap /cache /outputs /var/cache/hub \
    && PYTHONPATH="${ARENA_GAP_ROOT}/src" \
      "${GAP_ROOT}/.venv/bin/python" -c 'import gap, isaaclab_arena_gap' \
    && "${ARENA_ENV_ROOT}/.venv/bin/python" -c \
      'import isaaclab_arena_gap.gap_remote_policy, isaaclab, isaaclab_arena, msgpack_numpy' \
    && "${SKILLS_ROOT}/tools/grounding-dino/.venv/bin/python" \
      ./docker/bake_models.py \
      --manifest ./docker/runtime_dependencies.json \
      --cache-dir "${HF_HUB_CACHE}" \
      --notice-dir /opt/cap/model-notices \
      --offline-verify \
    && test ! -e "${HF_HOME}/token" \
    && test ! -e "${HF_HOME}/stored_tokens"

# The syringe documents name box-shifted workcells (scene_norobot_box*.xml). They used to
# be generated into the source tree the first time a variant was built; the repository now
# tracks all 58 under assets/vab_tool_sort/syringe/, so nothing is generated at build or
# run time. Check they arrived with the sources.
RUN cd "${AUTOLAB_VABAR_ROOT}" \
    && ls assets/vab_tool_sort/syringe/scene_norobot_box*.xml >/dev/null

USER cap
# Cache restoration clears these children as the runtime user, preserving /cache.
RUN find /cache -mindepth 1 -delete \
    && touch /cache/.write-test && rm /cache/.write-test
RUN PYTHONPATH="${ARENA_GAP_ROOT}/src" "${GAP_ROOT}/.venv/bin/python" -c 'import gap, isaaclab_arena_gap' \
    && "${ARENA_ENV_ROOT}/.venv/bin/python" -c \
      'import isaaclab_arena_gap.gap_remote_policy, isaaclab_arena_vabar, isaaclab_arena'
VOLUME ["/cache", "/outputs"]
HEALTHCHECK NONE
ENTRYPOINT ["/usr/bin/tini", "--", "/workspace/isaaclab_arena/docker/entrypoint.sh"]
CMD ["verify"]
