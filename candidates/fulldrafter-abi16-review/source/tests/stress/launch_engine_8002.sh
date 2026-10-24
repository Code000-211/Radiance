#!/usr/bin/env bash
set -euo pipefail

WORKTREE="/home/orca/RadianceMine/worktrees/tap_record"
BUILD_DIR="${WORKTREE}/build_hip"
BIN_PATH="${BUILD_DIR}/bin/radiance"
PORT="8002"
LOG_FILE="/home/orca/.gemini/antigravity-cli/scratch/engine_8002.log"
PID_FILE="/home/orca/.gemini/antigravity-cli/scratch/engine_8002.pid"

SIDECAR="/home/orca/drafter_data/checkpoints/flash_next_k12_exp/drafter_step_92500.safetensors"
if [ ! -f "${SIDECAR}" ]; then
  SIDECAR="/home/orca/drafter_data/checkpoints/flash_next_k12_exp/drafter_final.safetensors"
fi

MODEL_PATH="/root/models/qwen3.8-next-flash-fp8-iq4r-moe/qwen3.8-next-flash-fp8-iq4r-moe.rad"
RIDGEFILL_PROJECTOR="/root/models/qwen3.8-next-flash-fp8-iq4r-moe/ridgefill"

export HIP_VISIBLE_DEVICES="0,1"
export RADIANCE_HOME="${BUILD_DIR}/radiance_home"
export ROCM_PATH="/opt/rocm"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/core-10.0/lib:${LD_LIBRARY_PATH:-}"
export GPU_HEADROOM_MIB=384
export VRAM_KV_MIB=1400
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-12}"
export RADIANCE_FULLDRAFTER="${RADIANCE_FULLDRAFTER:-1}"
export RADIANCE_TEACHER_MATRICES="${RADIANCE_TEACHER_MATRICES:-/home/orca/drafter_data/teacher/teacher-matrices.safetensors}"
if [ "${NUM_SPECULATIVE_TOKENS}" -eq 0 ] || [ "${RADIANCE_FULLDRAFTER}" -eq 0 ]; then
  unset RADIANCE_FULLDRAFTER_SIDECAR || true
  export RADIANCE_FULLDRAFTER=0
  export NUM_SPECULATIVE_TOKENS=0
  export RADIANCE_GDN_ANCHOR=0
else
  export RADIANCE_FULLDRAFTER_SIDECAR="${RADIANCE_FULLDRAFTER_SIDECAR:-${SIDECAR}}"
  export RADIANCE_GDN_ANCHOR="${RADIANCE_GDN_ANCHOR:-1}"
fi
export RADIANCE_DARTREE_TOPOLOGY="${RADIANCE_DARTREE_TOPOLOGY:-linear}"

export RADIANCE_RIDGEFILL="${RADIANCE_RIDGEFILL:-speed}"
export RADIANCE_RIDGEFILL_PROJECTOR="${RIDGEFILL_PROJECTOR:-/root/models/qwen3.8-next-flash-fp8-iq4r-moe/ridgefill}"
export RADIANCE_RIDGEFILL_TAIL=512
export RADIANCE_RIDGEFILL_FINAL=off
export RADIANCE_RIDGEFILL_STAGE=auto
export RADIANCE_RIDGEFILL_FORCE_STREAM=1
export RADIANCE_PREFILL_BUFFERS=2
export RADIANCE_RIDGEFILL_SHARE=0.25
export RADIANCE_RIDGEFILL_ALPHA=1.0
export RADIANCE_ADAPTIVE_SPEC="${RADIANCE_ADAPTIVE_SPEC:-1}"
export RADIANCE_HEAT_RESIDENCY=0
export GPU_KEEP_ALIVE=0

WRAPPER=""
if [ -x "/usr/local/bin/with_render" ]; then
  WRAPPER="/usr/local/bin/with_render"
fi

echo "Launching Radiance engine directly on port ${PORT}..."
echo "BUILD_DIR: ${BUILD_DIR}"
echo "BINARY: ${BIN_PATH}"
echo "MODEL: ${MODEL_PATH}"
echo "SIDECAR: ${RADIANCE_FULLDRAFTER_SIDECAR:-none}"
echo "TOPOLOGY: ${RADIANCE_DARTREE_TOPOLOGY}"
echo "VRAM_KV_MIB: ${VRAM_KV_MIB}"
echo "GPU_HEADROOM_MIB: ${GPU_HEADROOM_MIB}"

nohup setsid ${WRAPPER} "${BIN_PATH}" \
  --model "${MODEL_PATH}" \
  --port "${PORT}" \
  --tp 2 \
  --tp-wire wht6 \
  --max-num-seqs 8 \
  --max-model-len 65536 \
  --placement expert_tiered \
  --ngram-placement auto \
  --host-pool-mib 12288 \
  --gpu-headroom-mib "${GPU_HEADROOM_MIB}" \
  --checkpoint-slots 0 \
  --vram-kv-mib "${VRAM_KV_MIB}" \
  --kv-cache-dtype fp8 \
  --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS}" \
  $( [ "${RADIANCE_ADAPTIVE_SPEC}" != "0" ] && echo "--adaptive-spec" ) \
  --prefix-cache-host-mib 8192 \
  --prefix-cache-disk-mib 0 \
  --prefix-aware-prefetch \
  --no-ced-prefetch \
  --ridgefill speed \
  --ridgefill-projector "${RIDGEFILL_PROJECTOR}" \
  --reasoning-effort none \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --repetition-penalty 1.0 \
  --max-num-batched-tokens 2048 \
  --checkpoint-interval 2048 \
  --mm-max-patches 0 \
  > "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${PID_FILE}"
echo "Engine process launched (PID ${PID}). Log: ${LOG_FILE}"

echo "Waiting for Radiance engine to report healthy on port ${PORT}..."
READY=0
for i in $(seq 1 360); do
  if ! kill -0 "${PID}" 2>/dev/null && ! pgrep -f "radiance.*--port ${PORT}" >/dev/null 2>&1; then
    echo "❌ Radiance engine process exited prematurely."
    exit 1
  fi
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    echo "✓ Radiance engine is HEALTHY on port ${PORT} after ${i}s!"
    break
  fi
  sleep 1
done

if [ "${READY}" -ne 1 ]; then
  echo "❌ Timeout waiting for Radiance engine on port ${PORT}."
  exit 1
fi

exit 0
