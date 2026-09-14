#!/usr/bin/env bash
# Run from the repository root on every GPU host. See docs/qwen3.6-deployment.md.
#
# By default each server sizes itself from the GPU it sees and claims the thinnest
# part of the swarm, so every host runs this same command and only ANNOUNCE_IP differs.
set -euo pipefail

: "${BOOTSTRAP_PEER:?Set BOOTSTRAP_PEER to the private DHT node multiaddress}"
model="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
port="${PORT:-31330}"
python_bin="${PYTHON:-python}"

args=(
  "$model" --port "$port"
  --initial_peers "$BOOTSTRAP_PEER"
  --torch_dtype "${TORCH_DTYPE:-float16}" --quant_type none --device "${DEVICE:-cuda:0}" --inference_only
  --inference_max_length 2048 --attn_cache_tokens 4096
  --max_batch_size 256 --max_chunk_size_bytes 16777216
  --num_handlers 2 --no_auto_relay
)

# Layer placement, in order of precedence:
#   BLOCKS=start:end  pin an exact range (disables rebalancing for this server)
#   NUM_BLOCKS=N      serve N layers, but let the swarm decide which ones
#   neither           size from free VRAM and let the swarm decide which ones
if [[ -n "${BLOCKS:-}" ]]; then
  args+=(--block_indices "$BLOCKS")
elif [[ -n "${NUM_BLOCKS:-}" ]]; then
  args+=(--num_blocks "$NUM_BLOCKS")
fi

# Downloading from the Hub pulls whole shard files, including layers this node does not
# serve, and rebalancing adds more over time. Cap the cache so it cannot grow without bound.
if [[ -n "${MAX_DISK_SPACE:-}" ]]; then args+=(--max_disk_space "$MAX_DISK_SPACE"); fi
if [[ -n "${CACHE_DIR:-}" ]]; then args+=(--cache_dir "$CACHE_DIR"); fi
# Unset, this keeps the upstream default of 0.75. Use 0 to freeze placement after startup.
if [[ -n "${BALANCE_QUALITY:-}" ]]; then args+=(--balance_quality "$BALANCE_QUALITY"); fi
# Left unset, the server and the client derive the same prefix from the repo name.
if [[ -n "${DHT_PREFIX:-}" ]]; then args+=(--dht_prefix "$DHT_PREFIX"); fi
if [[ -n "${MODEL_REVISION:-}" ]]; then args+=(--revision "$MODEL_REVISION"); fi
# Required whenever peers are not on one flat network: the address others must dial.
if [[ -n "${ANNOUNCE_IP:-}" ]]; then args+=(--public_ip "$ANNOUNCE_IP"); fi

exec "$python_bin" -m petals.cli.run_server "${args[@]}"
