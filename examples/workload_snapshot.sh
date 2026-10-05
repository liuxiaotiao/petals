#!/usr/bin/env bash
# Record everything needed to rerun a workload experiment: where every layer is served, the
# exact flags each server runs with, software and GPUs per node, the code, and the prompts.
# Run it from the repo root on the control node while the swarm is up -- ideally right
# before or right after the run, with no restart in between (peer IDs change on restart).
#
#   examples/workload_snapshot.sh ~/wl-results/2026-10-05/r4_2048_norobots/settings
#
# Layout of OUT_DIR:
#   config/   hosts.txt (placement as configured), petals-env.sh and the effective env
#   cluster/  peers.txt (node -> peer ID -> layers, live), status.txt, nodes.txt (per node:
#             GPU, driver, torch/petals versions, server.env, the running server command)
#   code/     the scripts and client code as they are now, with MD5SUMS
#   prompts/  task/workload/*.jsonl with MD5SUMS and per-file dataset counts
# Anything that looks like a token or password is replaced by <redacted>.
set -u

out="${1:?usage: $(basename "$0") OUT_DIR}"
[[ -f examples/qwen_cluster.sh ]] || { echo "run this from the repo root" >&2; exit 2; }
HOSTS_FILE="${HOSTS_FILE:-task/hosts.txt}"
ENV_FILE="${ENV_FILE:-$HOME/petals-env.sh}"
SSH="${SSH:-ssh}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_PORT="${SSH_PORT:-22}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
REMOTE_DIR="${REMOTE_DIR:-petals-qwen}"
REDACT='s/^((export[[:space:]]+)?[A-Za-z_]*(TOKEN|SECRET|PASSWORD|PASSWD)[A-Za-z_]*=).*/\1<redacted>/'

mkdir -p "$out"/{config,cluster,code,prompts}
date -Is > "$out/SNAPSHOT_TIME"

# --- config -----------------------------------------------------------------
cp "$HOSTS_FILE" "$out/config/hosts.txt"
[[ -r "$ENV_FILE" ]] && sed -E "$REDACT" "$ENV_FILE" > "$out/config/petals-env.sh"
env | grep -E '^(MODEL_NAME|MODEL_REVISION|DHT_PREFIX|INFERENCE_MAX_LENGTH|ATTN_CACHE_TOKENS|MAX_BATCH_SIZE|TORCH_DTYPE|PETALS_[A-Z_]*|HF_HUB_DISABLE_XET|HF_ENDPOINT|CLIENT_NODE|BOOTSTRAP_NODE)=' \
  | sort > "$out/config/effective-env.txt"

# --- cluster ----------------------------------------------------------------
echo "querying the swarm ..."
bash examples/qwen_cluster.sh peers > "$out/cluster/peers.txt" 2>&1
bash examples/qwen_cluster.sh status > "$out/cluster/status.txt" 2>&1

echo "querying every node (GPU, versions, server flags) ..."
tmp=$(mktemp -d)
ids=()
while read -r id addr _; do
  [[ -z "$id" ]] && continue
  ids+=("$id")
  # The pgrep pattern is bracketed so it cannot match this shell's own command line.
  $SSH -n $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@${addr%%:*}" "
echo \"host:     \$(hostname)\"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/gpu:      /'
\$HOME/$REMOTE_DIR/venv/bin/python -c 'import torch, petals, transformers, hivemind; print(\"versions: torch\", torch.__version__, \"cuda\", torch.version.cuda, \"petals\", petals.__version__, \"transformers\", transformers.__version__, \"hivemind\", hivemind.__version__)' 2>/dev/null
echo 'server.env:'
sed -E '$REDACT' \$HOME/$REMOTE_DIR/run/server.env 2>/dev/null | sed 's/^/  /'
echo 'server command:'
pgrep -af 'petals[.]cli[.]run_server' | head -1 | sed 's/^[0-9]* /  /'
" > "$tmp/$id" 2>&1 &
done < <(sed 's/#.*//' "$HOSTS_FILE")
wait
for id in "${ids[@]}"; do
  echo "== $id"
  cat "$tmp/$id"
  echo
done > "$out/cluster/nodes.txt"
rm -rf "$tmp"

# --- code -------------------------------------------------------------------
for f in examples/workload_bench.py examples/workload_sample.py examples/workload_split.py \
         examples/workload_run.sh examples/workload_snapshot.sh examples/cluster_gpumon.sh \
         examples/qwen_cluster.sh examples/run_qwen_server.sh src/petals/client/inference_session.py; do
  [[ -f "$f" ]] && cp "$f" "$out/code/"
done
(cd "$out/code" && md5sum -- * > MD5SUMS)
if git rev-parse --git-dir >/dev/null 2>&1; then
  git rev-parse HEAD > "$out/code/GIT_HEAD"
  git status --short > "$out/code/GIT_STATUS"
fi

# --- prompts ----------------------------------------------------------------
cp task/workload/*.jsonl "$out/prompts/" 2>/dev/null
if ls "$out"/prompts/*.jsonl >/dev/null 2>&1; then
  (cd "$out/prompts" && md5sum -- *.jsonl > MD5SUMS)
  python3 - "$out/prompts" > "$out/prompts/COUNTS.txt" <<'EOF'
import collections, glob, json, os, sys
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.jsonl"))):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    counts = collections.Counter(row["dataset"] for row in rows)
    print(f"{os.path.basename(path):28}{len(rows):5}  " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
EOF
fi

# --- summary ----------------------------------------------------------------
answered=$(grep -c '^host:' "$out/cluster/nodes.txt")
echo "snapshot written to $out"
echo "  nodes answered: $answered of ${#ids[@]}; peers listed: $(grep -c '12D3KooW' "$out/cluster/peers.txt")"
echo "  prompt files: $(ls "$out"/prompts/*.jsonl 2>/dev/null | wc -l); code files: $(ls "$out/code" | wc -l)"
(( answered == ${#ids[@]} )) || echo "  WARNING: some nodes did not answer; see cluster/nodes.txt" >&2
exit 0
