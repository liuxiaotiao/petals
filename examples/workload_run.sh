#!/usr/bin/env bash
# Rerun the sequential workload experiment the way it was run on 2026-10-05:
# one client per chain, started on the chain's layer-0 node and pinned to that chain's six
# servers, one request at a time; GPUs sampled throughout; settings snapshotted first and a
# report written at the end. Everything lands in one directory.
#
#   source ~/petals-env.sh
#   bash examples/workload_run.sh --name r5_2048 --max-new-tokens 2048
#   bash examples/workload_run.sh --name smoke --max-new-tokens 64 -- --limit 3
#
# Arguments after "--" go to every workload_bench.py client unchanged.
# Run it inside tmux: a full run is 2-3 hours.
#
# Chains are "TAG:node,node,..." in layer order; the first node hosts layer 0 and runs the
# client. The default is the pinned 2026-10-05 layout (docs/experiments/2026-10-05-*.md).
# Prompts come from task/workload/chain.<TAG>.jsonl (examples/workload_split.py), which must
# be deployed: the run refuses to start if a client node holds a different copy.
#
# Refuses to start while another workload client is running anywhere it can see: two clients
# on one chain double the load and, sharing a log name, overwrite each other's records.
set -u

name="" max_new=2048 extra=()
CHAINS="${CHAINS:-A:N01,N05,N03,N07,N02,N06 B:N11,N08,N09,N13,N04,N10}"
PROMPT_DIR="${PROMPT_DIR:-task/workload}"
RESULTS="${RESULTS:-$HOME/wl-results}"
HOSTS_FILE="${HOSTS_FILE:-task/hosts.txt}"
SSH="${SSH:-ssh}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_PORT="${SSH_PORT:-22}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
REMOTE_DIR="${REMOTE_DIR:-petals-qwen}"

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --name) name="${2:-}"; shift 2 ;;
    --max-new-tokens) max_new="${2:-}"; shift 2 ;;
    --) shift; extra=("$@"); break ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
[[ -n "$name" ]] || usage
[[ -f examples/qwen_cluster.sh ]] || { echo "run this from the repo root" >&2; exit 2; }
[[ -n "${MODEL_NAME:-}" ]] || { echo "MODEL_NAME is not set: source ~/petals-env.sh first" >&2; exit 2; }
out="$RESULTS/$(date +%F)/$name"
[[ -e "$out" ]] && { echo "$out already exists; pick another --name" >&2; exit 2; }

declare -A IP
while read -r id addr _; do [[ -n "$id" ]] && IP[$id]="${addr%%:*}"; done < <(sed 's/#.*//' "$HOSTS_FILE")
rsh() { $SSH -n $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$1" "$2"; }

# --- preflight --------------------------------------------------------------
echo "preflight ..."
busy=""
pgrep -f 'workload_bench[.]py' >/dev/null && busy+=" control-node"
for spec in $CHAINS; do
  tag="${spec%%:*}" nodes="${spec#*:}"; first="${nodes%%,*}"
  [[ -n "${IP[$first]:-}" ]] || { echo "unknown node $first (not in $HOSTS_FILE)" >&2; exit 2; }
  rsh "${IP[$first]}" "pgrep -f 'workload_bench[.]py'" >/dev/null 2>&1 && busy+=" $first"
  file="$PROMPT_DIR/chain.$tag.jsonl"
  [[ -s "$file" ]] || { echo "missing $file: run examples/workload_split.py first" >&2; exit 2; }
  here=$(md5sum < "$file" | cut -c1-32)
  there=$(rsh "${IP[$first]}" "md5sum < $REMOTE_DIR/repo/$file" 2>/dev/null | cut -c1-32)
  [[ "$here" == "$there" ]] || { echo "$file on $first differs from this copy: run 'bash examples/qwen_cluster.sh deploy'" >&2; exit 2; }
done
[[ -z "$busy" ]] || { echo "a workload client is already running on:$busy -- stop it first" >&2; exit 1; }

peers=$(bash examples/qwen_cluster.sh peers 2>&1)
declare -A ALLOW CLIENT
for spec in $CHAINS; do
  tag="${spec%%:*}" nodes="${spec#*:}"
  CLIENT[$tag]="${nodes%%,*}"
  ids=""
  for node in ${nodes//,/ }; do
    peer=$(awk -v n="$node" '$1==n {print $2; exit}' <<<"$peers")
    [[ -n "$peer" ]] || { echo "no peer ID for $node: is its server up? ('qwen_cluster.sh peers')" >&2; exit 1; }
    ids+="$peer "
  done
  ALLOW[$tag]="$ids"
done

# --- run --------------------------------------------------------------------
mkdir -p "$out"
echo "recording settings into $out/settings ..."
bash examples/workload_snapshot.sh "$out/settings" > "$out/snapshot.log" 2>&1 \
  || echo "warning: snapshot incomplete, see $out/snapshot.log" >&2
{
  echo "name=$name"
  echo "max_new_tokens=$max_new"
  echo "chains=$CHAINS"
  echo "client_args=${extra[*]:-}"
  for tag in "${!ALLOW[@]}"; do echo "allowed_servers_$tag=${ALLOW[$tag]}"; done
  echo "started=$(date -Is)"
} > "$out/RUN.txt"

bash examples/cluster_gpumon.sh "$out/gpu.csv" 2> "$out/gpumon.log" &
gpu=$!
pids=()
for spec in $CHAINS; do
  tag="${spec%%:*}"
  # ALLOW is deliberately unquoted: each peer ID is its own argument.
  CLIENT_SCRIPT=workload_bench.py bash examples/qwen_cluster.sh client --node "${CLIENT[$tag]}" \
    --prompts "$PROMPT_DIR/chain.$tag.jsonl" --allowed-servers ${ALLOW[$tag]} --tag "$tag" \
    --max-new-tokens "$max_new" ${extra[@]+"${extra[@]}"} > "$out/wl.$tag.log" 2>&1 &
  pids+=($!)
  echo "chain $tag: client on ${CLIENT[$tag]}, $(wc -l < "$PROMPT_DIR/chain.$tag.jsonl") prompts, log $out/wl.$tag.log"
done
trap 'echo "interrupted; stopping clients and gpumon" >&2; kill "${pids[@]}" "$gpu" 2>/dev/null; exit 130' INT TERM
echo "running; progress: grep -c '^REC' $out/wl.*.log"
wait "${pids[@]}"
kill "$gpu" 2>/dev/null; wait "$gpu" 2>/dev/null
echo "finished=$(date -Is)" >> "$out/RUN.txt"

# --- report -----------------------------------------------------------------
gpu_arg=()
[[ -s "$out/gpu.csv" ]] && gpu_arg=(--gpu-csv "$out/gpu.csv")
{
  python3 examples/workload_bench.py --report "$out"/wl.*.log ${gpu_arg[@]+"${gpu_arg[@]}"}
  echo "stop reasons:"
  cat "$out"/wl.*.log | grep -o '"stop": "[a-z]*"' | sort | uniq -c
  echo "records per chain:"
  grep -c '^REC' "$out"/wl.*.log
} 2>&1 | tee "$out/REPORT.txt"
echo "all results and settings: $out"
