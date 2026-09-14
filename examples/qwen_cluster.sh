#!/usr/bin/env bash
# Drive the whole private Qwen swarm over SSH from one control node.
#
#   examples/qwen_cluster.sh deploy    # rsync this repo to every host, build a venv
#   examples/qwen_cluster.sh start     # bootstrap DHT, then every GPU server
#   examples/qwen_cluster.sh status    # per-host process state + layer coverage
#   examples/qwen_cluster.sh logs N07  # tail one host's server log
#   examples/qwen_cluster.sh stop      # stop servers, then the DHT
#
# Hosts come from task/hosts.txt: "<id> <ip>:<petals_port>  # comment".
# SSH is a separate port (SSH_PORT, default 22).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

HOSTS_FILE="${HOSTS_FILE:-task/hosts.txt}"
SSH_USER="${SSH_USER:-$(id -un)}"
SSH_PORT="${SSH_PORT:-22}"
SSH="${QWEN_SSH:-ssh}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
REMOTE_DIR="${REMOTE_DIR:-petals-qwen}"          # relative to the remote user's home
BOOTSTRAP_NODE="${BOOTSTRAP_NODE:-N01}"
DHT_PORT="${DHT_PORT:-31337}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
MAX_DISK_SPACE="${MAX_DISK_SPACE:-40GB}"
PY="${PY:-python3.10}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.2.2}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"           # Hub download of ~25 GB per host takes a while
STATE_DIR="${STATE_DIR:-$REPO_ROOT/.qwen-cluster}"
PEER_FILE="$STATE_DIR/bootstrap_peer"

IDS=() IPS=() PORTS=()
while read -r id addr _rest; do
  [[ -z "${id:-}" || "$id" == \#* ]] && continue
  IDS+=("$id")
  IPS+=("${addr%%:*}")
  PORTS+=("${addr##*:}")
done < <(sed 's/#.*//' "$HOSTS_FILE")
(( ${#IDS[@]} )) || { echo "No hosts parsed from $HOSTS_FILE" >&2; exit 2; }

index_of() {
  local want="$1" i
  for i in "${!IDS[@]}"; do [[ "${IDS[$i]}" == "$want" ]] && { echo "$i"; return; }; done
  echo "Unknown node id: $want" >&2; exit 2
}

remote() {  # remote <ip> <shell-command>
  $SSH $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$1" "$2"
}

# Run one command on every host at once; report which hosts failed.
fanout() {  # fanout <label> <command-template with {ID} {IP} {PORT}>
  local label="$1" template="$2" i pids=() rc=0
  mkdir -p "$STATE_DIR/out"
  for i in "${!IDS[@]}"; do
    local cmd="${template//\{ID\}/${IDS[$i]}}"
    cmd="${cmd//\{IP\}/${IPS[$i]}}"
    cmd="${cmd//\{PORT\}/${PORTS[$i]}}"
    remote "${IPS[$i]}" "$cmd" > "$STATE_DIR/out/${IDS[$i]}.$label" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "  [FAIL] ${IDS[$i]} (${IPS[$i]}) -- see $STATE_DIR/out/${IDS[$i]}.$label" >&2
      rc=1
    fi
  done
  return $rc
}

cmd_deploy() {
  command -v rsync >/dev/null || { echo "rsync is required on the control node" >&2; exit 2; }
  mkdir -p "$STATE_DIR/out"
  echo "Copying $REPO_ROOT to ${#IDS[@]} hosts as ~/$REMOTE_DIR/repo ..."
  local i pids=() rc=0
  for i in "${!IDS[@]}"; do
    (
      remote "${IPS[$i]}" "mkdir -p '$REMOTE_DIR/repo' '$REMOTE_DIR/logs' '$REMOTE_DIR/run' '$REMOTE_DIR/cache'"
      rsync -az --delete \
        --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
        --exclude '.venv*' --exclude '.pytest_cache' --exclude '.qwen-cluster' \
        -e "$SSH $SSH_OPTS -p $SSH_PORT" \
        "$REPO_ROOT/" "$SSH_USER@${IPS[$i]}:$REMOTE_DIR/repo/"
    ) > "$STATE_DIR/out/${IDS[$i]}.rsync" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "  [FAIL] rsync ${IDS[$i]}" >&2; rc=1; }
  done
  (( rc == 0 )) || return 1

  echo "Building the virtualenv on every host (this pulls torch, expect several minutes) ..."
  fanout install "
set -e
cd '$REMOTE_DIR'
test -x venv/bin/python || $PY -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q 'setuptools<81' wheel 'grpcio-tools==1.60.0'
venv/bin/pip install -q '$TORCH_SPEC' --index-url '$TORCH_INDEX_URL'
venv/bin/pip install -q --no-build-isolation -e repo
venv/bin/python -c 'import petals, torch; print(\"{ID}\", petals.__version__, torch.__version__)'
"
  echo "Deployed. Versions:"
  local id
  for id in "${IDS[@]}"; do printf '  %s\n' "$(tail -1 "$STATE_DIR/out/$id.install")"; done
}

cmd_start() {
  mkdir -p "$STATE_DIR/out"
  local b; b=$(index_of "$BOOTSTRAP_NODE")
  local bip="${IPS[$b]}"

  echo "Starting the DHT bootstrap on $BOOTSTRAP_NODE ($bip:$DHT_PORT) ..."
  remote "$bip" "
set -e
cd '$REMOTE_DIR'
if [ -f run/dht.pid ] && kill -0 \$(cat run/dht.pid) 2>/dev/null; then echo already-running; exit 0; fi
nohup venv/bin/python -m petals.cli.run_dht \
  --host_maddrs /ip4/0.0.0.0/tcp/$DHT_PORT \
  --announce_maddrs /ip4/$bip/tcp/$DHT_PORT \
  --identity_path qwen-dht.identity \
  > logs/dht.log 2>&1 &
echo \$! > run/dht.pid
"
  # The identity file keeps this address stable across restarts, so one read is enough.
  local peer="" waited=0
  while (( waited < 60 )); do
    peer=$(remote "$bip" "grep -ao '/ip4/${bip//./\\.}/tcp/$DHT_PORT/p2p/[A-Za-z0-9]*' '$REMOTE_DIR/logs/dht.log' | head -1" || true)
    [[ -n "$peer" ]] && break
    sleep 2; waited=$((waited + 2))
  done
  [[ -n "$peer" ]] || { echo "Could not read the bootstrap address from $BOOTSTRAP_NODE:$REMOTE_DIR/logs/dht.log" >&2; exit 1; }
  mkdir -p "$STATE_DIR"; printf '%s\n' "$peer" > "$PEER_FILE"
  echo "Bootstrap peer: $peer"

  # Pass through only the knobs that are set, so run_qwen_server.sh keeps its own defaults.
  local passthrough=""
  local name
  for name in DEVICE TORCH_DTYPE NUM_BLOCKS BLOCKS BALANCE_QUALITY DHT_PREFIX MODEL_REVISION; do
    [[ -n "${!name:-}" ]] && passthrough+="$name='${!name}' "
  done

  echo "Starting ${#IDS[@]} servers ..."
  fanout start "
set -e
cd '$REMOTE_DIR'
if [ -f run/server.pid ] && kill -0 \$(cat run/server.pid) 2>/dev/null; then echo already-running; exit 0; fi
cd repo
BOOTSTRAP_PEER='$peer' ANNOUNCE_IP='{IP}' PORT='{PORT}' \
MODEL_NAME='$MODEL_NAME' MAX_DISK_SPACE='$MAX_DISK_SPACE' $passthrough \
CACHE_DIR=\"\$HOME/$REMOTE_DIR/cache\" PYTHON=\"\$HOME/$REMOTE_DIR/venv/bin/python\" \
nohup bash examples/run_qwen_server.sh > \"\$HOME/$REMOTE_DIR/logs/server.log\" 2>&1 &
echo \$! > \"\$HOME/$REMOTE_DIR/run/server.pid\"
echo started {ID}
"
  echo
  echo "Servers are loading weights. Watch coverage with:"
  echo "  examples/qwen_cluster.sh status --watch"
}

cmd_status() {
  local peer; peer=$(cat "$PEER_FILE" 2>/dev/null || true)
  [[ -n "$peer" ]] || { echo "No bootstrap address recorded; run 'start' first." >&2; exit 1; }

  printf '%-5s %-16s %-6s %s\n' NODE ADDRESS PORT PROCESS
  local i state
  for i in "${!IDS[@]}"; do
    state=$(remote "${IPS[$i]}" "
      if [ -f '$REMOTE_DIR/run/server.pid' ] && kill -0 \$(cat '$REMOTE_DIR/run/server.pid') 2>/dev/null
      then echo up; else echo DOWN; fi" 2>/dev/null || echo unreachable)
    printf '%-5s %-16s %-6s %s\n' "${IDS[$i]}" "${IPS[$i]}" "${PORTS[$i]}" "$state"
  done

  echo
  local b; b=$(index_of "$BOOTSTRAP_NODE")
  local extra=""
  [[ "${1:-}" == "--watch" ]] && extra="--watch --timeout $READY_TIMEOUT --interval 15"
  remote "${IPS[$b]}" "cd '$REMOTE_DIR/repo' && \"\$HOME/$REMOTE_DIR/venv/bin/python\" \
    examples/check_qwen_swarm.py --initial-peers '$peer' --model '$MODEL_NAME' $extra"
}

cmd_logs() {
  local id="${1:?usage: logs <node-id> [lines]}" lines="${2:-60}"
  local i; i=$(index_of "$id")
  remote "${IPS[$i]}" "tail -n $lines '$REMOTE_DIR/logs/server.log'"
}

cmd_stop() {
  echo "Stopping servers ..."
  fanout stop "
cd '$REMOTE_DIR' 2>/dev/null || exit 0
if [ -f run/server.pid ]; then kill \$(cat run/server.pid) 2>/dev/null || true; rm -f run/server.pid; fi
echo stopped {ID}
" || true
  local b; b=$(index_of "$BOOTSTRAP_NODE")
  echo "Stopping the DHT on $BOOTSTRAP_NODE ..."
  remote "${IPS[$b]}" "
cd '$REMOTE_DIR' 2>/dev/null || exit 0
if [ -f run/dht.pid ]; then kill \$(cat run/dht.pid) 2>/dev/null || true; rm -f run/dht.pid; fi
" || true
  echo "Stopped. The bootstrap identity is kept, so 'start' reuses the same peer address."
}

case "${1:-}" in
  deploy) shift; cmd_deploy "$@" ;;
  start)  shift; cmd_start "$@" ;;
  status) shift; cmd_status "$@" ;;
  logs)   shift; cmd_logs "$@" ;;
  stop)   shift; cmd_stop "$@" ;;
  hosts)  printf '%s %s %s\n' "${IDS[@]}" | : ; for i in "${!IDS[@]}"; do
            printf '%-5s %-16s %s\n' "${IDS[$i]}" "${IPS[$i]}" "${PORTS[$i]}"; done ;;
  *) sed -n '2,12p' "$0" >&2; exit 2 ;;
esac
