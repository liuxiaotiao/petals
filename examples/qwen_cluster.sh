#!/usr/bin/env bash
# Drive the whole private Qwen swarm over SSH from one control node.
#
#   examples/qwen_cluster.sh preflight # check every host can be deployed to, change nothing
#   examples/qwen_cluster.sh plan      # size each host's blocks= from its free VRAM/disk
#   examples/qwen_cluster.sh deploy    # rsync this repo to every host, build a venv
#   examples/qwen_cluster.sh start     # bootstrap DHT, then every GPU server
#   examples/qwen_cluster.sh status    # per-host process state + layer coverage
#   examples/qwen_cluster.sh diag      # why is nothing online: alive? downloading? crashed?
#   examples/qwen_cluster.sh logs N07  # tail one host's server log
#   examples/qwen_cluster.sh stop      # stop servers, then the DHT
#
# Hosts come from task/hosts.txt: "<id> <ip>:<petals_port> [blocks=N]  # comment".
# blocks=N overrides NUM_BLOCKS for that one host, so a 16 GB card can serve fewer
# layers than a 24 GB one in the same cluster.
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
# Hub cache ceiling per host. A contiguous 11-layer range needs at most 25.5 GB of shard
# files (14 layers would need up to 30.5 GB), so 30GB leaves room without letting rebalancing
# grow the cache without bound. Petals evicts least-recently-used shards to stay under it.
MAX_DISK_SPACE="${MAX_DISK_SPACE:-30GB}"
# NODE_PY: an interpreter that already exists on every host (e.g. a conda env with torch).
# The venv is then built on top of it with --system-site-packages, so torch is inherited
# and Petals' own pins (transformers==4.43.1, numpy<2, peft, bitsandbytes) land in the venv
# instead of mutating that environment.
NODE_PY="${NODE_PY:-}"
PY="${PY:-python3.10}"                           # only used when NODE_PY is empty
if [[ -n "$NODE_PY" ]]; then
  TORCH_SPEC="${TORCH_SPEC-}"                    # inherited from NODE_PY unless set explicitly
else
  TORCH_SPEC="${TORCH_SPEC-torch==2.2.2}"
fi
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"           # Hub download of ~25 GB per host takes a while
STATE_DIR="${STATE_DIR:-$REPO_ROOT/.qwen-cluster}"
PEER_FILE="$STATE_DIR/bootstrap_peer"

IDS=() IPS=() PORTS=() NBLOCKS=()
while read -r id addr extra _rest; do
  [[ -z "${id:-}" || "$id" == \#* ]] && continue
  IDS+=("$id")
  IPS+=("${addr%%:*}")
  PORTS+=("${addr##*:}")
  if [[ "${extra:-}" == blocks=* ]]; then NBLOCKS+=("${extra#blocks=}"); else NBLOCKS+=(""); fi
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
    # Empty unless this host pinned its own layer count; placed last so it wins.
    local nb=""; [[ -n "${NBLOCKS[$i]}" ]] && nb="NUM_BLOCKS='${NBLOCKS[$i]}' "
    cmd="${cmd//\{NBLOCKS\}/$nb}"
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

# Everything deploy needs but does not install. Read-only: touches nothing on the hosts.
cmd_preflight() {
  local interp="${NODE_PY:-$PY}"
  echo "Checking ${#IDS[@]} hosts against interpreter: $interp"

  # Sent base64 so the quoting survives two levels of shell.
  local probe_b64
  probe_b64=$(base64 <<'PROBE' | tr -d '\n'
import urllib.error
import urllib.request


def probe(url):
    """An HTTP error still proves we reached the host; only a transport failure does not."""
    try:
        urllib.request.urlopen(url, timeout=15)
        return "ok"
    except urllib.error.HTTPError:
        return "ok"
    except Exception:
        return "UNREACHABLE"


free = "n/a"
try:
    import torch

    if not torch.cuda.is_available():
        free = "no-cuda"
    else:
        # Needs a real CUDA context, so this fails when the GPU is already saturated.
        free = "%.1fG" % (torch.cuda.mem_get_info(0)[0] / 1024**3)
except Exception as exc:
    free = "ERR-%s" % type(exc).__name__
# huggingface.co serves the index; Xet-backed repos serve the actual bytes from xethub.
print("hub=%s xet=%s vram_free=%s" % (
    probe("https://huggingface.co/api/models"),
    probe("https://cas-server.xethub.hf.co"),
    free,
))
PROBE
)

  # Clock skew, one host at a time so SSH latency does not pollute the reading.
  # hivemind compares peers to EACH OTHER (MAX_DHT_TIME_DISCREPANCY_SECONDS = 3s), so the
  # bootstrap node is the reference; this control node's own clock is irrelevant.
  local i raw=()
  for i in "${!IDS[@]}"; do
    local t0 t1 epoch
    t0=$(date +%s.%N)
    epoch=$(remote "${IPS[$i]}" "date +%s.%N" 2>/dev/null || echo "")
    t1=$(date +%s.%N)
    if [[ -z "$epoch" ]]; then
      raw+=("nan")
    else
      raw+=("$(awk -v r="$epoch" -v a="$t0" -v b="$t1" 'BEGIN {printf "%.1f", r - (a + b) / 2}')")
    fi
  done
  local bi; bi=$(index_of "$BOOTSTRAP_NODE")
  local base="${raw[$bi]}"
  local skews=()
  for i in "${!IDS[@]}"; do
    if [[ "${raw[$i]}" == nan || "$base" == nan ]]; then
      skews+=("?")
    else
      skews+=("$(awk -v x="${raw[$i]}" -v y="$base" 'BEGIN {printf "%+.1f", x - y}')")
    fi
  done

  fanout preflight "
py=\$('$interp' -c 'import sys; print(\"%d.%d.%d\" % sys.version_info[:3])' 2>/dev/null) || py=MISSING
if [ \"\$py\" = MISSING ]; then
  echo '{ID} python=MISSING (interpreter not found: $interp)'
  exit 1
fi
torch=\$('$interp' -c 'import torch; print(torch.__version__)' 2>/dev/null) || torch=MISSING
gpu=\$('$interp' -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO-CUDA\")' 2>/dev/null) || gpu=NO-CUDA
'$interp' -m venv --help >/dev/null 2>&1 && venv=ok || venv=MISSING
command -v git >/dev/null && git=ok || git=MISSING
command -v rsync >/dev/null && rsync=ok || rsync=MISSING
# Petals pulls hivemind straight from GitHub, so pip needs to reach it from this host.
if [ \"\$git\" = ok ]; then
  timeout 25 git ls-remote --exit-code https://github.com/learning-at-home/hivemind.git HEAD >/dev/null 2>&1 \
    && github=ok || github=UNREACHABLE
else
  github=skipped
fi
timeout 25 '$interp' -m pip download --no-deps -d /tmp/.qwen-pipcheck packaging >/dev/null 2>&1 \
  && pypi=ok || pypi=UNREACHABLE
rm -rf /tmp/.qwen-pipcheck
net=\$(echo '$probe_b64' | base64 -d > /tmp/.qwen_probe.py && timeout 45 '$interp' /tmp/.qwen_probe.py 2>/dev/null)
rm -f /tmp/.qwen_probe.py
disk=\$(df -Pk \"\$HOME\" | awk 'NR==2 {printf \"%.0fG\", \$4/1048576}')
echo \"{ID} python=\$py torch=\$torch gpu=\$gpu venv=\$venv git=\$git rsync=\$rsync github=\$github pypi=\$pypi \${net:-hub=? xet=? vram_free=?} disk_free=\$disk\"
case \"\$torch\$venv\$git\$rsync\$github\$pypi\$net\" in *MISSING*|*UNREACHABLE*) exit 1 ;; esac
" || true

  echo
  local bad=0
  for i in "${!IDS[@]}"; do
    local line skew mark
    line=$(tail -1 "$STATE_DIR/out/${IDS[$i]}.preflight" 2>/dev/null)
    skew="${skews[$i]}"
    # hivemind's own limit is 3s; flag at 2s so there is margin.
    if [[ "$skew" == "?" ]] || awk -v s="$skew" 'BEGIN {exit !(s < -2 || s > 2)}'; then
      mark="CLOCK-SKEW"
    else
      mark="ok"
    fi
    printf '  %s clock=%ss-vs-%s/%s\n' \
      "${line:-${IDS[$i]} no-response}" "$skew" "$BOOTSTRAP_NODE" "$mark"
    [[ "$line" == *MISSING* || "$line" == *UNREACHABLE* || -z "$line" || "$mark" != ok ]] && bad=$((bad + 1))
  done
  echo
  if (( bad )); then
    echo "$bad host(s) are not ready. Fix those before running deploy." >&2
    echo "  CLOCK-SKEW  -> hivemind drops peers >3s apart; sync NTP (chrony / systemd-timesyncd)" >&2
    echo "  xet=UNREACHABLE -> set HF_HUB_DISABLE_XET=1, or allow *.xethub.hf.co through egress" >&2
    return 1
  fi
  echo "All hosts are ready for deploy."
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

  local make_venv torch_step
  if [[ -n "$NODE_PY" ]]; then
    make_venv="'$NODE_PY' -m venv --system-site-packages venv"
    echo "Building a venv on top of $NODE_PY on every host (torch is inherited) ..."
  else
    make_venv="$PY -m venv venv"
    echo "Building the virtualenv on every host (this pulls torch, expect several minutes) ..."
  fi
  if [[ -n "$TORCH_SPEC" ]]; then
    torch_step="venv/bin/pip install -q '$TORCH_SPEC' --index-url '$TORCH_INDEX_URL'"
  else
    torch_step="venv/bin/python -c 'import torch' || { echo \"no torch in $NODE_PY\" >&2; exit 1; }"
  fi
  fanout install "
set -e
cd '$REMOTE_DIR'
test -x venv/bin/python || $make_venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q 'setuptools<81' wheel 'grpcio-tools==1.60.0'
$torch_step
venv/bin/pip install -q --no-build-isolation -e repo
venv/bin/python -c 'import petals, torch; print(\"{ID}\", petals.__version__, torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NO-CUDA\")'
"
  echo "Deployed. node / petals / torch / GPU:"
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
  for name in DEVICE TORCH_DTYPE NUM_BLOCKS BLOCKS BALANCE_QUALITY DHT_PREFIX MODEL_REVISION \
              HF_HUB_DISABLE_XET HF_ENDPOINT HF_TOKEN HTTP_PROXY HTTPS_PROXY NO_PROXY; do
    [[ -n "${!name:-}" ]] && passthrough+="$name='${!name}' "
  done

  echo "Starting ${#IDS[@]} servers ..."
  fanout start "
set -e
cd '$REMOTE_DIR'
if [ -f run/server.pid ] && kill -0 \$(cat run/server.pid) 2>/dev/null; then echo already-running; exit 0; fi
cd repo
BOOTSTRAP_PEER='$peer' ANNOUNCE_IP='{IP}' PORT='{PORT}' \
MODEL_NAME='$MODEL_NAME' MAX_DISK_SPACE='$MAX_DISK_SPACE' $passthrough {NBLOCKS}\
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

# One line per host answering "is it alive, is it downloading, what broke".
cmd_diag() {
  echo "host  state  cache  log  | last error"
  fanout diag "
log=\"\$HOME/$REMOTE_DIR/logs/server.log\"
if [ -f \"\$HOME/$REMOTE_DIR/run/server.pid\" ] && kill -0 \$(cat \"\$HOME/$REMOTE_DIR/run/server.pid\") 2>/dev/null
then alive=running; else alive=DEAD; fi
cache=\$(du -sh \"\$HOME/$REMOTE_DIR/cache\" 2>/dev/null | cut -f1)
if [ ! -f \"\$log\" ]; then
  echo \"{ID} NO-LOG cache=\${cache:-0} | server was never started on this host\"
  exit 0
fi
lines=\$(wc -l < \"\$log\")
last=\$(grep -aiE 'error|exception|traceback|assert|killed|no kernel image|out of memory' \"\$log\" | tail -1 | cut -c1-140)
echo \"{ID} \$alive cache=\${cache:-0} lines=\$lines | \${last:-no error lines}\"
" || true
  local id
  for id in "${IDS[@]}"; do
    printf '  %s\n' "$(tail -1 "$STATE_DIR/out/$id.diag" 2>/dev/null || echo "$id no-response")"
  done
  echo
  echo "cache= grows while weights download. DEAD with a traceback means it crashed;"
  echo "full log: bash examples/qwen_cluster.sh logs <node-id> 80"
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
  preflight) shift; cmd_preflight "$@" ;;
  plan)   shift; python3 examples/plan_blocks.py --state-dir "$STATE_DIR" \
            --hosts-file "$HOSTS_FILE" "$@" ;;
  deploy) shift; cmd_deploy "$@" ;;
  start)  shift; cmd_start "$@" ;;
  status) shift; cmd_status "$@" ;;
  diag)   shift; cmd_diag "$@" ;;
  logs)   shift; cmd_logs "$@" ;;
  stop)   shift; cmd_stop "$@" ;;
  hosts)  printf '%-5s %-16s %-6s %s\n' NODE ADDRESS PORT BLOCKS; for i in "${!IDS[@]}"; do
            printf '%-5s %-16s %-6s %s\n' "${IDS[$i]}" "${IPS[$i]}" "${PORTS[$i]}" \
              "${NBLOCKS[$i]:-(NUM_BLOCKS)}"; done ;;
  *) sed -n '2,15p' "$0" >&2; exit 2 ;;
esac
