#!/usr/bin/env bash
# Drive the whole private Qwen swarm over SSH from one control node.
#
#   examples/qwen_cluster.sh preflight # check every host can be deployed to, change nothing
#   examples/qwen_cluster.sh synctime  # clock skew, read from each host's NTP daemon where
#                                     # there is one; --yes steps unsynced clocks to the bootstrap
#   examples/qwen_cluster.sh plan      # size each host's blocks= from its free VRAM/disk
#   examples/qwen_cluster.sh deploy    # rsync this repo to every host, build a venv
#   examples/qwen_cluster.sh proxy start  # lend PROXY_NODE's egress to hosts that have none
#   examples/qwen_cluster.sh start     # bootstrap DHT, then every GPU server
#   examples/qwen_cluster.sh start --restart  # also restart servers already running,
#                                     # the only way to change a running server's environment
#   examples/qwen_cluster.sh status    # per-host process state + layer coverage
#   examples/qwen_cluster.sh client --prompt '...'   # generate, from a node that has Petals
#   examples/qwen_cluster.sh diag      # why is nothing online: alive? downloading? crashed?
#   examples/qwen_cluster.sh cleanup   # list stale/GPU-holding processes (kills nothing)
#   examples/qwen_cluster.sh cleanup --ours --yes   # kill this deployment's leftovers
#   examples/qwen_cluster.sh cleanup --gpu  --yes   # ALSO kill other processes on the GPU
#   examples/qwen_cluster.sh cleanup --stale --yes  # kill only servers from an earlier
#                                     # generation that survived a restart and still hold the port
#   examples/qwen_cluster.sh logs N07  # tail one host's server log
#   examples/qwen_cluster.sh stop      # stop the servers in HOSTS_FILE; leaves the DHT up
#   examples/qwen_cluster.sh stop --dht  # also stop the bootstrap DHT (takes the whole swarm down)
#
# CLEANUP_ON_START=gpu makes 'start' first kill every foreign process holding GPU
# memory. Only set it where this cluster owns the GPUs outright.
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
# The node that lends its egress to the rest. It needs to reach the Hub and to be
# reachable from the other hosts; nothing else.
PROXY_NODE="${PROXY_NODE:-N08}"
PROXY_PORT="${PROXY_PORT:-8899}"
PROXY_SKIP="${PROXY_SKIP:-}"            # node ids with their own egress, space separated
CLIENT_NODE="${CLIENT_NODE:-$PROXY_NODE}"  # the node the client runs on
PROXY_ALLOW="${PROXY_ALLOW:-}"          # client IPs; empty means every host in HOSTS_FILE
PROXY_PORTS="${PROXY_PORTS:-80,443}"    # destination ports the proxy will open
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

# Returns non-zero instead of aborting: a subset HOSTS_FILE legitimately omits nodes that
# some commands still need to reach, and those can fall back rather than refuse to run.
find_index() {
  local want="$1" i
  for i in "${!IDS[@]}"; do [[ "${IDS[$i]}" == "$want" ]] && { echo "$i"; return 0; }; done
  return 1
}

index_of() {
  local i
  i=$(find_index "$1") || { echo "Unknown node id: $1" >&2; exit 2; }
  echo "$i"
}

# The bootstrap node's address, even when this HOSTS_FILE does not list it. Restarting a
# few hosts from a subset file is a normal thing to do, and it should not require listing
# the bootstrap node just so its address can be looked up: the cached peer multiaddr
# already carries it, and its presence means that DHT is the one already running.
bootstrap_ip() {
  local i
  if i=$(find_index "$BOOTSTRAP_NODE"); then printf '%s\n' "${IPS[$i]}"; return 0; fi
  local ip; ip=$(sed -n 's|^/ip4/\([0-9.]*\)/.*|\1|p' "$PEER_FILE" 2>/dev/null | head -1)
  [[ -n "$ip" ]] || return 1
  printf '%s\n' "$ip"
}

remote() {  # remote <ip> <shell-command>
  # -n detaches stdin. Without it, a remote command that tries to read the terminal
  # (sudo asking for a password, say) gets SIGTTIN and the whole backgrounded fanout
  # silently stops instead of failing. rsync must NOT get -n, so it is set here only.
  $SSH -n $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$1" "$2"
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
    cmd="${cmd//\{PROXY\}/$(proxy_env_for "${IDS[$i]}")}"
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

# Fill SKEWS[] with each host's clock offset from the bootstrap node in seconds,
# SKEW_SRC[] with how it was measured, and SKEW_ERR[] with that measurement's error bar.
#
# Timing a round trip over SSH cannot resolve seconds. The midpoint estimate assumes the
# path is symmetric, so a host whose session teardown runs a few seconds slower than its
# setup reads as multi-second skew while its clock is in fact perfect -- an artifact that
# once sent this cluster chasing a clock problem that did not exist. So when a host runs a
# time daemon, ask the daemon for its offset from real NTP time instead: two hosts each
# disciplined to true time agree with each other, which is all hivemind checks. The
# round-trip estimate is the fallback for hosts with no daemon, and it is tagged with the
# RTT that bounds its error so nobody reads it as precise.
SKEWS=()
SKEW_SRC=()
SKEW_ERR=()
measure_skew() {
  local i ntp=() rtt=() err=() src=()
  for i in "${!IDS[@]}"; do
    local t0 t1 out epoch offset
    t0=$(date +%s.%N)
    # No $ or quotes in this command: it is passed through two shells before it runs.
    out=$(remote "${IPS[$i]}" 'date +%s.%N; command -v chronyc >/dev/null 2>&1 && chronyc tracking 2>/dev/null | grep -E "^(Leap status|System time)"' 2>/dev/null || echo "")
    t1=$(date +%s.%N)
    epoch=$(printf '%s\n' "$out" | head -1)
    # "System time : 0.000000029 seconds slow of NTP time" -> the clock is that far behind
    # true time. Only trusted while chrony reports a Normal leap status.
    offset=$(printf '%s\n' "$out" | awk '
      /^Leap status/ { leap = $4 }
      /^System time/ { mag = $4; dir = $6 }
      END { if (leap == "Normal" && mag != "") printf "%s%s", (dir == "fast" ? "+" : "-"), mag }')
    if [[ -z "$epoch" ]]; then
      ntp+=("nan"); rtt+=("nan"); err+=("nan"); src+=("unreachable")
      continue
    fi
    ntp+=("${offset:-nan}")
    rtt+=("$(awk -v r="$epoch" -v a="$t0" -v b="$t1" 'BEGIN {printf "%.3f", r - (a + b) / 2}')")
    err+=("$(awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.2f", (b - a) / 2}')")
    if [[ -n "$offset" ]]; then src+=("ntp"); else src+=("rtt"); fi
  done

  local bi
  bi=$(find_index "$BOOTSTRAP_NODE") || bi=0   # subset file: any host serves as the reference
  SKEWS=(); SKEW_SRC=(); SKEW_ERR=()
  for i in "${!IDS[@]}"; do
    # Daemon offsets are in the true-time frame and round-trip estimates are in the control
    # node's frame; the two cannot be subtracted from each other, so a pair falls back to
    # the round-trip frame unless BOTH ends read their own daemon.
    if [[ "${src[$i]}" == unreachable || "${src[$bi]}" == unreachable ]]; then
      SKEWS+=("?"); SKEW_SRC+=("unreachable"); SKEW_ERR+=("0")
    elif [[ "${src[$i]}" == ntp && "${src[$bi]}" == ntp ]]; then
      SKEWS+=("$(awk -v x="${ntp[$i]}" -v y="${ntp[$bi]}" 'BEGIN {printf "%+.3f", x - y}')")
      SKEW_SRC+=("ntp"); SKEW_ERR+=("0.05")
    else
      SKEWS+=("$(awk -v x="${rtt[$i]}" -v y="${rtt[$bi]}" 'BEGIN {printf "%+.1f", x - y}')")
      SKEW_SRC+=("rtt"); SKEW_ERR+=("$(awk -v x="${err[$i]}" -v y="${err[$bi]}" 'BEGIN {printf "%.2f", x + y}')")
    fi
  done
}

# Step every host's clock to the bootstrap node's. A stopgap for clusters whose egress
# blocks public NTP: it aligns peers with each other, which is all hivemind checks, but
# nothing keeps them aligned afterwards.
cmd_synctime() {
  local confirm=0 force=0 a
  for a in "$@"; do
    case "$a" in
      --yes) confirm=1 ;;
      --force) force=1 ;;
    esac
  done

  measure_skew
  local i bi
  bi=$(find_index "$BOOTSTRAP_NODE") || bi=0
  local synced=0 known=0
  printf '%-5s %-16s %-12s %s\n' NODE ADDRESS "skew vs $BOOTSTRAP_NODE" measured-by
  for i in "${!IDS[@]}"; do
    local note="${SKEW_SRC[$i]}"
    [[ "$note" == rtt ]] && note="rtt (+-${SKEW_ERR[$i]}s)"
    [[ "$note" == ntp ]] && note="its own NTP daemon"
    printf '%-5s %-16s %-12s %s\n' "${IDS[$i]}" "${IPS[$i]}" "${SKEWS[$i]}s" "$note"
    [[ "${SKEW_SRC[$i]}" == ntp ]] && synced=$((synced + 1))
    [[ "${SKEW_SRC[$i]}" != unreachable ]] && known=$((known + 1))
  done

  echo
  if (( synced == known && known > 0 )); then
    echo "Every host is disciplined by a running NTP daemon, so these offsets are read from"
    echo "the daemons and are accurate to milliseconds. There is nothing to step."
    if (( confirm && ! force )); then
      echo
      echo "Refusing --yes: stepping a clock out from under chrony makes things worse, not" >&2
      echo "better -- chrony drags it back, and the cluster is genuinely skewed until it does." >&2
      echo "Pass --force only if you have stopped the time daemons first." >&2
      return 1
    fi
    (( confirm )) || return 0
  elif (( synced )); then
    echo "$synced of $known host(s) read their offset from a running NTP daemon (millisecond"
    echo "accuracy); the rest are timed over SSH, where +-a second or two is measurement noise."
  else
    echo "No host runs a time daemon, so every number above is an SSH round-trip estimate."
    echo "Its error bar is in the last column: treat anything inside it as noise, not skew."
  fi

  if (( ! confirm )); then
    echo
    echo "Dry run. 'synctime --yes' steps each host's clock to $BOOTSTRAP_NODE's."
    echo "Needs passwordless sudo. Fix the NTP source too -- this does not stop the drift."
    return 0
  fi

  echo
  echo "Stepping clocks to $BOOTSTRAP_NODE ..."
  for i in "${!IDS[@]}"; do
    [[ "$i" == "$bi" ]] && continue
    # Re-read the reference per host: the loop itself takes time.
    local ref; ref=$(remote "${IPS[$bi]}" "date +%s.%N" 2>/dev/null || echo "")
    if [[ -z "$ref" ]]; then echo "  ${IDS[$i]} SKIPPED (bootstrap unreachable)"; continue; fi
    printf '  %-5s %s\n' "${IDS[$i]}" \
      "$(remote "${IPS[$i]}" "sudo -n date -s @$ref >/dev/null 2>&1 && echo stepped || echo 'FAILED (passwordless sudo?)'" 2>/dev/null || echo unreachable)"
  done

  echo
  echo "After:"
  measure_skew
  for i in "${!IDS[@]}"; do printf '  %-5s %ss\n' "${IDS[$i]}" "${SKEWS[$i]}"; done
}

# Everything deploy needs but does not install. Read-only: touches nothing on the hosts.
cmd_preflight() {
  local interp="${NODE_PY:-$PY}"
  echo "Checking ${#IDS[@]} hosts against interpreter: $interp"

  # Sent base64 so the quoting survives two levels of shell.
  local probe_b64
  probe_b64=$(base64 <<'PROBE' | tr -d '\n'
import json
import os
import urllib.error
import urllib.request


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Follow 308 as well, which urllib only learned to do in Python 3.11.

    Mirrors lean on 308 heavily. On 3.10 an unfollowed one surfaces as an HTTPError that
    is indistinguishable from a missing repo, so a working mirror reads as a broken one.
    """

    def http_error_308(self, req, fp, code, msg, headers):
        # Aliasing it to http_error_301 is not enough: redirect_request() checks the code
        # against a hardcoded list that predates 308 and raises on anything outside it.
        # 308 is 307 with permanence, and permanence is irrelevant to a one-shot probe.
        return self.http_error_307(req, fp, 307, msg, headers)


OPENER = urllib.request.build_opener(_Redirects)


def open_url(url, headers=None, timeout=20):
    return OPENER.open(urllib.request.Request(url, headers=headers or {}), timeout=timeout)


def describe(exc):
    """Name what actually went wrong, not the wrapper it arrived in.

    A bare "UNREACHABLE" cannot tell a blocked port from a rejected certificate, and those
    want opposite fixes -- one is the firewall, the other is this interpreter's CA bundle.
    urllib buries the real failure in URLError.reason, so dig it out.
    """
    reason = getattr(exc, "reason", None)
    if isinstance(exc, urllib.error.URLError) and reason is not None:
        name = reason if isinstance(reason, str) else type(reason).__name__
        # A plain OSError names nothing useful; its strerror is the actual message,
        # which is what separates a DNS failure from a refused connection.
        if name == "OSError":
            detail = getattr(reason, "strerror", None) or (reason.args[0] if reason.args else None)
            if isinstance(detail, str):
                name = detail
    else:
        name = type(exc).__name__
    return str(name).replace(" ", "-")[:40]


def probe(url):
    """An HTTP error still proves we reached the host; only a transport failure does not."""
    try:
        open_url(url, timeout=15)
        return "ok"
    except urllib.error.HTTPError:
        return "ok"
    except Exception as exc:
        return "UNREACHABLE-%s" % describe(exc)


def probe_weights(endpoint, model):
    """Pull one real byte of one real shard, end to end, through this endpoint.

    Metadata and file content travel different paths: the API host answers with a redirect
    and the bytes come from a CDN on another domain, and a mirror can do the same. An
    allowlist that knows only the API host therefore lets a name-based check pass while the
    servers retry forever on an empty cache. Fetching actual weight bytes is the only probe
    that distinguishes the two, and it follows whatever endpoint is really in effect instead
    of a hardcoded CDN hostname that a mirror would never use.
    """
    revision = os.environ.get("QWEN_REVISION") or "main"
    base = "%s/%s/resolve/%s" % (endpoint, model, revision)
    try:
        index = json.loads(open_url(base + "/model.safetensors.index.json").read().decode())
        shard = sorted(set(index["weight_map"].values()))[0]
    except urllib.error.HTTPError as error:
        # An HTTP status means a server answered: the repo is missing, gated or misnamed.
        # Which one is the difference between "use another mirror" and "set HF_TOKEN",
        # so carry the code instead of collapsing both into one word.
        return "NO-INDEX-%d" % error.code
    except Exception as exc:
        # Anything else never reached one, which is a network problem, not a repo problem.
        return "UNREACHABLE-%s" % describe(exc)
    try:
        response = open_url(base + "/" + shard, headers={"Range": "bytes=0-0"}, timeout=30)
        return "ok" if response.read(1) else "EMPTY"
    except Exception as exc:
        return "UNREACHABLE-%s" % describe(exc)


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
endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
endpoint = endpoint.rstrip("/")
model = os.environ.get("QWEN_MODEL") or "Qwen/Qwen3.6-35B-A3B"
# Xet-backed repos serve their bytes from xethub rather than the CDN. With
# HF_HUB_DISABLE_XET set, or through a mirror, that path is never taken, so probing it
# would only report a blocker that is already handled.
if os.environ.get("QWEN_SKIP_XET") or endpoint != "https://huggingface.co":
    xet = "disabled"
else:
    xet = probe("https://cas-server.xethub.hf.co")
print("hub=%s cdn=%s xet=%s vram_free=%s%s" % (
    probe("%s/api/models/%s" % (endpoint, model)),
    probe_weights(endpoint, model),
    xet,
    free,
    "" if endpoint == "https://huggingface.co" else " via=%s" % endpoint.split("//")[-1],
))
PROBE
)

  measure_skew
  local skews=("${SKEWS[@]}") skewsrc=("${SKEW_SRC[@]}") skewerr=("${SKEW_ERR[@]}")

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
net=\$(echo '$probe_b64' | base64 -d > /tmp/.qwen_probe.py && \
  {PROXY}QWEN_SKIP_XET='${HF_HUB_DISABLE_XET:-}' HF_ENDPOINT='${HF_ENDPOINT:-}' QWEN_MODEL='$MODEL_NAME' QWEN_REVISION='${MODEL_REVISION:-}' timeout 75 '$interp' /tmp/.qwen_probe.py 2>/dev/null)
rm -f /tmp/.qwen_probe.py
disk=\$(df -Pk \"\$HOME\" | awk 'NR==2 {printf \"%.0fG\", \$4/1048576}')
echo \"{ID} python=\$py torch=\$torch gpu=\$gpu venv=\$venv git=\$git rsync=\$rsync github=\$github pypi=\$pypi \${net:-hub=? xet=? vram_free=?} disk_free=\$disk\"
case \"\$torch\$venv\$git\$rsync\$github\$pypi\" in *MISSING*|*UNREACHABLE*) exit 1 ;; esac
case \"\$net\" in *hub=ok*) ;; *) exit 1 ;; esac
case \"\$net\" in *cdn=ok*) ;; *) exit 1 ;; esac
" || true

  echo
  local bad=0
  for i in "${!IDS[@]}"; do
    local line skew mark
    line=$(tail -1 "$STATE_DIR/out/${IDS[$i]}.preflight" 2>/dev/null)
    skew="${skews[$i]}"
    # hivemind drops peers >3s apart. 2-3s still works but has no margin, so warn without
    # failing the host on it. Judge the SMALLEST skew the measurement is consistent with
    # (|skew| minus its error bar): a round-trip estimate with seconds of slop cannot prove
    # a clock is wrong, and blocking deploy on its noise is how a healthy cluster gets
    # flagged. Only a daemon-read offset, whose error bar is milliseconds, can fail a host.
    if [[ "$skew" == "?" ]]; then
      mark="CLOCK-SKEW"
    elif awk -v s="$skew" -v e="${skewerr[$i]}" 'BEGIN {exit !((s < 0 ? -s : s) - e > 3)}'; then
      mark="CLOCK-SKEW"
    elif awk -v s="$skew" -v e="${skewerr[$i]}" 'BEGIN {exit !((s < 0 ? -s : s) - e > 2)}'; then
      mark="clock-marginal"
    else
      mark="ok"
    fi
    printf '  %s clock=%ss-vs-%s(%s)/%s\n' \
      "${line:-${IDS[$i]} no-response}" "$skew" "$BOOTSTRAP_NODE" "${skewsrc[$i]}" "$mark"
    # Require hub=ok and cdn=ok by name: a host that cannot read the index or pull a
    # weight byte is not ready, whatever new word the probe invents to say so.
    [[ "$line" == *MISSING* || "$line" == *UNREACHABLE* || -z "$line" || "$mark" == CLOCK-SKEW \
       || "$line" != *hub=ok* || "$line" != *cdn=ok* ]] \
      && bad=$((bad + 1))
  done
  echo
  if (( bad )); then
    echo "$bad host(s) are not ready. Fix those before running deploy." >&2
    echo "  CLOCK-SKEW  -> hivemind drops peers >3s apart; sync NTP (chrony / systemd-timesyncd)." >&2
    echo "                 (rtt) means the offset was timed over SSH and is only good to a" >&2
    echo "                 second or two; (ntp) means the host's own daemon reported it." >&2
    echo "  xet=UNREACHABLE -> export HF_HUB_DISABLE_XET=1 and re-run; that makes this check pass" >&2
    echo "  cdn=UNREACHABLE-* -> no bytes arrived; the suffix says why:" >&2
    echo "     -timed-out / -ConnectionRefusedError -> the firewall drops it: allow" >&2
    echo "        *.cdn.hf.co and cdn-lfs*.huggingface.co, or point HF_ENDPOINT at a mirror" >&2
    echo "     -SSLCertVerificationError -> something is terminating TLS in the middle and" >&2
    echo "        this interpreter does not trust its CA. curl may still work here, because" >&2
    echo "        it reads the system store while Python reads certifi. Point the venv at" >&2
    echo "        the system bundle (SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt) or" >&2
    echo "        add the CA to certifi -- do NOT disable verification" >&2
    echo "     -NameResolutionError -> DNS does not resolve that host from this node" >&2
    echo "  cdn=NO-INDEX-404 -> that endpoint does not carry $MODEL_NAME; try another mirror" >&2
    echo "  cdn=NO-INDEX-401/403 -> the repo is gated: accept its terms and set HF_TOKEN" >&2
    echo "  vram_free=ERR-* -> CUDA context could not be created; the GPU is full or wedged" >&2
    echo "  (clock-marginal is a warning only and does not block deploy)" >&2
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
  local restart=0
  [[ "${1:-}" == --restart ]] && restart=1

  # Freeing the GPUs comes first: it is independent of the DHT, and a server that
  # starts onto an occupied card just OOMs. Kills only foreign GPU holders; our own
  # running servers are left to the already-running check further down.
  if [[ "${CLEANUP_ON_START:-}" == gpu ]]; then
    echo "CLEANUP_ON_START=gpu: freeing GPUs held by other processes ..."
    cmd_cleanup --gpu --yes
  fi

  local bip
  bip=$(bootstrap_ip) || {
    echo "$BOOTSTRAP_NODE is not in $HOSTS_FILE and no address is cached in $PEER_FILE." >&2
    echo "Start the full cluster once, or add $BOOTSTRAP_NODE to this hosts file." >&2
    exit 2
  }
  if ! find_index "$BOOTSTRAP_NODE" >/dev/null; then
    echo "$BOOTSTRAP_NODE is not in $HOSTS_FILE; using its cached address $bip and assuming its DHT is up."
  fi

  echo "Starting the DHT bootstrap on $BOOTSTRAP_NODE ($bip:$DHT_PORT) ..."
  remote "$bip" "
set -e
cd '$REMOTE_DIR'
if [ -f run/dht.pid ] && kill -0 \$(cat run/dht.pid) 2>/dev/null; then echo already-running; exit 0; fi
# The pidfile can be lost while the process lives on. Starting a second one then fails
# to bind the port and dies, leaving a confusing pair, so adopt the survivor instead.
# The launch line below uses a relative path, so match that form; grep -vx drops this
# very shell, whose own command line contains the launch line as literal text.
orphan=\$(pgrep -f \"venv/bin/python -m petals\\.cli\\.run_dht\" 2>/dev/null | grep -vx \"\$\$\" | head -1)
if [ -n \"\$orphan\" ]; then echo \"\$orphan\" > run/dht.pid; echo adopted-orphan; exit 0; fi
nohup venv/bin/python -m petals.cli.run_dht \
  --host_maddrs /ip4/0.0.0.0/tcp/$DHT_PORT \
  --announce_maddrs /ip4/$bip/tcp/$DHT_PORT \
  --identity_path qwen-dht.identity \
  > logs/dht.log 2>&1 &
echo \$! > run/dht.pid
"
  # Read it from the log rather than the cache: this DHT may have just been started.
  local peer
  peer=$(read_bootstrap_peer --fresh 60) || {
    echo "Could not read the bootstrap address from $BOOTSTRAP_NODE:$REMOTE_DIR/logs/dht.log" >&2
    exit 1
  }
  echo "Bootstrap peer: $peer"

  # Pass through only the knobs that are set, so run_qwen_server.sh keeps its own defaults.
  local passthrough=""
  local name
  for name in DEVICE TORCH_DTYPE NUM_BLOCKS BLOCKS BALANCE_QUALITY DHT_PREFIX MODEL_REVISION \
              HF_HUB_DISABLE_XET HF_ENDPOINT HF_TOKEN HTTP_PROXY HTTPS_PROXY NO_PROXY; do
    [[ -n "${!name:-}" ]] && passthrough+="$name='${!name}' "
  done

  # A server from an earlier run whose pidfile was lost would fight this one for the
  # port and the GPU. These match this deployment's own venv, so they are ours to kill.
  echo "Clearing stale servers from earlier runs ..."
  fanout clearstale "
venv_python=\"\$HOME/$REMOTE_DIR/venv/bin/python\"
if [ -f \"\$HOME/$REMOTE_DIR/run/server.pid\" ] && kill -0 \$(cat \"\$HOME/$REMOTE_DIR/run/server.pid\") 2>/dev/null
then echo '{ID} running-already'; exit 0; fi
for pid in \$(pgrep -f \"\$venv_python -m petals\\.cli\\.run_server\" 2>/dev/null); do
  [ \"\$pid\" = \"\$\$\" ] && continue
  kill -TERM \"\$pid\" 2>/dev/null || true
done
echo '{ID} cleared'
" || true

  # Say out loud which hosts get the proxy. Its absence is otherwise invisible until a
  # server has spent hours retrying against an endpoint it cannot reach.
  local addr; addr=$(proxy_addr)
  if [[ -n "$addr" ]]; then
    local routed=() i
    for i in "${!IDS[@]}"; do
      [[ -n "$(proxy_env_for "${IDS[$i]}")" ]] && routed+=("${IDS[$i]}")
    done
    echo "Hub access: ${#routed[@]} host(s) via $addr; the rest use their own egress."
  else
    echo "Hub access: no proxy registered, every host will use its own egress."
    echo "  If some hosts have none, run 'proxy start' first or they will retry forever." >&2
  fi

  local restart_step=""
  if (( restart )); then
    echo "--restart: stopping running servers first so they pick up this environment."
    # A Petals server needs time to unregister its blocks and release the port. Starting
    # the replacement a few seconds after SIGTERM leaves two processes on one address:
    # the DHT then advertises the new peer id at a port the old process still answers,
    # and every client that dials it fails with a peer id mismatch. So wait for the old
    # one to actually be gone, and stop being polite about it if it will not go.
    restart_step="
venv_python=\"\$HOME/$REMOTE_DIR/venv/bin/python\"
if [ -f run/server.pid ]; then kill \$(cat run/server.pid) 2>/dev/null || true; fi
waited=0
while [ \$waited -lt 60 ]; do
  alive=\$(pgrep -f \"\$venv_python -m petals\\.cli\\.run_server\" 2>/dev/null | grep -vx \"\$\$\" | head -1)
  [ -z \"\$alive\" ] && break
  sleep 2; waited=\$(( waited + 2 ))
done
for pid in \$(pgrep -f \"\$venv_python -m petals\\.cli\\.run_server\" 2>/dev/null); do
  [ \"\$pid\" = \"\$\$\" ] && continue
  kill -9 \"\$pid\" 2>/dev/null || true
done
rm -f run/server.pid
sleep 2"
  fi

  echo "Starting ${#IDS[@]} servers ..."
  fanout start "
set -e
cd '$REMOTE_DIR'$restart_step
if [ -f run/server.pid ] && kill -0 \$(cat run/server.pid) 2>/dev/null; then echo already-running; exit 0; fi
cd repo
BOOTSTRAP_PEER='$peer' ANNOUNCE_IP='{IP}' PORT='{PORT}' \
MODEL_NAME='$MODEL_NAME' MAX_DISK_SPACE='$MAX_DISK_SPACE' $passthrough {NBLOCKS}{PROXY}\
CACHE_DIR=\"\$HOME/$REMOTE_DIR/cache\" PYTHON=\"\$HOME/$REMOTE_DIR/venv/bin/python\" \
nohup bash examples/run_qwen_server.sh > \"\$HOME/$REMOTE_DIR/logs/server.log\" 2>&1 &
echo \$! > \"\$HOME/$REMOTE_DIR/run/server.pid\"
echo started {ID}
"
  echo
  echo "Servers are loading weights. Watch coverage with:"
  echo "  examples/qwen_cluster.sh status --watch"
}

# Report each host's process state and how fast its weight cache is growing.
#
# "JOINING" is indistinguishable from "wedged" without this. A 35B model over one shared
# uplink stays in JOINING for a long time legitimately, so the question that matters is
# not what state a server is in but whether its cache moved since the last look.
status_hosts() {
  fanout hoststate "
if [ -f '$REMOTE_DIR/run/server.pid' ] && kill -0 \$(cat '$REMOTE_DIR/run/server.pid') 2>/dev/null
then state=up; else state=DOWN; fi
bytes=\$(du -sb '$REMOTE_DIR/cache' 2>/dev/null | cut -f1)
echo \"\$state \${bytes:-0}\"
" >/dev/null 2>&1 || true

  local now; now=$(date +%s)
  mkdir -p "$STATE_DIR/cache_prev"
  printf '%-5s %-16s %-6s %-11s %-9s %s\n' NODE ADDRESS PORT PROCESS CACHE RATE
  local i
  for i in "${!IDS[@]}"; do
    local id="${IDS[$i]}" line state bytes
    line=$(tail -1 "$STATE_DIR/out/$id.hoststate" 2>/dev/null || true)
    state=$(awk '{print $1}' <<<"$line"); bytes=$(awk '{print $2}' <<<"$line")
    [[ "$state" =~ ^(up|DOWN)$ ]] || { state="unreachable"; bytes=""; }

    local rate="-"
    local prev="$STATE_DIR/cache_prev/$id"
    if [[ -n "$bytes" && -f "$prev" ]]; then
      rate=$(awk -v now="$now" -v bytes="$bytes" '
        {
          elapsed = now - $2
          if (elapsed > 0 && bytes >= $1) {
            speed = (bytes - $1) / elapsed / 1048576
            printf "%.1f MB/s", speed
          } else { printf "-" }
        }' "$prev")
    fi
    [[ -n "$bytes" ]] && printf '%s %s\n' "$bytes" "$now" > "$prev"

    printf '%-5s %-16s %-6s %-11s %-9s %s\n' "$id" "${IPS[$i]}" "${PORTS[$i]}" \
      "$state" "$(human_bytes "${bytes:-}")" "$rate"
  done
}

human_bytes() {
  [[ -n "${1:-}" ]] || { echo "?"; return; }
  awk -v b="$1" 'BEGIN {
    split("B KB MB GB TB", unit, " ")
    i = 1
    while (b >= 1024 && i < 5) { b /= 1024; i++ }
    printf "%.1f%s", b, unit[i]
  }'
}

cmd_status() {
  local peer
  peer=$(read_bootstrap_peer) || {
    echo "No bootstrap address cached, and $BOOTSTRAP_NODE's logs/dht.log has none either." >&2
    echo "Either the swarm was never started, or the DHT log was cleared. Run 'start'." >&2
    exit 1
  }
  local watch=0
  [[ "${1:-}" == "--watch" ]] && watch=1
  local bip; bip=$(bootstrap_ip) || { echo "Cannot locate $BOOTSTRAP_NODE." >&2; exit 2; }
  local deadline=$(( $(date +%s) + READY_TIMEOUT ))

  # Both halves are polled together: the DHT says which layers are claimed, the hosts say
  # whether the weights behind those claims are still arriving. Watching only one of them
  # is what makes a slow download look like a hung cluster.
  while true; do
    status_hosts
    echo
    local rc=0
    remote "$bip" "cd '$REMOTE_DIR/repo' && \"\$HOME/$REMOTE_DIR/venv/bin/python\" \
      examples/check_qwen_swarm.py --initial-peers '$peer' --model '$MODEL_NAME'" || rc=$?
    (( watch )) || return "$rc"
    (( rc == 0 )) && return 0
    if (( $(date +%s) >= deadline )); then
      echo "Gave up after ${READY_TIMEOUT}s. Raise READY_TIMEOUT, or run 'diag'." >&2
      return 1
    fi
    sleep 15
    echo
  done
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
# A dead server's last error can be hours old. Without the age of the log there is no
# way to tell a problem happening now from the wreckage of one already fixed.
now=\$(date +%s); mtime=\$(stat -c %Y \"\$log\" 2>/dev/null || echo \"\$now\")
age=\$(( now - mtime ))
if [ \"\$age\" -lt 120 ]; then age=\"\${age}s\"
elif [ \"\$age\" -lt 7200 ]; then age=\"\$(( age / 60 ))m\"
else age=\"\$(( age / 3600 ))h\"; fi
last=\$(grep -aiE 'error|exception|traceback|assert|killed|no kernel image|out of memory' \"\$log\" | tail -1 | cut -c1-130)
echo \"{ID} \$alive cache=\${cache:-0} lines=\$lines age=\$age | \${last:-no error lines}\"
" || true
  local id
  for id in "${IDS[@]}"; do
    printf '  %s\n' "$(tail -1 "$STATE_DIR/out/$id.diag" 2>/dev/null || echo "$id no-response")"
  done
  # Every server depends on the bootstrap DHT, so when nothing is online this is the
  # first thing to rule out. diag used to only read server logs and miss it entirely.
  local bip; bip=$(bootstrap_ip) || bip=""
  echo
  echo "bootstrap DHT on $BOOTSTRAP_NODE (${bip:-unknown}:$DHT_PORT):"
  [[ -n "$bip" ]] && remote "$bip" "
cd '$REMOTE_DIR' 2>/dev/null || { echo '  no $REMOTE_DIR on this host'; exit 0; }
if [ -f run/dht.pid ] && kill -0 \$(cat run/dht.pid) 2>/dev/null; then state=running; else state=DEAD; fi
listening=no
if command -v ss >/dev/null 2>&1; then
  ss -ltn 2>/dev/null | grep -q ':$DHT_PORT ' && listening=yes
elif command -v netstat >/dev/null 2>&1; then
  netstat -ltn 2>/dev/null | grep -q ':$DHT_PORT ' && listening=yes
else
  listening=unknown
fi
echo \"  state=\$state  port_$DHT_PORT=\$listening\"
echo '  --- last lines of logs/dht.log ---'
tail -8 logs/dht.log 2>/dev/null | cut -c1-140 | sed 's/^/  /' || echo '  (no dht.log)'
" 2>/dev/null || echo "  (host unreachable)"

  echo
  echo "age= is how long since that log was last written: a DEAD host with an old age"
  echo "is showing the wreckage of a past failure, not a live one."
  echo "cache= grows while weights download. DEAD with a traceback means it crashed;"
  echo "full log: bash examples/qwen_cluster.sh logs <node-id> 80"
  echo "If the DHT above is DEAD, every server will fail the same way -- fix it first."
}

# List, and optionally kill, processes that would get in a fresh start's way.
#   (no flags)     dry run: show everything, kill nothing
#   --ours         processes launched from this deployment's venv -- always safe to kill
#   --gpu          ANY process holding GPU memory, including other people's jobs
#   --yes          actually kill; without it this only reports
cmd_cleanup() {
  local want_ours=0 want_gpu=0 want_stale=0 confirm=0 arg
  for arg in "$@"; do
    case "$arg" in
      --ours) want_ours=1 ;;
      --gpu)  want_gpu=1 ;;
      --stale) want_stale=1 ;;
      --yes)  confirm=1 ;;
      *) echo "cleanup: unknown flag $arg" >&2; return 2 ;;
    esac
  done
  (( want_ours || want_gpu || want_stale )) || { want_ours=1; want_gpu=1; }   # dry run shows both

  # --stale is the surgical one: kill servers from an EARLIER generation while leaving the
  # current one running. A server that ignores SIGTERM keeps its p2pd bound to the Petals
  # port, and because libp2p sets SO_REUSEPORT the new server binds the same port happily.
  # Connections are then split between two peer ids, and clients fail the dial with a peer
  # id mismatch against what the DHT advertises. --ours cannot be used for this: it would
  # take down the healthy server too.
  if (( want_stale )); then
    echo "Stale servers (an earlier generation still holding the port):"
    fanout stale "
current=\$(cat '$REMOTE_DIR/run/server.pid' 2>/dev/null)
if [ -z \"\$current\" ]; then echo '{ID} no pidfile -- cannot tell which generation is current'; exit 0; fi
ps -eo pid=,ppid=,etimes=,args= | grep '[p]etals.cli.run_server' | while read -r pid ppid age rest; do
  [ \"\$ppid\" = 1 ] || continue                 # children of the live server, not generations
  [ \"\$pid\" = \"\$current\" ] && continue
  echo \"{ID} STALE pid=\$pid age=\${age}s\"
  if [ '$confirm' = 1 ]; then
    kids=\$(ps -eo pid=,ppid= | while read -r p pp; do [ \"\$pp\" = \"\$pid\" ] && echo \"\$p\"; done)
    kill -9 \$kids \$pid 2>/dev/null || true
    echo \"{ID}   killed \$pid \${kids:+and children \$kids}\"
  fi
done
echo \"{ID} live=\$current\"
" || true
    local id
    for id in "${IDS[@]}"; do
      sed 's/^/  /' "$STATE_DIR/out/$id.stale" 2>/dev/null || echo "  $id no-response"
    done
    if (( ! confirm )); then
      echo
      echo "Dry run. 'cleanup --stale --yes' kills the pids listed as STALE and their children."
      echo "Nothing marked live= is touched."
    fi
    (( want_ours || want_gpu )) || return 0
  fi

  if (( confirm && want_gpu )); then
    echo "WARNING: --gpu --yes kills every process holding GPU memory on all ${#IDS[@]} hosts," >&2
    echo "including jobs that are not yours. Run without --yes first and read the list." >&2
  fi

  fanout cleanup "
venv_python=\"\$HOME/$REMOTE_DIR/venv/bin/python\"
report() {  # report <pid> <tag> <extra>
  info=\$(ps -o user=,etime=,args= -p \"\$1\" 2>/dev/null | head -1 | cut -c1-110)
  [ -n \"\$info\" ] && echo \"{ID} pid=\$1 \$2 \$3 \$info\"
}
kill_pid() {
  kill -TERM \"\$1\" 2>/dev/null || return 0
  for _ in 1 2 3 4 5; do kill -0 \"\$1\" 2>/dev/null || return 0; sleep 1; done
  kill -KILL \"\$1\" 2>/dev/null || true
}

ours=\$(pgrep -f \"\$venv_python -m petals\\.cli\\.run_\" 2>/dev/null | grep -vx \"\$\$\" | tr '\\n' ' ')
gpu_pids=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\\n' ' ')

for pid in \$ours; do
  mem=\"\"
  for g in \$gpu_pids; do [ \"\$g\" = \"\$pid\" ] && mem=on-gpu; done
  report \"\$pid\" OURS \"\$mem\"
  [ '$want_ours$confirm' = '11' ] && kill_pid \"\$pid\"
done
for pid in \$gpu_pids; do
  mine=0
  for o in \$ours; do [ \"\$o\" = \"\$pid\" ] && mine=1; done
  [ \"\$mine\" = 1 ] && continue
  mb=\$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
        | awk -F, -v p=\"\$pid\" '\$1+0==p {gsub(/ /,\"\",\$2); print \$2}')
  report \"\$pid\" OTHER \"\$mb\"
  [ '$want_gpu$confirm' = '11' ] && kill_pid \"\$pid\"
done
echo '{ID} done'
" || true

  echo
  local id shown=0
  for id in "${IDS[@]}"; do
    local file="$STATE_DIR/out/$id.cleanup"
    [[ -f "$file" ]] || continue
    while IFS= read -r line; do
      [[ "$line" == *" done" ]] && continue
      printf '  %s\n' "$line"
      shown=$((shown + 1))
    done < "$file"
  done
  (( shown )) || { echo "  nothing running on any host"; return 0; }
  echo
  if (( confirm )); then
    echo "Killed the processes listed above (OURS$( ((want_gpu)) && echo " and OTHER" ))."
  else
    echo "Dry run -- nothing was killed. To act:"
    echo "  cleanup --ours --yes   # only this deployment's own leftovers"
    echo "  cleanup --gpu  --yes   # also other processes holding GPU memory"
  fi
}

cmd_logs() {
  local id="${1:?usage: logs <node-id> [lines]}" lines="${2:-60}"
  local i; i=$(index_of "$id")
  remote "${IPS[$i]}" "tail -n $lines '$REMOTE_DIR/logs/server.log'"
}

# Give every host without egress a way to the Hub through one that has it.
#
# HTTPS_PROXY is the whole mechanism: huggingface_hub goes through requests, requests
# honours it, and this script already forwards it to the servers. So one node with egress
# serves the cluster without touching the firewall, without pinning block ranges, and
# without a separate copy of the weights -- Petals keeps choosing its own layer ranges.
# The cost is that every host's download crosses that one node's uplink.
proxy_addr() {
  # Always succeeds: under set -e an assignment from a failing substitution kills the
  # caller, and "no proxy configured" is an ordinary state, not an error.
  [[ -f "$STATE_DIR/proxy_addr" ]] || return 0
  cat "$STATE_DIR/proxy_addr"
}

# Hosts reach the Hub through the proxy unless they are the proxy, or PROXY_SKIP exempts
# them because they already have their own egress and need not add load to that uplink.
# HTTP_PROXY is deliberately left unset: the proxy speaks CONNECT only, so pointing plain
# HTTP at it would turn a working request into a 405.
proxy_env_for() {
  local id="$1" addr skip
  addr=$(proxy_addr) || return 0
  [[ -n "$addr" ]] || return 0
  [[ "$id" == "$PROXY_NODE" ]] && return 0
  for skip in $PROXY_SKIP; do [[ "$id" == "$skip" ]] && return 0; done
  local direct; direct="localhost,127.0.0.1,$(IFS=,; echo "${IPS[*]}")"
  printf "HTTPS_PROXY='%s' https_proxy='%s' NO_PROXY='%s' no_proxy='%s' " \
    "$addr" "$addr" "$direct" "$direct"
}

# Recover the bootstrap address. --identity_path pins it, so it survives restarts and can
# always be re-read from the bootstrap node's own log; losing the cached copy (a fresh clone
# of this repo, say) must not look like "there is no swarm".
#   read_bootstrap_peer [--fresh] [wait_seconds]
read_bootstrap_peer() {
  local fresh=0
  [[ "${1:-}" == --fresh ]] && { fresh=1; shift; }
  local limit="${1:-0}" peer="" waited=0
  if (( ! fresh )); then
    peer=$(cat "$PEER_FILE" 2>/dev/null || true)
    if [[ -n "$peer" ]]; then printf '%s\n' "$peer"; return 0; fi
  fi
  local bip; bip=$(bootstrap_ip) || return 1
  while :; do
    peer=$(remote "$bip" \
      "grep -ao '/ip4/${bip//./\\.}/tcp/$DHT_PORT/p2p/[A-Za-z0-9]*' '$REMOTE_DIR/logs/dht.log' 2>/dev/null | head -1" \
      2>/dev/null || true)
    [[ -n "$peer" ]] && break
    (( waited >= limit )) && break
    sleep 2; waited=$((waited + 2))
  done
  [[ -n "$peer" ]] || return 1
  mkdir -p "$STATE_DIR"; printf '%s\n' "$peer" > "$PEER_FILE"
  printf '%s\n' "$peer"
}

cmd_proxy() {
  local action="${1:-status}"
  local i; i=$(index_of "$PROXY_NODE")
  local ip="${IPS[$i]}"
  local allow="${PROXY_ALLOW:-$(IFS=,; echo "${IPS[*]}")}"

  case "$action" in
    start)
      remote "$ip" "
set -e
cd '$REMOTE_DIR'
mkdir -p run logs
if [ -f run/proxy.pid ] && kill -0 \$(cat run/proxy.pid) 2>/dev/null; then echo already-running; exit 0; fi
nohup venv/bin/python repo/examples/qwen_http_proxy.py --port $PROXY_PORT --allow '$allow' --ports '$PROXY_PORTS' \
  > logs/proxy.log 2>&1 &
echo \$! > run/proxy.pid
sleep 1
if kill -0 \$(cat run/proxy.pid) 2>/dev/null; then echo started; else echo FAILED; tail -5 logs/proxy.log; exit 1; fi
"
      mkdir -p "$STATE_DIR"
      printf 'http://%s:%s\n' "$ip" "$PROXY_PORT" > "$STATE_DIR/proxy_addr"
      echo "Proxy up: http://$ip:$PROXY_PORT (on $PROXY_NODE)"

      # Starting is not the same as being reachable. Prove it from a host that needs it,
      # before preflight reports fifteen failures that all have one cause.
      local j; for j in "${!IDS[@]}"; do [[ "${IDS[$j]}" != "$PROXY_NODE" ]] && break; done
      local endpoint="${HF_ENDPOINT:-https://huggingface.co}"
      echo -n "Reachability from ${IDS[$j]}: "
      remote "${IPS[$j]}" "
HTTPS_PROXY='http://$ip:$PROXY_PORT' https_proxy='http://$ip:$PROXY_PORT' \
'$REMOTE_DIR/venv/bin/python' -c \"
import urllib.request as u
try:
    print('ok', u.urlopen('$endpoint/api/models/$MODEL_NAME', timeout=25).status)
except Exception as exc:
    print('FAILED', type(exc).__name__, getattr(exc, 'reason', ''))
\"" || echo "FAILED (host unreachable)"
      echo "Every other host now routes the Hub through it. Verify with: preflight"
      ;;

    stop)
      remote "$ip" "
cd '$REMOTE_DIR' 2>/dev/null || exit 0
if [ -f run/proxy.pid ]; then kill \$(cat run/proxy.pid) 2>/dev/null || true; rm -f run/proxy.pid; fi
echo stopped
" || true
      rm -f "$STATE_DIR/proxy_addr"
      echo "Proxy stopped. Hosts go back to direct egress."
      ;;

    status)
      echo -n "$PROXY_NODE ($ip:$PROXY_PORT): "
      remote "$ip" "
cd '$REMOTE_DIR' 2>/dev/null || { echo 'not deployed'; exit 0; }
if [ -f run/proxy.pid ] && kill -0 \$(cat run/proxy.pid) 2>/dev/null; then
  # grep -c prints 0 and still exits non-zero when nothing matches, so let it fail
  # quietly rather than appending a second count behind the first.
  tunnels=\$(grep -c ' -> ' logs/proxy.log 2>/dev/null || true)
  echo \"running, \${tunnels:-0} tunnel(s) opened\"
else
  echo DEAD
fi
" || echo "(host unreachable)"
      local addr; addr=$(proxy_addr)
      if [[ -n "$addr" ]]; then
        echo "Hosts routed through it: all except $PROXY_NODE${PROXY_SKIP:+ and $PROXY_SKIP}"
      else
        echo "Not registered in $STATE_DIR: start and preflight are NOT using it."
      fi
      ;;

    logs)
      remote "$ip" "tail -${2:-40} '$REMOTE_DIR/logs/proxy.log'" || true
      ;;

    *) echo "usage: qwen_cluster.sh proxy {start|stop|status|logs}" >&2; return 2 ;;
  esac
}

# Run the client from a node, because the control node has no Petals install and does not
# need one: every deployed host already has the venv, the repo and a warm tokenizer cache.
# CLIENT_NODE defaults to the proxy node, which by definition can reach the Hub.
cmd_client() {
  local node="$CLIENT_NODE"
  if [[ "${1:-}" == --node ]]; then node="$2"; shift 2; fi
  local i; i=$(index_of "$node")
  local peer
  peer=$(read_bootstrap_peer) || {
    echo "No bootstrap address; is the swarm running?" >&2; exit 1
  }

  # Anything after the subcommand goes to qwen_generate.py untouched, so its own flags
  # (--prompt, --max-new-tokens, --revision, --dht-prefix) work without being mirrored here.
  local args=""
  local a
  for a in "$@"; do args+=" '${a//\'/\'\\\'\'}'"; done

  echo "Running the client on $node (${IPS[$i]}) ..."
  # PETALS_MAX_RETRIES: the client's default retry budget behaves as unlimited on this
  # path, so a real failure shows up as an endless wait instead of a traceback.
  remote "${IPS[$i]}" "
cd '$REMOTE_DIR/repo'
HF_HUB_DISABLE_XET='${HF_HUB_DISABLE_XET:-1}' \
PETALS_MAX_RETRIES='${PETALS_MAX_RETRIES:-3}' \
$(proxy_env_for "$node")\
"\$HOME/$REMOTE_DIR/venv/bin/python" examples/qwen_generate.py \
  --initial-peers '$peer' --model '$MODEL_NAME'${MODEL_REVISION:+ --revision '$MODEL_REVISION'}$args
"
}

cmd_stop() {
  local with_dht=0
  [[ "${1:-}" == --dht || "${1:-}" == --all ]] && with_dht=1
  echo "Stopping servers ..."
  fanout stop "
cd '$REMOTE_DIR' 2>/dev/null || exit 0
if [ -f run/server.pid ]; then kill \$(cat run/server.pid) 2>/dev/null || true; rm -f run/server.pid; fi
echo stopped {ID}
" || true
  if (( ! with_dht )); then
    echo
    echo "Servers stopped. The bootstrap DHT on $BOOTSTRAP_NODE is still running:"
    echo "stopping it takes down every server in the swarm, including any this hosts file"
    echo "does not list. Use 'stop --dht' when that is what you mean."
    return 0
  fi

  local bip; bip=$(bootstrap_ip) || {
    echo "Cannot locate $BOOTSTRAP_NODE; the DHT was left running." >&2
    return 0
  }
  echo "Stopping the DHT on $BOOTSTRAP_NODE ..."
  remote "$bip" "
cd '$REMOTE_DIR' 2>/dev/null || exit 0
if [ -f run/dht.pid ]; then kill \$(cat run/dht.pid) 2>/dev/null || true; rm -f run/dht.pid; fi
" || true
  echo "Stopped. The bootstrap identity is kept, so 'start' reuses the same peer address."
}

case "${1:-}" in
  preflight) shift; cmd_preflight "$@" ;;
  synctime) shift; cmd_synctime "$@" ;;
  plan)   shift; python3 examples/plan_blocks.py --state-dir "$STATE_DIR" \
            --hosts-file "$HOSTS_FILE" "$@" ;;
  deploy) shift; cmd_deploy "$@" ;;
  start)  shift; cmd_start "$@" ;;
  status) shift; cmd_status "$@" ;;
  diag)   shift; cmd_diag "$@" ;;
  cleanup) shift; cmd_cleanup "$@" ;;
  logs)   shift; cmd_logs "$@" ;;
  proxy)  shift; cmd_proxy "$@" ;;
  client) shift; cmd_client "$@" ;;
  stop)   shift; cmd_stop "$@" ;;
  hosts)  printf '%-5s %-16s %-6s %s\n' NODE ADDRESS PORT BLOCKS; for i in "${!IDS[@]}"; do
            printf '%-5s %-16s %-6s %s\n' "${IDS[$i]}" "${IPS[$i]}" "${PORTS[$i]}" \
              "${NBLOCKS[$i]:-(NUM_BLOCKS)}"; done ;;
  *) sed -n '2,20p' "$0" >&2; exit 2 ;;
esac
