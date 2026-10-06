#!/usr/bin/env bash
# Rerun the sequential workload experiment the way it was run on 2026-10-05:
# one client per chain, started on the chain's layer-0 node and pinned to that chain's six
# servers, one request at a time; GPUs sampled throughout; settings snapshotted first and a
# report written at the end. Everything lands in one directory.
#
#   source ~/petals-env.sh
#   bash examples/workload_run.sh --name r5_2048 --max-new-tokens 2048
#   bash examples/workload_run.sh --name smoke --max-new-tokens 64 --limit 4
#
#   # rerun on the exact placement and prompts of an earlier run:
#   bash examples/workload_run.sh --restore-from ~/wl-results/2026-10-05/r4_2048_norobots \
#        --name r5_2048 --max-new-tokens 2048
#   bash examples/workload_run.sh --restore-from <run dir> --pin-all --restore-only
#
# Arguments after "--" go to every workload_bench.py client unchanged.
# Run it inside tmux: a full run is 2-3 hours.
#
# --restore-from DIR  takes a run directory (or its settings/ snapshot). It puts back that
#                     run's hosts.txt and prompt files; if the placement differs, it reinstalls
#                     the services, restarts every server and waits until each pinned node
#                     serves its recorded layers with a fresh peer ID and the swarm is usable.
# --pin-all           also pin the auto-placed nodes (the T4s) to the layers they served in
#                     the snapshot, so all 15 nodes match, not only the 12 on the chains.
# --restore-only      restore and verify, then stop without running.
# --shared            one queue for all chains instead of a fixed split: every chain takes the
#                     next prompt from task/workload/prompts.jsonl as soon as it is free
#                     (examples/workload_dispatch.py), so a faster chain runs more of them, each
#                     prompt still runs once, and the chains finish at about the same time.
#                     Without it each chain runs its own chain.<TAG>.jsonl, as on 2026-10-05.
# Without --restore-from the run still checks that every pinned node in hosts.txt serves the
# layers written there, and refuses to start otherwise.
#
# Chains are "TAG:node,node,..." in layer order; the first node hosts layer 0 and runs the
# client. The default is the pinned 2026-10-05 layout (docs/experiments/2026-10-05-*.md).
# Prompts come from task/workload/chain.<TAG>.jsonl (examples/workload_split.py), which must
# be deployed: the run refuses to start if a client node holds a different copy.
#
# Refuses to start while another workload client is running anywhere it can see: two clients
# on one chain double the load and, sharing a log name, overwrite each other's records.
set -u

name="" max_new=2048 extra=() restore="" pin_all=0 restore_only=0 shared=0 limit=""
CHAINS="${CHAINS:-A:N01,N05,N03,N07,N02,N06 B:N11,N08,N09,N13,N04,N10}"
PROMPT_DIR="${PROMPT_DIR:-task/workload}"
RESULTS="${RESULTS:-$HOME/wl-results}"
HOSTS_FILE="${HOSTS_FILE:-task/hosts.txt}"
SSH="${SSH:-ssh}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_PORT="${SSH_PORT:-22}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
REMOTE_DIR="${REMOTE_DIR:-petals-qwen}"

usage() { sed -n '2,43p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --name) name="${2:-}"; shift 2 ;;
    --max-new-tokens) max_new="${2:-}"; shift 2 ;;
    --restore-from) restore="${2:-}"; shift 2 ;;
    --pin-all) pin_all=1; shift ;;
    --restore-only) restore_only=1; shift ;;
    --shared) shared=1; shift ;;
    --limit) limit="${2:-}"; shift 2 ;;  # a smoke test: N prompts per chain (split) or in total (shared)
    --) shift; extra=("$@"); break ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done
(( restore_only )) && [[ -z "$restore" ]] && { echo "--restore-only needs --restore-from" >&2; exit 2; }
(( pin_all )) && [[ -z "$restore" ]] && { echo "--pin-all needs --restore-from" >&2; exit 2; }
[[ -n "$name" ]] || (( restore_only )) || usage
[[ -f examples/qwen_cluster.sh ]] || { echo "run this from the repo root" >&2; exit 2; }
[[ -n "${MODEL_NAME:-}" ]] || { echo "MODEL_NAME is not set: source ~/petals-env.sh first" >&2; exit 2; }
out="$RESULTS/$(date +%F)/$name"
(( restore_only )) || [[ ! -e "$out" ]] || { echo "$out already exists; pick another --name" >&2; exit 2; }

# --- placement --------------------------------------------------------------
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

live_placement() {  # "node peer_id start:end" for every node the swarm reports
  bash examples/qwen_cluster.sh peers 2>/dev/null | awk '/^N[0-9][0-9] / && $3 ~ /^[0-9]+:[0-9]+$/ {print $1, $2, $3}'
}
pinned_in() {  # "node start:end" for every node a hosts file pins with blocks=start:end
  sed 's/#.*//' "$1" | awk '{for (i = 3; i <= NF; i++) if ($i ~ /^blocks=[0-9]+:[0-9]+$/) {sub(/^blocks=/, "", $i); print $1, $i}}'
}
check_placement() {  # check_placement EXPECTED [IDS_BEFORE]: print what is off; fail if anything is
  live_placement > "$work/live"
  awk -v before="${2:-/dev/null}" -v livefile="$work/live" '
    BEGIN {
      while ((getline line < livefile) > 0) { split(line, f, " "); id[f[1]] = f[2]; rng[f[1]] = f[3] }
      while ((getline line < before) > 0) { split(line, f, " "); old[f[1]] = f[2] }
    }
    { node = $1; want = $2
      if (!(node in rng))        { print node " is not serving"; bad = 1; next }
      if (rng[node] != want)     { print node " serves " rng[node] ", should be " want; bad = 1; next }
      if (old[node] == id[node]) { print node " has not restarted yet"; bad = 1 } }
    END { exit bad }' "$1"
}
wait_until() {  # wait_until DESCRIPTION COMMAND...: retry every RESTORE_POLL_S (30) s for RESTORE_WAIT_MIN (20) minutes
  local what="$1" deadline=$((SECONDS + ${RESTORE_WAIT_MIN:-20} * 60)); shift
  until "$@" > "$work/why" 2>&1; do
    if (( SECONDS > deadline )); then
      echo "gave up waiting for $what:" >&2; cat "$work/why" >&2; exit 1
    fi
    echo "  waiting for $what: $(wc -l < "$work/why") left, e.g. $(head -1 "$work/why")"
    sleep "${RESTORE_POLL_S:-30}"
  done
}
swarm_usable() { bash examples/qwen_cluster.sh status 2>&1 | grep -q 'The swarm is usable' || { echo "swarm not usable yet"; return 1; }; }

if [[ -n "$restore" ]]; then
  snap="$restore"; [[ -d "$snap/settings" ]] && snap="$snap/settings"
  [[ -f "$snap/config/hosts.txt" && -f "$snap/cluster/peers.txt" ]] \
    || { echo "$restore is not a snapshot (no config/hosts.txt and cluster/peers.txt)" >&2; exit 2; }
  echo "restoring from $snap ..."

  # The hosts file to put back: as recorded, or with every node pinned where it then was.
  if (( pin_all )); then
    python3 - "$snap" > "$work/hosts.txt" <<'EOF' || exit 1
import re, sys
snap = sys.argv[1]
served = {}
for line in open(snap + "/cluster/peers.txt"):
    f = line.split()
    if len(f) >= 3 and re.fullmatch(r"N\d\d", f[0]) and re.fullmatch(r"\d+:\d+", f[2]):
        served[f[0]] = f[2]
for line in open(snap + "/config/hosts.txt"):
    line = line.rstrip("\n")
    m = re.match(r"^(N\d\d)(\s+\S+\s+)blocks=\S+(.*)$", line)
    if m and m[1] in served:
        line = f"{m[1]}{m[2]}blocks={served[m[1]]}{m[3]}"
    elif m:
        sys.exit(f"{m[1]} is not in the snapshot's peers.txt; cannot pin it")
    print(line)
EOF
  else
    cp "$snap/config/hosts.txt" "$work/hosts.txt"
  fi
  pinned_in "$work/hosts.txt" > "$work/expected"

  # Prompts: the recorded files, not a fresh draw (the datasets on the Hub can change).
  prompts_changed=0
  for f in "$snap"/prompts/prompts.jsonl "$snap"/prompts/chain.*.jsonl; do
    [[ -f "$f" ]] || continue
    if ! cmp -s "$f" "$PROMPT_DIR/${f##*/}"; then
      mkdir -p "$PROMPT_DIR"; cp "$f" "$PROMPT_DIR/"; prompts_changed=1
      echo "  restored $PROMPT_DIR/${f##*/}"
    fi
  done
  if (( prompts_changed )); then
    echo "  deploying the restored prompts ..."
    bash examples/qwen_cluster.sh deploy > "$work/deploy.log" 2>&1 \
      || { echo "deploy failed:" >&2; tail -20 "$work/deploy.log" >&2; exit 1; }
  fi

  if cmp -s "$work/hosts.txt" "$HOSTS_FILE" && check_placement "$work/expected" > "$work/why"; then
    echo "  placement already matches the snapshot ($(wc -l < "$work/expected") pinned nodes); no restart"
  else
    [[ -s "$work/why" ]] && { echo "  placement differs:"; sed 's/^/    /' "$work/why"; }
    backup="$HOSTS_FILE.bak-$(date +%Y%m%d-%H%M%S)"
    cp "$HOSTS_FILE" "$backup" && cp "$work/hosts.txt" "$HOSTS_FILE"
    echo "  $HOSTS_FILE replaced (previous copy: $backup)"
    live_placement > "$work/before"
    echo "  reinstalling the services and restarting every server ..."
    bash examples/qwen_cluster.sh service install > "$work/install.log" 2>&1 \
      || { echo "service install failed:" >&2; tail -20 "$work/install.log" >&2; exit 1; }
    bash examples/qwen_cluster.sh service restart > "$work/restart.log" 2>&1 \
      || { echo "service restart failed:" >&2; tail -20 "$work/restart.log" >&2; exit 1; }
    wait_until "every pinned node to serve its recorded layers" check_placement "$work/expected" "$work/before"
    wait_until "the swarm to be usable" swarm_usable
  fi
  if (( pin_all )); then
    diff <(awk '{print $1, $3}' "$snap/cluster/peers.txt" | grep -E '^N[0-9]{2} [0-9]+:[0-9]+$' | sort) \
         <(live_placement | awk '{print $1, $3}' | sort) > "$work/why" \
      && echo "  all nodes serve exactly the layers in the snapshot" \
      || { echo "  placement still differs from the snapshot:" >&2; cat "$work/why" >&2; exit 1; }
  fi
  echo "restore done."
  (( restore_only )) && exit 0
fi

# Always: every node hosts.txt pins must serve exactly those layers before anything runs.
pinned_in "$HOSTS_FILE" > "$work/expected"
if [[ -s "$work/expected" ]] && ! check_placement "$work/expected" > "$work/why"; then
  echo "placement does not match $HOSTS_FILE:" >&2; sed 's/^/  /' "$work/why" >&2
  echo "fix it, or rerun with --restore-from <an earlier run> to put a recorded placement back" >&2
  exit 1
fi

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
  # The client code itself must be what is here: an old copy silently measures differently.
  here=$(md5sum < examples/workload_bench.py | cut -c1-32)
  there=$(rsh "${IP[$first]}" "md5sum < $REMOTE_DIR/repo/examples/workload_bench.py" 2>/dev/null | cut -c1-32)
  [[ "$here" == "$there" ]] || { echo "examples/workload_bench.py on $first differs from this copy: run 'bash examples/qwen_cluster.sh deploy'" >&2; exit 2; }
  (( shared )) && continue  # shared queue: prompts are sent from here, nothing to check on the node
  file="$PROMPT_DIR/chain.$tag.jsonl"
  [[ -s "$file" ]] || { echo "missing $file: run examples/workload_split.py first" >&2; exit 2; }
  here=$(md5sum < "$file" | cut -c1-32)
  there=$(rsh "${IP[$first]}" "md5sum < $REMOTE_DIR/repo/$file" 2>/dev/null | cut -c1-32)
  [[ "$here" == "$there" ]] || { echo "$file on $first differs from this copy: run 'bash examples/qwen_cluster.sh deploy'" >&2; exit 2; }
done
if (( shared )); then
  [[ -s "$PROMPT_DIR/prompts.jsonl" ]] || { echo "missing $PROMPT_DIR/prompts.jsonl" >&2; exit 2; }
  grep -q remote_stdin examples/qwen_cluster.sh \
    || { echo "examples/qwen_cluster.sh cannot feed a client on stdin (CLIENT_STDIN); update it" >&2; exit 2; }
  grep -q -- '--worker' examples/workload_bench.py \
    || { echo "examples/workload_bench.py has no --worker mode; update it" >&2; exit 2; }
fi
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
  echo "mode=$( (( shared )) && echo "shared queue ($PROMPT_DIR/prompts.jsonl)" || echo "split ($PROMPT_DIR/chain.<TAG>.jsonl)")"
  echo "chains=$CHAINS"
  echo "client_args=${extra[*]:-}"
  echo "limit=${limit:-none}"
  echo "restored_from=${restore:-}${restore:+ (pin_all=$pin_all)}"
  for tag in "${!ALLOW[@]}"; do echo "allowed_servers_$tag=${ALLOW[$tag]}"; done
  echo "started=$(date -Is)"
} > "$out/RUN.txt"

bash examples/cluster_gpumon.sh "$out/gpu.csv" 2> "$out/gpumon.log" &
gpu=$!
if (( shared )); then
  chain_args=()
  for spec in $CHAINS; do
    tag="${spec%%:*}"
    # ALLOW is deliberately unquoted: each peer ID is its own argument.
    chain_args+=(--chain "$tag" "${CLIENT[$tag]}" ${ALLOW[$tag]})
    echo "chain $tag: worker on ${CLIENT[$tag]}, log $out/wl.$tag.log"
  done
  echo "running $(grep -c . "$PROMPT_DIR/prompts.jsonl") prompts from one shared queue; progress below and in $out/dispatch.log"
  python3 -u examples/workload_dispatch.py --prompts "$PROMPT_DIR/prompts.jsonl" --out "$out" \
    ${limit:+--limit "$limit"} "${chain_args[@]}" \
    -- --max-new-tokens "$max_new" ${extra[@]+"${extra[@]}"} > >(tee "$out/dispatch.log") 2>&1 &
  pids=($!)
else
  pids=()
  for spec in $CHAINS; do
    tag="${spec%%:*}"
    # ALLOW is deliberately unquoted: each peer ID is its own argument.
    CLIENT_SCRIPT=workload_bench.py bash examples/qwen_cluster.sh client --node "${CLIENT[$tag]}" \
      --prompts "$PROMPT_DIR/chain.$tag.jsonl" --allowed-servers ${ALLOW[$tag]} --tag "$tag" \
      --max-new-tokens "$max_new" ${limit:+--limit "$limit"} ${extra[@]+"${extra[@]}"} > "$out/wl.$tag.log" 2>&1 &
    pids+=($!)
    echo "chain $tag: client on ${CLIENT[$tag]}, $(wc -l < "$PROMPT_DIR/chain.$tag.jsonl") prompts, log $out/wl.$tag.log"
  done
  echo "running; progress: grep -c '^REC' $out/wl.*.log"
fi
trap 'echo "interrupted; stopping clients and gpumon" >&2; kill "${pids[@]}" "$gpu" 2>/dev/null; exit 130' INT TERM
trap 'rm -rf "$work"' EXIT
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
  [[ -f "$out/DISPATCH.txt" ]] && { echo; grep -v '^  ran:' "$out/DISPATCH.txt"; }
} 2>&1 | tee "$out/REPORT.txt"
echo "all results and settings: $out"
