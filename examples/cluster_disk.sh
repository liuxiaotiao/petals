#!/usr/bin/env bash
# Survey and reclaim disk space across a cluster. Nothing here is Petals-specific:
# it reads the same task/hosts.txt and speaks plain ssh, so it works unchanged while
# testing any other framework on the same machines.
#
# Why it exists: switching frameworks or models leaves behind caches that no single
# tool owns -- pip's download cache, conda's package cache, Hugging Face snapshots a
# client run pulled, installers nobody deleted. On this cluster that was ~190 GB over
# 15 hosts, and it only surfaced as "No space left on device" in the middle of an
# unrelated job. A survey you can run in ten seconds is the point; the deleting is
# the easy part.
#
#   examples/cluster_disk.sh survey                  # read-only, every target
#   examples/cluster_disk.sh clean pip conda         # dry run: what would go
#   examples/cluster_disk.sh clean pip conda --yes   # actually delete
#   examples/cluster_disk.sh clean hf --yes          # every HF repo except $KEEP
#   examples/cluster_disk.sh survey --host N03       # one host
#
# Safety, in the same shape as qwen_cluster.sh purge:
#   * dry run is the default; --yes is the only way to remove anything
#   * only the fixed targets below are removable, and only under the remote $HOME
#   * KEEP protects Hugging Face repos by exact name (space-separated)
#   * a host that is down is reported, not fatal: the other hosts still run
set -euo pipefail

HOSTS_FILE="${HOSTS_FILE:-task/hosts.txt}"
SSH="${SSH:-ssh}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_PORT="${SSH_PORT:-22}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10}"
# Models the cluster is actually serving. Deleting these costs a re-download over a
# link that, on this cluster, is the slowest thing in the building.
KEEP="${KEEP:-Qwen/Qwen3-30B-A3B}"

ALL_TARGETS="pip conda torch hf hfdata installers"

usage() {
  cat >&2 <<EOF
usage: $(basename "$0") {survey|clean} [target ...] [--host ID] [--yes]

targets (default: all)
  pip          ~/.cache/pip                          pip's download cache
  conda        ~/{anaconda3,miniconda3,.conda}/pkgs  conda's package cache
  torch        ~/.cache/torch{,_extensions}          hub checkpoints, JIT builds
  hf           ~/.cache/huggingface/hub/models--*    except KEEP="$KEEP"
  hfdata       ~/.cache/huggingface/datasets
  installers   ~/{Anaconda3,Miniconda3}-*.sh, ~/cuda_*.run

Every target is a cache or an installer: deleting one costs a re-download, never
state. Server weights (~/petals-qwen/cache) are deliberately NOT reachable from
here -- use 'qwen_cluster.sh purge', which knows to refuse while a server is up.
EOF
  exit 2
}

# --- hosts ------------------------------------------------------------------
# Accepts the qwen_cluster.sh format ("<id> <ip>:<port> [blocks=N] # comment") and
# also a bare list of addresses, so this runs against a cluster that has no hosts.txt.
IDS=() ADDRS=()
load_hosts() {
  [[ -r "$HOSTS_FILE" ]] || { echo "cannot read $HOSTS_FILE (set HOSTS_FILE=)" >&2; exit 2; }
  local id rest
  while read -r id rest; do
    [[ -z "$id" ]] && continue
    if [[ -z "$rest" ]]; then IDS+=("$id"); ADDRS+=("${id%%:*}")
    else IDS+=("$id"); ADDRS+=("${rest%%:*}")
    fi
  done < <(sed 's/#.*//' "$HOSTS_FILE")
  (( ${#IDS[@]} )) || { echo "no hosts parsed from $HOSTS_FILE" >&2; exit 2; }
}

# --- the payload that runs on each host -------------------------------------
# Piped in base64 rather than written to a temp file: two hosts cleaning at once
# would otherwise race on the same path, and quoting this through ssh by hand is
# how the last three bugs in this repo got written.
payload() {
  cat <<'PAYLOAD'
set -u
apply="${DISK_APPLY:-0}"
keep=" ${DISK_KEEP:-} "
targets="${DISK_TARGETS:-}"

paths_for() {
  local d name
  case "$1" in
    pip)        printf '%s\n' "$HOME/.cache/pip" ;;
    conda)      printf '%s\n' "$HOME/anaconda3/pkgs" "$HOME/miniconda3/pkgs" "$HOME/.conda/pkgs" ;;
    torch)      printf '%s\n' "$HOME/.cache/torch" "$HOME/.cache/torch_extensions" ;;
    hfdata)     printf '%s\n' "$HOME/.cache/huggingface/datasets" ;;
    installers) ls -d "$HOME"/Anaconda3-*.sh "$HOME"/Miniconda3-*.sh "$HOME"/cuda_*.run 2>/dev/null ;;
    hf)
      for d in "$HOME"/.cache/huggingface/hub/models--*; do
        [ -d "$d" ] || continue
        name=${d##*/}; name=${name#models--}; name=${name//--//}
        case "$keep" in *" $name "*) continue ;; esac
        printf '%s\n' "$d"
      done
      ;;
  esac
}

avail() { df -B1 --output=avail "$HOME" 2>/dev/null | tail -1 | tr -d ' '; }

echo "FREE $(avail)"

for t in $targets; do
  existing=()
  while IFS= read -r p; do [ -e "$p" ] && existing+=("$p"); done < <(paths_for "$t")
  if [ "${#existing[@]}" -eq 0 ]; then echo "TARGET $t 0 0"; continue; fi
  # du -sb over several paths prints one line each plus no total, so sum them.
  bytes=$(du -sb "${existing[@]}" 2>/dev/null | awk '{s += $1} END {print s+0}')
  if [ "$apply" = 1 ]; then
    for p in "${existing[@]}"; do
      # Belt and braces: the payload only ever builds paths under $HOME, but a
      # typo in paths_for should not be able to reach outside it.
      case "$p" in
        "$HOME"/?*) rm -rf -- "$p" ;;
        *) echo "REFUSED $p" >&2 ;;
      esac
    done
  fi
  echo "TARGET $t ${#existing[@]} $bytes"
done

echo "FREE_AFTER $(avail)"
PAYLOAD
}

human() {  # bytes -> 1.2G, without depending on numfmt being present
  awk -v b="${1:-0}" 'BEGIN {
    split("B K M G T", u, " ")
    i = 1
    while (b >= 1024 && i < 5) { b /= 1024; i++ }
    printf (i == 1 ? "%d%s" : "%.1f%s"), b, u[i]
  }'
}

# --- main -------------------------------------------------------------------
action="${1:-}"; shift || usage
[[ "$action" == survey || "$action" == clean ]] || usage

targets=() only_host="" confirm=0
while (( $# )); do
  case "$1" in
    --yes)  confirm=1; shift ;;
    --host) only_host="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    -*) echo "unknown flag $1" >&2; usage ;;
    *)  targets+=("$1"); shift ;;
  esac
done
(( ${#targets[@]} )) || read -r -a targets <<<"$ALL_TARGETS"

for t in "${targets[@]}"; do
  [[ " $ALL_TARGETS " == *" $t "* ]] || { echo "unknown target '$t'" >&2; usage; }
done
[[ "$action" == survey ]] && confirm=0

load_hosts
b64=$(payload | base64 | tr -d '\n')

if (( confirm )); then
  echo "DELETING ${targets[*]} on ${#IDS[@]} host(s). Keeping HF repos: $KEEP"
else
  echo "Dry run (${targets[*]}). Nothing is removed; add --yes to clean."
fi
printf '%-6s %10s' host free
for t in "${targets[@]}"; do printf ' %10s' "$t"; done
printf ' %10s\n' "reclaim"

total=0 done_hosts=0 failed=()
for i in "${!IDS[@]}"; do
  [[ -n "$only_host" && "${IDS[$i]}" != "$only_host" ]] && continue
  out=$($SSH -n $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@${ADDRS[$i]}" \
          "echo $b64 | base64 -d | DISK_APPLY=$confirm DISK_KEEP='$KEEP' DISK_TARGETS='${targets[*]}' bash" \
        2>/dev/null) || { failed+=("${IDS[$i]}"); printf '%-6s %10s\n' "${IDS[$i]}" "unreachable"; continue; }

  # awk, not grep: grep exiting 1 on no match would kill this under errexit.
  free=$(awk '$1=="FREE_AFTER"{v=$2} $1=="FREE"&&v==""{v=$2} END{print v+0}' <<<"$out")
  printf '%-6s %10s' "${IDS[$i]}" "$(human "$free")"
  host_total=0
  for t in "${targets[@]}"; do
    bytes=$(awk -v t="$t" '$1=="TARGET" && $2==t {print $4+0; exit}' <<<"$out")
    bytes=${bytes:-0}
    printf ' %10s' "$( ((bytes)) && human "$bytes" || echo '-' )"
    host_total=$(( host_total + bytes ))
  done
  printf ' %10s\n' "$(human "$host_total")"
  total=$(( total + host_total ))
  done_hosts=$(( done_hosts + 1 ))
done

echo
if (( confirm )); then
  echo "Reclaimed $(human "$total") across $done_hosts host(s)."
else
  echo "$(human "$total") reclaimable on $done_hosts host(s). Re-run with --yes to delete it."
fi
# Unreachable hosts keep whatever they are holding, so say so loudly enough to
# come back to: a silent skip is how a nearly-full node stays nearly full.
(( ${#failed[@]} )) && echo "Unreachable (not cleaned): ${failed[*]}" >&2
(( done_hosts )) || { echo "No host matched${only_host:+ --host $only_host}." >&2; exit 1; }
exit 0
