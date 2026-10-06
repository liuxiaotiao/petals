#!/usr/bin/env bash
# One file to bring the cluster back to a recorded run and repeat it, or to tear it down
# again to free disk. Defaults to the standard setting, r5 (docs/experiments/2026-10-06-*.md).
#
#   bash examples/workload_repro.sh all            # up + check + run: everything, unattended
#   bash examples/workload_repro.sh up             # start servers at the recorded placement and
#                                                  #   model revision; wait until every one serves
#   bash examples/workload_repro.sh check          # verify placement and prompts, run nothing
#   bash examples/workload_repro.sh run [NAME]     # repeat the recorded run (default name:
#                                                  #   replay_<date>_<time>)
#   bash examples/workload_repro.sh smoke          # same, but 6 prompts of 64 tokens
#   bash examples/workload_repro.sh down [--yes]   # stop servers and delete the model weights
#   bash examples/workload_repro.sh status
#
# Another recorded run: RUN_DIR=~/wl-results/2026-10-05/r4_2048_norobots bash examples/workload_repro.sh all
#
# Run it from anywhere; it changes to the repo itself. Use tmux: 'up' downloads ~28 GB per
# node and 'run' takes about 2.5 hours.
#
# What 'up' does, in order:
#   1. loads ~/petals-env.sh and exports MODEL_REVISION from RUN_DIR/settings/MODEL_REVISION, so
#      the servers download the same weights as the recorded run, not whatever is newest
#   2. puts RUN_DIR's hosts.txt back (the old one is kept as hosts.txt.bak-<time>)
#   3. 'service install' (rewrites each node's server.env: placement + revision) and
#      'service restart'
#   4. polls 'status' every minute (download progress per node) until all hosts serve, none is
#      still joining and the swarm is usable; gives up after WAIT_HOURS (default 6)
set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo" || exit 2
RUN_DIR="${RUN_DIR:-$HOME/wl-results/2026-10-06/r5_shared_2048}"
ENV_FILE="${ENV_FILE:-$HOME/petals-env.sh}"
HOSTS_FILE="${HOSTS_FILE:-task/hosts.txt}"
WAIT_HOURS="${WAIT_HOURS:-6}"
POLL_S="${POLL_S:-60}"
QC=(bash examples/qwen_cluster.sh)

say() { printf '\n== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

load_env() {
  [[ -r "$ENV_FILE" ]] || die "$ENV_FILE not found"
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  [[ -n "${MODEL_NAME:-}" ]] || die "MODEL_NAME is not set by $ENV_FILE"
  [[ -d "$RUN_DIR/settings" ]] || die "$RUN_DIR has no settings/ snapshot"
  if [[ -s "$RUN_DIR/settings/MODEL_REVISION" ]]; then
    MODEL_REVISION=$(tr -d ' \n' < "$RUN_DIR/settings/MODEL_REVISION")
    [[ "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]] || die "$RUN_DIR/settings/MODEL_REVISION is not a commit hash"
    export MODEL_REVISION
  else
    echo "warning: $RUN_DIR/settings/MODEL_REVISION missing; servers will fetch the newest weights" >&2
  fi
  echo "run: $RUN_DIR"
  echo "model: $MODEL_NAME @ ${MODEL_REVISION:-(newest)}"
}

host_count() { sed 's/#.*//' "$HOSTS_FILE" | awk 'NF' | wc -l; }

swarm_ready() {  # prints a one-line progress summary; succeeds once every host serves
  local out online joining usable
  out=$("${QC[@]}" status 2>&1)
  online=$(sed -n 's/.* \([0-9][0-9]*\) server(s) online.*/\1/p' <<<"$out" | tail -1)
  joining=$(sed -n 's/.* \([0-9][0-9]*\) still joining.*/\1/p' <<<"$out" | tail -1)
  usable=$(grep -c 'The swarm is usable' <<<"$out")
  # Per-node cache size and download rate from the table at the top, on one line.
  echo "  online ${online:-0}/$(host_count), joining ${joining:-?}, usable ${usable}:" \
       "$(awk '/^N[0-9][0-9] / {printf "%s %s %s  ", $1, $4, $5}' <<<"$out")"
  [[ "${online:-0}" -ge "$(host_count)" && "${joining:-1}" == 0 && "$usable" -ge 1 ]]
}

cmd_up() {
  load_env
  if ! cmp -s "$RUN_DIR/settings/config/hosts.txt" "$HOSTS_FILE"; then
    local backup="$HOSTS_FILE.bak-$(date +%Y%m%d-%H%M%S)"
    cp "$HOSTS_FILE" "$backup"
    cp "$RUN_DIR/settings/config/hosts.txt" "$HOSTS_FILE"
    say "restored $HOSTS_FILE from the run (previous copy: $backup)"
  else
    say "$HOSTS_FILE already matches the run"
  fi
  say "installing the services (placement and model revision go into each node's server.env)"
  "${QC[@]}" service install || die "service install failed (is the bootstrap DHT up? try: bash examples/qwen_cluster.sh start)"
  say "starting every server; the ones without weights download them now"
  "${QC[@]}" service restart || die "service restart failed"
  say "waiting until all $(host_count) hosts serve (polling every ${POLL_S}s, at most ${WAIT_HOURS}h)"
  local deadline=$((SECONDS + WAIT_HOURS * 3600))
  until swarm_ready; do
    (( SECONDS > deadline )) && die "not ready after ${WAIT_HOURS}h; see 'bash examples/qwen_cluster.sh status' and 'diag'"
    sleep "$POLL_S"
  done
  say "swarm is up"
}

cmd_check() {
  load_env
  say "checking placement and prompts against the run (no requests are sent)"
  bash examples/workload_run.sh --replay "$RUN_DIR" --restore-only
}

cmd_run() {
  load_env
  local name="${1:-replay_$(date +%m%d_%H%M)}"; shift || true
  say "repeating the run as '$name'"
  bash examples/workload_run.sh --replay "$RUN_DIR" --name "$name" "$@"
}

cmd_down() {
  load_env
  local yes=0; [[ "${1:-}" == --yes ]] && yes=1
  say "stopping every server (systemd, so nothing restarts them)"
  "${QC[@]}" service stop
  local up
  up=$("${QC[@]}" status 2>&1 | awk '/^N[0-9][0-9] / && $4 != "DOWN" {print $1}' | xargs)
  [[ -z "$up" ]] || die "still running on: $up -- not deleting anything"
  say "weights that would be deleted"
  "${QC[@]}" purge
  if (( ! yes )); then
    read -r -p "delete these weights on every node? type yes: " answer
    [[ "$answer" == yes ]] || { echo "nothing deleted"; return 0; }
  fi
  "${QC[@]}" purge --yes
  say "done; to bring it back: bash examples/workload_repro.sh up"
}

action="${1:-}"; shift || true
case "$action" in
  up) cmd_up ;;
  check) cmd_check ;;
  run) cmd_run "$@" ;;
  smoke) cmd_run "smoke_$(date +%m%d_%H%M)" --max-new-tokens 64 --limit 6 ;;
  all) cmd_up && cmd_check && cmd_run "$@" ;;
  down) cmd_down "$@" ;;
  status) load_env; "${QC[@]}" status ;;
  *) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
