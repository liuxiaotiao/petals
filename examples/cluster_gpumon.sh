#!/usr/bin/env bash
# Sample every host's GPU memory and utilization while a benchmark runs.
#
#   examples/cluster_gpumon.sh /tmp/gpu.csv &     # start, from an interactive shell
#   ... run the benchmark ...
#   kill %1                                        # stop
#   python3 examples/workload_bench.py --report /tmp/wl.*.log --gpu-csv /tmp/gpu.csv
#
# One CSV line per GPU per interval: epoch,host,mem_used_mib,mem_total_mib,util_pct.
# Each host runs `nvidia-smi -l` over one long-lived ssh connection, so sampling costs one
# connection per host for the whole run rather than one per sample. Timestamps come from
# the hosts' clocks, which this cluster keeps within a few ms of each other (see synctime).
#
# Stopping: killing this script kills its ssh connections; each host's nvidia-smi exits
# at its next sample, when it finds nobody listening.
set -u

HOSTS_FILE="${HOSTS_FILE:-task/hosts.txt}"
SSH="${SSH:-ssh}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_PORT="${SSH_PORT:-22}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o ServerAliveInterval=30}"
INTERVAL="${INTERVAL:-5}"

out="${1:?usage: $(basename "$0") OUT.csv   (env: HOSTS_FILE INTERVAL ONLY=\"N01 N05 ...\")}"
[[ -r "$HOSTS_FILE" ]] || { echo "cannot read $HOSTS_FILE (set HOSTS_FILE=)" >&2; exit 2; }

pids=()
stop() {
  trap - INT TERM EXIT
  kill "${pids[@]}" 2>/dev/null
  wait 2>/dev/null
  echo "gpumon: stopped; $(wc -l < "$out" 2>/dev/null || echo 0) samples in $out" >&2
  exit 0
}
trap stop INT TERM EXIT

while read -r id addr _; do
  [[ -z "$id" ]] && continue
  [[ -n "${ONLY:-}" && " $ONLY " != *" $id "* ]] && continue
  # The host formats its own lines, so the ssh process is the one to kill (no local pipeline
  # to orphan) and a line is written whole.
  $SSH -n $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@${addr%%:*}" \
    "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l $INTERVAL |
     while IFS= read -r line; do echo \"\$(date +%s),$id,\${line// /}\"; done" >> "$out" 2>/dev/null &
  pids+=($!)
done < <(sed 's/#.*//' "$HOSTS_FILE")

(( ${#pids[@]} )) || { echo "no hosts selected" >&2; exit 2; }
echo "gpumon: sampling ${#pids[@]} hosts every ${INTERVAL}s into $out (kill $$ to stop)" >&2
wait
