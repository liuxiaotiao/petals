"""Run one shared prompt queue over several chains: whichever chain finishes first takes the next.

  python3 examples/workload_dispatch.py --prompts task/workload/prompts.jsonl --out DIR \\
      --chain A N01 <6 peer IDs> --chain B N11 <6 peer IDs> -- --max-new-tokens 2048

Each chain still runs one request at a time, from a client on its layer-0 node. Every prompt
runs exactly once, on one chain, so a faster chain simply serves more of them and both chains
finish within about one request of each other, instead of the slow chain finishing long
after the fast one sits idle.

How: for each chain this starts `qwen_cluster.sh client --node <node> ... --worker` over ssh
with stdin kept open (CLIENT_STDIN=1). The worker prints "WORKER READY" when it is free; the
dispatcher answers with the next prompt as one JSON line, or closes stdin when the queue is
empty. Prompts travel from here, so nothing has to be deployed for them.

The queue order interleaves the datasets (gsm8k, mbpp, norobots, gsm8k, ...) so that whatever
each chain ends up with is a similar mix. Logs are DIR/wl.<TAG>.log, the same format as a
normal run, so `workload_bench.py --report DIR/wl.*.log` works unchanged; DIR/DISPATCH.txt
records which chain ran which prompt and when each chain finished.

A prompt already handed to a chain that then dies is reported as lost, not retried: a request
that killed one chain would likely do the same to the next. A prompt that could not even be
handed over goes back to the queue.
"""
import argparse
import collections
import json
import os
import signal
import subprocess
import sys
import threading
import time


def interleave(prompts):
    by_dataset = collections.defaultdict(list)
    for prompt in prompts:
        by_dataset[prompt["dataset"]].append(prompt)
    groups = [items for _, items in sorted(by_dataset.items())]
    return [group[i] for i in range(max(map(len, groups), default=0)) for group in groups if i < len(group)]


class Chain:
    def __init__(self, tag, node, peers):
        self.tag, self.node, self.peers = tag, node, peers
        self.proc = None
        self.inflight = None  # the prompt handed over and not yet answered with a REC line
        self.done, self.lost, self.ids = 0, [], []
        self.started = self.finished = None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", default="task/workload/prompts.jsonl")
    parser.add_argument("--out", required=True, help="directory for wl.<TAG>.log and DISPATCH.txt")
    parser.add_argument("--chain", nargs="+", action="append", required=True, metavar="TAG NODE PEER_ID",
                        help="a chain: its tag, the layer-0 node that runs its client, then its peer IDs")
    parser.add_argument("--order", choices=["interleave", "file"], default="interleave")
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts of the queue")
    parser.add_argument("client_args", nargs=argparse.REMAINDER,
                        help="after --: passed to every workload_bench.py worker (e.g. --max-new-tokens 2048)")
    args = parser.parse_args()
    client_args = args.client_args[1:] if args.client_args[:1] == ["--"] else args.client_args

    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8") if line.strip()]
    if len({p["id"] for p in prompts}) != len(prompts):
        sys.exit(f"{args.prompts} has duplicate ids")
    if args.order == "interleave":
        prompts = interleave(prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    queue = collections.deque(prompts)
    total = len(prompts)

    chains = []
    for spec in args.chain:
        if len(spec) < 3:
            sys.exit(f"--chain needs TAG NODE PEER_ID...; got {' '.join(spec)}")
        chains.append(Chain(spec[0], spec[1], spec[2:]))
    os.makedirs(args.out, exist_ok=True)

    lock = threading.Lock()
    began = time.time()

    def progress(chain, what):
        done = sum(c.done for c in chains)
        split = " ".join(f"{c.tag}:{c.done}" for c in chains)
        print(f"{time.strftime('%H:%M:%S')} [{done}/{total}] ({split}) {chain.tag} {what}", flush=True)

    def hand_next(chain):
        """Called when a worker says it is free: give it the next prompt, or tell it we are done."""
        with lock:
            while True:
                item = queue.popleft() if queue else None
                if item is None:
                    try:
                        chain.proc.stdin.close()
                    except OSError:
                        pass
                    return
                try:
                    chain.proc.stdin.write(json.dumps(item, ensure_ascii=False) + "\n")
                    chain.proc.stdin.flush()
                except OSError:
                    queue.appendleft(item)  # never reached the worker: safe to give to another chain
                    return
                chain.inflight = item
                chain.ids.append(item["id"])
                return

    def pump_stdout(chain, log):
        for line in chain.proc.stdout:
            with lock:
                log.write(line)
                log.flush()
            if line.startswith("WORKER READY"):
                if chain.started is None:
                    chain.started = time.time()
                hand_next(chain)
            elif line.startswith("REC "):
                with lock:
                    chain.inflight = None
                    chain.done += 1
                try:
                    rec = json.loads(line[4:])
                    status = rec.get("error") or rec.get("skipped") or (
                        f"{rec['output_tokens']} tokens in {rec['latency']:.0f}s")
                    progress(chain, f"{rec['id']}: {status}")
                except (ValueError, KeyError):
                    progress(chain, "finished a request")

    def pump_stderr(chain, log):
        for line in chain.proc.stderr:
            with lock:
                log.write(line)
                log.flush()

    env = dict(os.environ, CLIENT_SCRIPT="workload_bench.py", CLIENT_STDIN="1")
    threads = []
    logs = []
    for chain in chains:
        log = open(os.path.join(args.out, f"wl.{chain.tag}.log"), "w", encoding="utf-8")
        logs.append(log)
        command = ["bash", "examples/qwen_cluster.sh", "client", "--node", chain.node,
                   "--allowed-servers", *chain.peers, "--tag", chain.tag, "--worker", *client_args]
        chain.proc = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                      bufsize=1)
        print(f"chain {chain.tag}: worker on {chain.node} ({len(chain.peers)} servers)", flush=True)
        for target in (pump_stdout, pump_stderr):
            thread = threading.Thread(target=target, args=(chain, log), daemon=True)
            thread.start()
            threads.append(thread)

    def stop(signum, _frame):
        print(f"signal {signum}: stopping every worker", flush=True)
        for chain in chains:
            if chain.proc.poll() is None:
                chain.proc.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    for chain in chains:
        chain.proc.wait()
        chain.finished = time.time()
        with lock:
            if chain.inflight is not None:
                chain.lost.append(chain.inflight["id"])
                chain.inflight = None
        if chain.proc.returncode:
            print(f"chain {chain.tag}: worker exited with code {chain.proc.returncode}", flush=True)
    for thread in threads:
        thread.join(timeout=5)
    for log in logs:
        log.close()

    lines = [f"prompts: {args.prompts} ({args.order}), {total} queued",
             f"client args: {' '.join(client_args)}", ""]
    for chain in chains:
        span = (chain.finished - chain.started) if chain.started else 0
        lines.append(f"chain {chain.tag} on {chain.node}: {chain.done} done, finished "
                     f"{time.strftime('%H:%M:%S', time.localtime(chain.finished))} "
                     f"({span / 60:.1f} min after its first prompt), exit {chain.proc.returncode}")
        if chain.lost:
            lines.append(f"  lost (handed over, never answered): {' '.join(chain.lost)}")
        lines.append(f"  ran: {' '.join(chain.ids)}")
    finishes = [c.finished for c in chains]
    lines.append("")
    lines.append(f"finish spread between chains: {(max(finishes) - min(finishes)) / 60:.1f} min; "
                 f"total wall {(max(finishes) - began) / 60:.1f} min")
    if queue:
        lines.append(f"NOT RUN ({len(queue)}; every worker had stopped): {' '.join(p['id'] for p in queue)}")
    with open(os.path.join(args.out, "DISPATCH.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    print("\n".join(lines[3:]))
    ran = sum(c.done for c in chains)
    return 0 if ran == total and not any(c.proc.returncode for c in chains) else 1


if __name__ == "__main__":
    sys.exit(main())
