"""Measure what this swarm actually delivers: time to first token, decode speed, concurrency.

Petals is a pipeline, not a replica set. One request's speed is the sum of its hops, so adding
servers does not make a single request faster -- what they buy is the ability to run more
requests at once. A single throughput number hides that, so this reports one session and N
sessions side by side and lets you see whether the swarm actually scales.

Per-token times are reported as median and p90 rather than a mean: one 2-second stall while a
hop is re-dialed would drag a mean far from what a user experiences, and the tail is the part
worth knowing about anyway.

  bash examples/qwen_cluster.sh client --node N08 2>/dev/null   # (this script runs the same way)
  python examples/bench_qwen.py --initial-peers "$BOOTSTRAP_PEER" --concurrency 1 4
"""
import argparse
import logging
import math
import statistics
import threading
from time import perf_counter

import torch
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class RouteRecorder(logging.Handler):
    """Keep the 'Route found' lines: a slow run is only interpretable alongside the path taken."""

    def __init__(self):
        super().__init__()
        self.routes = []
        self.lock = threading.Lock()

    def emit(self, record):
        try:
            message = record.getMessage()
        except Exception:
            return
        if "Route found" in message:
            with self.lock:
                self.routes.append(message.split("Route found:", 1)[-1].strip())


def one_session(model, prompt_ids, new_tokens, out, index):
    """Time the first token separately from the rest: they measure different things.

    The first token pays for the prompt's forward pass through every hop plus session setup on
    each server; the ones after it are a single round trip each. Averaging them together would
    make a long prompt look like a slow cluster.
    """
    session_length = prompt_ids.shape[1] + new_tokens + 2
    steps = []
    with model.inference_session(max_length=session_length) as session:
        start = perf_counter()
        model.generate(prompt_ids, max_new_tokens=1, session=session, do_sample=False)
        first = perf_counter() - start

        for _ in range(new_tokens - 1):
            start = perf_counter()
            model.generate(None, max_new_tokens=1, session=session, do_sample=False)
            steps.append(perf_counter() - start)
    out[index] = (first, steps)


def say(*parts, end="\n"):
    """Print and flush. Over ssh stdout is a pipe, so block buffering makes a run that takes
    minutes show nothing at all until it ends, which is indistinguishable from a hang."""
    print(*parts, end=end, flush=True)


def percentile(values, fraction):
    """Nearest-rank: the smallest value at least this fraction of the sample is below."""
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * fraction))
    return ordered[rank - 1]


def run(model, prompt_ids, new_tokens, concurrency):
    out = [None] * concurrency
    threads = [
        threading.Thread(target=one_session, args=(model, prompt_ids, new_tokens, out, i)) for i in range(concurrency)
    ]
    wall_start = perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = perf_counter() - wall_start

    first_tokens = [result[0] for result in out if result]
    all_steps = [step for result in out if result for step in result[1]]
    if not all_steps:
        return None
    return {
        "ttft_ms": statistics.median(first_tokens) * 1000,
        "median_ms": statistics.median(all_steps) * 1000,
        "p90_ms": percentile(all_steps, 0.9) * 1000,
        "per_session_tps": 1 / statistics.median(all_steps),
        # Every token produced, over the wall clock that produced them -- first tokens
        # included, since the wall clock paid for them.
        "aggregate_tps": (len(all_steps) + len(first_tokens)) / wall,
        "wall_s": wall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-peers", nargs="+", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dht-prefix", default=None)
    parser.add_argument("--torch-dtype", default="float16", choices=sorted(DTYPES))
    parser.add_argument("--prompt-tokens", type=int, default=128, help="synthetic prompt length")
    parser.add_argument("--new-tokens", type=int, default=32, help="tokens generated per session")
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 4], help="session counts to measure, in order"
    )
    parser.add_argument("--warmup", type=int, default=1, help="discarded rounds before measuring")
    args = parser.parse_args()

    recorder = RouteRecorder()
    logging.getLogger().addHandler(recorder)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model,
        initial_peers=args.initial_peers,
        revision=args.revision,
        dht_prefix=args.dht_prefix,
        torch_dtype=DTYPES[args.torch_dtype],
    )

    # A synthetic prompt keeps the prompt length exact, which is what the timings are indexed on.
    prompt_ids = torch.randint(0, tokenizer.vocab_size, (1, args.prompt_tokens))

    say(f"\nmodel {args.model}  dtype {args.torch_dtype}")
    say(f"prompt {args.prompt_tokens} tokens, {args.new_tokens} generated per session\n")

    say("warming up (route discovery and cache allocation on every hop) ...")
    for _ in range(args.warmup):
        # The first run pays for route discovery and cache allocation on every hop.
        run(model, prompt_ids, min(4, args.new_tokens), 1)

    header = f"{'sessions':>8} {'TTFT ms':>9} {'median ms':>10} {'p90 ms':>8} {'tok/s/sess':>11} {'tok/s total':>12}"
    say(header)
    say("-" * len(header))
    baseline = None
    result = None
    for concurrency in args.concurrency:
        say(f"{concurrency:>8}   measuring ...", end="\r")
        result = run(model, prompt_ids, args.new_tokens, concurrency)
        if result is None:
            say(f"{concurrency:>8}   no successful session")
            continue
        if baseline is None:
            baseline = result["aggregate_tps"]
        say(
            f"{concurrency:>8} {result['ttft_ms']:>9.0f} {result['median_ms']:>10.0f}"
            f" {result['p90_ms']:>8.0f} {result['per_session_tps']:>11.2f}"
            f" {result['aggregate_tps']:>12.2f}"
        )

    if baseline and len(args.concurrency) > 1:
        say(
            f"\nscaling vs {args.concurrency[0]} session(s): "
            f"{result['aggregate_tps'] / baseline:.2f}x at {args.concurrency[-1]} sessions"
        )
        say("Well below linear means the pipeline, not the client, is the limit:")
        say("every session shares the same servers, and each hop serves them one batch at a time.")

    routes = sorted(set(recorder.routes))
    if routes:
        say(f"\nroutes used ({len(recorder.routes)} sessions, {len(routes)} distinct):")
        for route in routes[:6]:
            say(f"  {route}")


if __name__ == "__main__":
    main()
