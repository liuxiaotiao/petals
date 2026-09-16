"""Measure what this swarm delivers: time to first token, decode speed, and whether it scales.

Petals is a pipeline, not a replica set. One request's speed is the sum of its hops, so extra
servers do not make a single request faster -- what they buy is running more requests at once.
A single number hides that, so this reports one session and N sessions side by side.

Timing comes from two plain generate() calls per session, because on this adapter the richer
options do not return: a per-token streamer hangs, and so does the generate(max_new_tokens=1)
loop that would exercise Petals' session-resume path. Both were bisected against the working
client -- same prompt, same route, same dtype -- and only those differences mattered. So this
stays on exactly the call the client makes, and infers decode speed from the gap between a
one-token run and an N-token one.

  bash examples/qwen_cluster.sh bench --concurrency 1 4
"""
import argparse
import logging
import math
import os
import statistics
import threading
from time import perf_counter

import torch
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def say(*parts, end="\n"):
    """Print and flush. Over ssh stdout is a pipe, so block buffering makes a slow run look
    identical to a hung one until it finally exits."""
    print(*parts, end=end, flush=True)


class RouteRecorder(logging.Handler):
    """Keep the 'Route found' lines: a slow run is only interpretable with the path taken."""

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


def one_session(model, prompt_ids, new_tokens, out, index, trace=False):
    """Time one session with two plain generate() calls and nothing else.

    The obvious design attaches a streamer and timestamps every token. On this adapter that
    call never returns -- same prompt, same route, same dtype, only the streamer differs --
    so this measures the way the working client calls the model and accepts coarser numbers:
    one call for the first token, one for the whole sequence, decode inferred from the gap.
    A benchmark that runs on the proven path beats a richer one that hangs.
    """
    try:
        with torch.inference_mode():
            start = perf_counter()
            model.generate(prompt_ids, max_new_tokens=1, do_sample=False)
            ttft = perf_counter() - start
            if trace:
                say(f"  first token in {ttft:.1f}s, now generating {new_tokens} ...")

            start = perf_counter()
            model.generate(prompt_ids, max_new_tokens=new_tokens, do_sample=False)
            total = perf_counter() - start
    except Exception as error:  # a failed session must not take the whole run down
        out[index] = ("error", repr(error)[:200])
        return

    if new_tokens < 2:
        out[index] = ("error", "need at least 2 new tokens to separate decode from prefill")
        return
    # The second call pays the same prefill, so the difference is the extra decode steps.
    per_token = max(1e-6, (total - ttft) / (new_tokens - 1))
    out[index] = ("ok", (ttft, [per_token] * (new_tokens - 1)))


def arm_watchdog(seconds):
    """Force an exit if nothing comes back.

    Needed for the inline path, where there is no thread to join with a timeout. Blunt, but a
    benchmark that cannot end is the failure mode this script has already had twice.
    """

    def fire():
        say(f"\nTIMED OUT after {seconds:.0f}s with no result; exiting.")
        os._exit(3)

    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    return timer


def run(model, prompt_ids, new_tokens, concurrency, timeout, inline=False):
    """Run `concurrency` sessions and summarize them.

    Every session goes on a worker thread, including a single one. Running one inline looked
    tidier and cost the only thing enforcing --timeout -- join(timeout) IS the timeout, and a
    benchmark that cannot give up is worse than one that reports a failure. Threading was also
    suspected of causing a hang and is not: the hang reproduced identically inline.
    """
    out = [None] * concurrency

    # Whether generate() runs on this thread or a worker one is the last untested difference
    # against the client that works. Keeping both paths makes it an A/B rather than an
    # argument, and inline is the one the client itself uses.
    if inline and concurrency == 1:
        timer = arm_watchdog(timeout)
        wall_start = perf_counter()
        one_session(model, prompt_ids, new_tokens, out, 0, trace=True)
        wall = perf_counter() - wall_start
        timer.cancel()
        return summarize(out, wall, concurrency)

    threads = [
        threading.Thread(
            target=one_session,
            args=(model, prompt_ids, new_tokens, out, i),
            kwargs={"trace": concurrency == 1},
            daemon=True,
        )
        for i in range(concurrency)
    ]
    wall_start = perf_counter()
    for thread in threads:
        thread.start()

    deadline = perf_counter() + timeout
    while perf_counter() < deadline and any(thread.is_alive() for thread in threads):
        # Say something while waiting: an alive-but-slow run and a wedged one look identical
        # otherwise, and that difference decides whether to wait or go look at the servers.
        # Wait on a thread that is still running; joining an already-finished one
        # returns instantly and turns this into a busy loop.
        alive = next((t for t in threads if t.is_alive()), None)
        if alive is None:
            break
        alive.join(min(15, max(0.1, deadline - perf_counter())))
        if any(thread.is_alive() for thread in threads):
            done = sum(1 for r in out if r is not None)
            say(
                f"  ... {timeout - (deadline - perf_counter()):.0f}s elapsed," f" {done}/{concurrency} finished",
                end="\r",
            )
    wall = perf_counter() - wall_start
    return summarize(out, wall, concurrency)


def summarize(out, wall, concurrency):
    good = [payload for status, payload in (r for r in out if r) if status == "ok"]
    errors = [payload for status, payload in (r for r in out if r) if status == "error"]
    unfinished = sum(1 for r in out if r is None)
    if not good:
        return {"failed": True, "errors": errors, "unfinished": unfinished}

    first_tokens = [ttft for ttft, _ in good]
    all_steps = [step for _, steps in good for step in steps]
    if not all_steps:
        return {"failed": True, "errors": ["no decode steps recorded"], "unfinished": unfinished}
    return {
        "failed": False,
        "ttft_ms": statistics.median(first_tokens) * 1000,
        "median_ms": statistics.median(all_steps) * 1000,
        "per_session_tps": 1 / statistics.median(all_steps),
        # Every token produced over the wall clock that produced them, first tokens included.
        "aggregate_tps": (len(all_steps) + len(first_tokens)) / wall,
        "sessions_ok": len(good),
        "errors": errors,
        "unfinished": unfinished,
    }


def build_prompt(tokenizer, length):
    """Real text, not random ids: a random id can be a special token, and a stray EOS would end
    generation early and quietly halve the sample."""
    filler = "分布式推理把一个大模型的层拆分到多台机器上顺序执行。" * 200
    ids = tokenizer(filler, return_tensors="pt")["input_ids"][:, :length]
    if ids.shape[1] < length:  # tokenizer produced fewer tokens than asked
        ids = ids.repeat(1, math.ceil(length / max(1, ids.shape[1])))[:, :length]
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-peers", nargs="+", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dht-prefix", default=None)
    # float32, not the servers' float16. Only the embeddings, the final norm and the LM
    # head run here, on CPU, and CPU fp16 is emulated in software -- the LM head alone is
    # a hidden x vocab matmul, so fp16 turns "waiting on the swarm" into "waiting on this
    # machine" and a benchmark measures the wrong computer. The working client says the
    # same thing in its own comment; ignoring it cost a night of silent runs.
    parser.add_argument("--torch-dtype", default="float32", choices=sorted(DTYPES))
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--inline",
        action="store_true",
        help="run a single session on the main thread instead of a worker thread",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="seconds to wait per concurrency level before giving up on it",
    )
    args = parser.parse_args()

    recorder = RouteRecorder()
    logging.getLogger().addHandler(recorder)

    say("loading tokenizer and the client-side layers ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model,
        initial_peers=args.initial_peers,
        revision=args.revision,
        dht_prefix=args.dht_prefix,
        torch_dtype=DTYPES[args.torch_dtype],
        max_retries=3,  # the default behaves as unlimited, which hides a failure as a hang
    )
    prompt_ids = build_prompt(tokenizer, args.prompt_tokens)

    say(f"\nmodel {args.model}  servers float16, client-side layers {args.torch_dtype}")
    say(f"prompt {prompt_ids.shape[1]} tokens, {args.new_tokens} generated per session")
    say(f"per-level timeout {args.timeout:.0f}s, single session on {'this' if args.inline else 'a worker'} thread\n")

    for _ in range(args.warmup):
        say("warming up (route discovery and cache allocation on every hop) ...")
        warm = run(model, prompt_ids, min(4, args.new_tokens), 1, args.timeout, args.inline)
        if warm.get("failed"):
            say(f"warmup failed: {warm['errors'] or 'timed out'}")
            say("The swarm is not answering; check 'status' and 'diag' before reading further.")
            return 1

    header = f"{'sessions':>8} {'TTFT ms':>9} {'ms/token':>10}" f" {'tok/s/sess':>11} {'tok/s total':>12}"
    say(header)
    say("-" * len(header))
    baseline = None
    last = None
    for concurrency in args.concurrency:
        say(f"{concurrency:>8}   measuring ...", end="\r")
        result = run(model, prompt_ids, args.new_tokens, concurrency, args.timeout, args.inline)
        if result.get("failed"):
            say(f"{concurrency:>8}   FAILED: {result['errors'] or 'all sessions timed out'}")
            continue
        last = result
        if baseline is None:
            baseline = result["aggregate_tps"]
        note = ""
        if result["unfinished"] or result["errors"]:
            note = f"  ({result['sessions_ok']}/{concurrency} finished)"
        say(
            f"{concurrency:>8} {result['ttft_ms']:>9.0f} {result['median_ms']:>10.0f}"
            f" {result['per_session_tps']:>11.2f}"
            f" {result['aggregate_tps']:>12.2f}{note}"
        )

    if baseline and last and len(args.concurrency) > 1:
        say(
            f"\nscaling vs {args.concurrency[0]} session(s): "
            f"{last['aggregate_tps'] / baseline:.2f}x at {args.concurrency[-1]} sessions"
        )
        say("Well below linear means the pipeline is the limit, not the client: every session")
        say("crosses the same servers, and each hop serves them one batch at a time.")

    routes = sorted(set(recorder.routes))
    if routes:
        say(f"\nroutes used ({len(recorder.routes)} sessions, {len(routes)} distinct):")
        for route in routes[:6]:
            say(f"  {route}")
    return 0


if __name__ == "__main__":
    os._exit(main() or 0)  # threads may be mid-RPC; do not wait on them to exit
