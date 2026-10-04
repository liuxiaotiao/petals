"""Replay real prompts one after another through the swarm and report latency, TTFT,
percentiles, throughput, memory and computation.

Run (from a node, via the cluster script, so peers/model/proxy are filled in):
  CLIENT_SCRIPT=workload_bench.py bash examples/qwen_cluster.sh client --node N01 \\
      --prompts task/workload/prompts.jsonl --allowed-servers $A --tag A > /tmp/wl.A.log 2>&1
Report (anywhere with python3; no torch needed):
  python3 examples/workload_bench.py --report /tmp/wl.A.log /tmp/wl.B.log --gpu-csv /tmp/gpu.csv

Requests are strictly sequential: the next prompt is sent the moment the previous answer
ends, so every number is what one user sees on an otherwise idle chain.

Each request runs inside one inference session and asks for one token per generate() call.
That is the Petals-supported way to resume a session (generate(session=...)), and it is
what makes TTFT and the gap between every pair of tokens measurable: a streamer attached to
a single generate() call never returns on this adapter (see bench_qwen.py).

Every finished request is printed as one "REC {json}" line, so the log is the result file
and several logs can be reported together.
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from time import perf_counter, process_time

DTYPE_NAMES = ("float16", "bfloat16", "float32")


def say(*parts):
    print(*parts, flush=True)


def hard_exit(code):
    """Take hivemind's DHT child with us; otherwise ssh waits on it forever (see bench_qwen.py)."""
    import multiprocessing

    children = multiprocessing.active_children()
    for child in children:
        child.terminate()
    for child in children:
        child.join(2)
        if child.is_alive():
            child.kill()
    os._exit(code)


# --------------------------------------------------------------------------- running
def load_tokenizer(model, revision):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model, revision=revision)
    except Exception as error:  # Qwen3's tokenizer.json is newer than the pinned tokenizers
        say(f"fast tokenizer unavailable ({type(error).__name__}), using the slow one")
        return AutoTokenizer.from_pretrained(model, revision=revision, use_fast=False)


def render(tokenizer, messages, thinking):
    """Chat-template the prompt the way a chat client would."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
    except Exception:
        # Qwen3's ChatML, written out, for a tokenizer that cannot render its own template.
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + "<|im_start|>assistant\n" + ("" if thinking else "<think>\n\n</think>\n\n")


def run_one(model, ids, max_new_tokens, eos, pad_id, torch):
    """One request: TTFT, every inter-token gap, the generated token count and why it stopped."""
    with torch.inference_mode(), model.inference_session(max_length=ids.shape[1] + max_new_tokens) as session:
        start = perf_counter()
        out = model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=1, do_sample=False,
            pad_token_id=pad_id, session=session,
        )
        ttft = perf_counter() - start
        token, generated, gaps = int(out[0, -1]), 1, []
        while token not in eos and generated < max_new_tokens:
            step = perf_counter()
            out = model.generate(None, max_new_tokens=1, do_sample=False, pad_token_id=pad_id, session=session)
            gaps.append(perf_counter() - step)
            token, generated = int(out[0, -1]), generated + 1
        latency = perf_counter() - start
    return dict(ttft=ttft, latency=latency, output_tokens=generated, gaps=gaps,
                stop="eos" if token in eos else "length")


def run(args):
    import torch
    from petals import AutoDistributedModelForCausalLM

    dtypes = {name: getattr(torch, name) for name in DTYPE_NAMES}
    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8") if line.strip()]
    if args.shard:
        index, count = map(int, args.shard.split("/"))
        prompts = prompts[index::count]
    if args.limit:
        prompts = prompts[: args.limit]

    say("loading tokenizer and the client-side layers ...")
    tokenizer = load_tokenizer(args.model, args.revision)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model, initial_peers=args.initial_peers, revision=args.revision, dht_prefix=args.dht_prefix,
        torch_dtype=dtypes[args.torch_dtype], max_retries=3, allowed_servers=args.allowed_servers,
    )
    eos = model.generation_config.eos_token_id
    eos = {token for token in (list(eos) if isinstance(eos, (list, tuple)) else [eos]) + [tokenizer.eos_token_id]
           if token is not None}
    if not eos:
        raise SystemExit("no EOS token id in the generation config or tokenizer")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(eos)
    config = model.config
    kv_bytes_per_token = (
        config.num_hidden_layers * 2 * config.num_key_value_heads
        * getattr(config, "head_dim", config.hidden_size // config.num_attention_heads) * 2  # fp16 on servers
    )
    meta = dict(
        tag=args.tag, model=args.model, prompts=args.prompts, requests=len(prompts),
        max_new_tokens=args.max_new_tokens, max_prompt_tokens=args.max_prompt_tokens, thinking=args.thinking,
        allowed_servers=["…" + peer[-6:] for peer in args.allowed_servers or []],
        kv_bytes_per_token=kv_bytes_per_token, active_params=args.active_params,
        routing=os.environ.get("PETALS_INFERENCE_ROUTING", "min_latency"),
    )
    say("RUN " + json.dumps(meta, ensure_ascii=False))

    say("warming up (route discovery and cache allocation on every hop; not measured) ...")
    warm = torch.tensor([tokenizer.encode(render(tokenizer, [{"role": "user", "content": "Hi"}], args.thinking))])
    run_one(model, warm, 2, eos, pad_id, torch)

    records, started = [], time.time()
    for number, prompt in enumerate(prompts, 1):
        ids = torch.tensor([tokenizer.encode(render(tokenizer, prompt["messages"], args.thinking))])
        record = dict(tag=args.tag, id=prompt["id"], dataset=prompt["dataset"], prompt_tokens=ids.shape[1],
                      submitted=round(time.time(), 3))
        if ids.shape[1] > args.max_prompt_tokens:
            record["skipped"] = f"prompt {ids.shape[1]} > {args.max_prompt_tokens} tokens"
        else:
            # A request that never returns would stall the whole run silently; give up loudly.
            watchdog = threading.Timer(args.request_timeout, lambda: (
                say(f"TIMEOUT: {prompt['id']} not finished after {args.request_timeout:.0f}s; stopping"),
                hard_exit(3)))
            watchdog.daemon = True
            watchdog.start()
            cpu = process_time()
            try:
                record.update(run_one(model, ids, args.max_new_tokens, eos, pad_id, torch))
            except Exception as error:  # one failed request must not end the run
                record["error"] = repr(error)[:300]
            finally:
                watchdog.cancel()
            record["client_cpu_s"] = round(process_time() - cpu, 3)
        for key in ("ttft", "latency"):
            if key in record:
                record[key] = round(record[key], 4)
        if "gaps" in record:
            record["gaps"] = [round(gap, 4) for gap in record["gaps"]]
        say("REC " + json.dumps(record, ensure_ascii=False))
        records.append(record)
        status = record.get("error") or record.get("skipped") or (
            f"TTFT {record['ttft']:.2f}s, {record['output_tokens']} tokens in {record['latency']:.1f}s")
        say(f"[{number}/{len(prompts)}] {prompt['id']}: {status}")

    try:
        import resource

        rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux
    except Exception:
        rss_mib = None
    end = dict(tag=args.tag, wall_s=round(time.time() - started, 1), client_peak_rss_mib=rss_mib)
    say("END " + json.dumps(end))
    summarize([meta], records, [end], None)


# --------------------------------------------------------------------------- reporting
def pct(values, p):
    """Nearest-rank percentile: the smallest value with at least p% of the sample at or below it."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def load(paths):
    metas, recs, ends = [], [], []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as file:
            for line in file.read().replace("\r", "\n").splitlines():
                for prefix, sink in (("RUN ", metas), ("REC ", recs), ("END ", ends)):
                    if line.startswith(prefix):
                        sink.append(json.loads(line[len(prefix):]))
    return metas, recs, ends


def gpu_summary(path):
    """CSV lines: epoch,node,mem_used_mib,mem_total_mib,util_pct (from the gpumon loop)."""
    nodes = {}
    for line in open(path, encoding="utf-8"):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            _, node, used, total, util = parts[0], parts[1], float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        stats = nodes.setdefault(node, dict(used=[], total=total, util=[]))
        stats["used"].append(used)
        stats["util"].append(util)
    return nodes


def report(paths, gpu_csv):
    summarize(*load(paths), gpu_csv)


def summarize(metas, recs, ends, gpu_csv):
    if not recs:
        say("no REC lines found")
        return
    meta = metas[0] if metas else {}
    kv_per_token = meta.get("kv_bytes_per_token", 48 * 2 * 4 * 128 * 2)
    active = meta.get("active_params", 3.3e9)

    ok = [r for r in recs if "latency" in r]
    if not ok:
        say(f"{len(recs)} records, none finished: " + (recs[0].get("error") or recs[0].get("skipped") or "?"))
        return
    skipped = [r for r in recs if "skipped" in r]
    errors = [r for r in recs if "error" in r]
    ttft = [r["ttft"] for r in ok]
    lat = [r["latency"] for r in ok]
    tpot = [(r["latency"] - r["ttft"]) / (r["output_tokens"] - 1) for r in ok if r["output_tokens"] > 1]
    gaps = [gap for r in ok for gap in r.get("gaps", [])]
    out_tokens = sum(r["output_tokens"] for r in ok)
    in_tokens = sum(r["prompt_tokens"] for r in ok)
    busy = sum(lat)
    tags = sorted({r.get("tag") or "-" for r in recs})

    line = "-" * 72
    say(f"\n{line}\nworkload report: {', '.join(tags)}  |  {len(ok)} ok, {len(skipped)} skipped, {len(errors)} failed")
    if meta:
        say(f"max_new_tokens {meta.get('max_new_tokens')}, thinking {meta.get('thinking')}, "
            f"routing {meta.get('routing')}, chain {' '.join(meta.get('allowed_servers') or ['(any)'])}")
    say(line)
    say(f"{'seconds':30}{'mean':>9}{'p50':>9}{'p90':>9}{'p99':>9}")
    for name, values in (("TTFT", ttft), ("generation latency, end to end", lat),
                         ("time per output token", tpot), ("inter-token gap", gaps)):
        say(f"{name:30}{mean(values):9.3f}{pct(values, 50):9.3f}{pct(values, 90):9.3f}{pct(values, 99):9.3f}")
    say(line)
    say("throughput (sequential: one request in flight per chain)")
    say(f"  output tokens / busy time      {out_tokens / busy:8.2f} tok/s   ({out_tokens} tokens in {busy:.0f}s)")
    say(f"  prompt+output / busy time      {(in_tokens + out_tokens) / busy:8.2f} tok/s")
    say(f"  requests per minute            {60 * len(ok) / busy:8.2f}")
    say(line)
    groups = [("dataset", name, [r for r in ok if r["dataset"] == name]) for name in sorted({r["dataset"] for r in ok})]
    if len(tags) > 1:  # several logs, e.g. one per chain: show whether the chains behave alike
        groups += [("tag", tag, [r for r in ok if (r.get("tag") or "-") == tag]) for tag in tags]
    say("by dataset / tag        n   prompt tok  output tok   TTFT s  latency s  TPOT ms")
    for kind, name, group in groups:
        if not group:
            continue
        group_tpot = [(r["latency"] - r["ttft"]) / (r["output_tokens"] - 1) for r in group if r["output_tokens"] > 1]
        say(f"  {name if kind == 'dataset' else 'tag ' + name:16.16}{len(group):5}{mean([r['prompt_tokens'] for r in group]):12.0f}"
            f"{mean([r['output_tokens'] for r in group]):12.0f}{mean([r['ttft'] for r in group]):9.2f}"
            f"{mean([r['latency'] for r in group]):11.1f}{1000 * mean(group_tpot):9.0f}")
    say(line)
    kv = [(r["prompt_tokens"] + r["output_tokens"]) * kv_per_token / 2**20 for r in ok]
    say("memory")
    say(f"  server KV cache per request    mean {mean(kv):.1f} MiB, max {max(kv):.1f} MiB "
        f"(= tokens x {kv_per_token} B, summed over all layers of the chain)")
    rss = [e["client_peak_rss_mib"] for e in ends if e.get("client_peak_rss_mib")]
    if rss:
        say(f"  client process peak RSS        {max(rss):.0f} MiB")
    cpu = [r["client_cpu_s"] for r in ok if "client_cpu_s" in r]
    flops = [2 * active * (r["prompt_tokens"] + r["output_tokens"]) for r in ok]
    say("computation")
    say(f"  forward FLOPs per request      mean {mean(flops) / 1e12:.2f} TFLOP "
        f"(estimate: 2 x {active / 1e9:.1f}B active params x tokens)")
    say(f"  achieved, over busy time       {sum(flops) / busy / 1e12:.3f} TFLOP/s across the whole chain")
    if cpu:
        say(f"  client CPU per request         mean {mean(cpu):.2f} s ({100 * sum(cpu) / busy:.0f}% of one core)")
    if gpu_csv:
        nodes = gpu_summary(gpu_csv)
        if nodes:
            say(f"  server GPUs (sampled, {gpu_csv})")
            say(f"    {'node':6}{'util mean':>10}{'util max':>10}{'mem max MiB':>13}{'of':>8}")
            for node in sorted(nodes):
                s = nodes[node]
                say(f"    {node:6}{mean(s['util']):9.1f}%{max(s['util']):9.0f}%{max(s['used']):13.0f}{s['total']:8.0f}")
    if errors:
        say(line)
        say(f"failed requests, e.g. {errors[0]['id']}: {errors[0]['error']}")
    say(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", nargs="+", metavar="LOG", help="summarize existing logs instead of running")
    parser.add_argument("--gpu-csv", default=None, help="with --report: GPU samples from the gpumon loop")
    parser.add_argument("--initial-peers", nargs="+")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dht-prefix", default=None)
    parser.add_argument("--allowed-servers", nargs="+", default=None, metavar="PEER_ID")
    parser.add_argument("--prompts", default="task/workload/prompts.jsonl")
    parser.add_argument("--tag", default="", help="label carried into every record, e.g. the chain name")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--thinking", action="store_true", help="let Qwen3 think first (much longer answers)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts (a smoke test)")
    parser.add_argument("--shard", default=None, metavar="I/N", help="every N-th prompt starting at I")
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--torch-dtype", default="float32", choices=DTYPE_NAMES,
                        help="client-side layers; CPU fp16 is emulated and slow (see bench_qwen.py)")
    parser.add_argument("--active-params", type=float, default=3.3e9,
                        help="parameters used per token, for the FLOP estimate (Qwen3-30B-A3B: 3.3B)")
    args = parser.parse_args()

    if args.report:
        report(args.report, args.gpu_csv)
        return 0
    if not args.initial_peers:
        parser.error("--initial-peers is required to run (the cluster script passes it)")
    run(args)
    return 0


if __name__ == "__main__":
    code = main()
    if "--report" in sys.argv:
        sys.exit(code)
    hard_exit(code)
