"""Turn preflight results into a per-host layer count.

Reads the .preflight lines qwen_cluster.sh left in its state directory and sizes each
host from what it actually has free right now, not from the GPU's nameplate capacity.
Prints hosts.txt lines; --write updates the file in place (keeping a .bak).

  bash examples/qwen_cluster.sh preflight   # produces the inputs
  python examples/plan_blocks.py            # preview
  python examples/plan_blocks.py --write
"""
import argparse
import math
import os
import re
import shutil
from pathlib import Path

# Server._choose_num_blocks() reserves this for rpc_backward, proportional to hidden_size.
# Both supported models are hidden_size 2048, so it is the same number for each.
AUTOGRAD_GIB = 0.286
# CUDA context plus activation peaks, which _choose_num_blocks() does NOT account for.
# The eager MoE needs real room here; this is what the first Qwen3.6 run OOMed on.
HEADROOM_GIB = 1.5
DISK_MARGIN_GB = 5.0  # leave room for logs, the venv and the OS
GIB = 1024**3


def measured_disk(table):
    """Worst case over every contiguous window, read off the model's real shard index."""

    def worst_case_gb(blocks):
        return table.get(blocks, table[max(table)] if blocks > max(table) else None)

    return worst_case_gb


def shard_bound_disk(layer_gb, shard_gb):
    """Upper bound when the shard index has not been tabulated.

    Servers download whole shard files. A contiguous run of `blocks` layers is a byte
    range of blocks*layer_gb laid over shards of shard_gb, which touches at most
    ceil(range / shard) + 1 of them however it happens to be aligned. That is a bound,
    not a measurement, so it reserves a little more disk than a real index would.
    """

    def worst_case_gb(blocks):
        span = blocks * layer_gb
        return (math.ceil(span / shard_gb) + 1) * shard_gb

    return worst_case_gb


class Model:
    def __init__(self, num_layers, weights_gib, cache_per_token, cache_fixed, disk_gb):
        self.num_layers = num_layers
        self.weights_gib = weights_gib  # one block in fp16, including Petals' 1% metadata eps
        self.cache_per_token = cache_per_token  # bytes of attention cache per block, per token
        self.cache_fixed = cache_fixed  # bytes per block that do not depend on the length
        self.disk_gb = disk_gb

    def gib_per_block(self, attn_cache_tokens):
        """Weights plus the cache pool Server.__init__ budgets for one block."""
        return self.weights_gib + (self.cache_per_token * attn_cache_tokens + self.cache_fixed) / GIB


MODELS = {
    # 40 layers, 256 experts, hybrid linear/full attention. The pool per block is the
    # priciest layer type at attn_cache_tokens: a linear layer's history plus its state.
    "Qwen/Qwen3.6-35B-A3B": Model(
        num_layers=40,
        weights_gib=1.585,
        cache_per_token=4096,
        cache_fixed=2_162_696,
        disk_gb=measured_disk(
            {
                1: 6.1,
                2: 10.1,
                3: 10.1,
                4: 12.0,
                5: 15.2,
                6: 15.4,
                7: 16.8,
                8: 20.4,
                9: 21.2,
                10: 21.9,
                11: 25.5,
                12: 26.3,
                13: 26.9,
                14: 30.5,
            }
        ),
    ),
    # 48 layers, 128 experts, full attention everywhere: 623.1M parameters per block
    # (604.2M of them experts), and 2 * 4 kv heads * 128 head_dim * 2 bytes per token.
    # 61,064,245,248 bytes over 16 shards, per the Hub's index metadata.
    "Qwen/Qwen3-30B-A3B": Model(
        num_layers=48,
        weights_gib=623.12e6 * 2 * 1.01 / GIB,
        cache_per_token=2 * 4 * 128 * 2,
        cache_fixed=0,
        disk_gb=shard_bound_disk(layer_gb=623.12e6 * 2 / 1e9, shard_gb=61.064 / 16),
    ),
}


def blocks_from_vram(free_gib, gib_per_block):
    return max(0, math.floor((free_gib - AUTOGRAD_GIB - HEADROOM_GIB) / gib_per_block))


def blocks_from_disk(free_gb, cap_gb, model):
    budget = min(free_gb - DISK_MARGIN_GB, cap_gb)
    allowed = [n for n in range(1, model.num_layers + 1) if (model.disk_gb(n) or float("inf")) <= budget]
    return max(allowed) if allowed else 0


def parse(state_dir):
    """Yield (node, vram_free_gib or None, disk_free_gb or None) from preflight output."""
    for path in sorted(Path(state_dir, "out").glob("*.preflight")):
        line = path.read_text().strip().splitlines()[-1] if path.read_text().strip() else ""
        node = path.name[: -len(".preflight")]
        vram = re.search(r"vram_free=([\d.]+)G", line)
        disk = re.search(r"disk_free=([\d.]+)G", line)
        yield node, (float(vram.group(1)) if vram else None), (float(disk.group(1)) if disk else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default=".qwen-cluster")
    parser.add_argument("--hosts-file", default="task/hosts.txt")
    parser.add_argument("--max-disk-gb", type=float, default=30.0, help="matches MAX_DISK_SPACE")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.6-35B-A3B"),
        choices=sorted(MODELS),
        help="defaults to $MODEL_NAME, the same variable qwen_cluster.sh reads",
    )
    parser.add_argument(
        "--attn-cache-tokens",
        type=int,
        default=int(os.environ.get("ATTN_CACHE_TOKENS", 65536)),
        help="must match ATTN_CACHE_TOKENS on the servers; it is part of each block's VRAM",
    )
    parser.add_argument("--num-layers", type=int, default=None, help="defaults to the model's own layer count")
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="ceiling per host; sizing to the last layer that fits leaves no\nroom for activation peaks, and fewer layers per host also means less to download",
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    model = MODELS[args.model]
    num_layers = args.num_layers or model.num_layers
    gib_per_block = model.gib_per_block(args.attn_cache_tokens)
    print(
        f"{args.model}: {num_layers} layers, {gib_per_block:.2f} GiB per block "
        f"(weights {model.weights_gib:.2f} + cache at {args.attn_cache_tokens} tokens)"
    )

    limits = {}
    for node, vram, disk in parse(args.state_dir):
        if vram is None or disk is None:
            limits[node] = (None, None, None)
            continue
        by_vram = blocks_from_vram(vram, gib_per_block)
        by_disk = blocks_from_disk(disk, args.max_disk_gb, model)
        allowed = min(by_vram, by_disk)
        if args.cap is not None:
            allowed = min(allowed, args.cap)
        limits[node] = (allowed, by_vram, by_disk)

    print(f"{'node':5s} {'blocks':>6s} {'by vram':>8s} {'by disk':>8s}   limited by")
    total = 0
    unknown = []
    for node, (n, by_vram, by_disk) in sorted(limits.items()):
        if n is None:
            unknown.append(node)
            print(f"{node:5s} {'?':>6s} {'?':>8s} {'?':>8s}   preflight did not report free space")
            continue
        if args.cap is not None and n == args.cap and n < min(by_vram, by_disk):
            who = "--cap"
        else:
            who = "vram" if by_vram <= by_disk else "disk"
        note = "  <- cannot serve anything" if n == 0 else ""
        print(f"{node:5s} {n:>6d} {by_vram:>8d} {by_disk:>8d}   {who}{note}")
        total += n
    print(f"\ntotal layer slots: {total} for {num_layers} layers ({total / num_layers:.1f}x coverage)" if total else "")
    if total < num_layers:
        print(f"NOT ENOUGH: {num_layers - total} layer slots short. Free GPU memory or add hosts.")
    if unknown:
        print(f"unknown hosts (left unchanged): {', '.join(unknown)}")

    path = Path(args.hosts_file)
    lines = path.read_text().splitlines()
    out = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            out.append(line)
            continue
        body, _, comment = line.partition("#")
        fields = body.split()
        node, addr = fields[0], fields[1]
        # "blocks=start:end" pins a host to an exact range, which is a deliberate choice
        # (usually to close a gap the rebalancer will not fill). Sizing must not undo it.
        if any(f.startswith("blocks=") and ":" in f for f in fields[2:]):
            out.append(line)
            continue
        n = limits.get(node, (None,))[0]
        if n is None:  # keep whatever was there
            out.append(line)
        elif n == 0:  # cannot serve: comment it out rather than drop it
            out.append(f"# {node}  {addr}   blocks=0  # SKIPPED (no free VRAM) {comment}".rstrip())
        else:
            out.append(f"{node}  {addr}   blocks={n}  #{comment}".rstrip())

    print("\n--- proposed hosts.txt ---")
    print("\n".join(out))
    if args.write:
        shutil.copy(path, str(path) + ".bak")
        path.write_text("\n".join(out) + "\n")
        print(f"\nwritten to {path} (previous version kept as {path}.bak)")


if __name__ == "__main__":
    main()
