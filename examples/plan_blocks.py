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
import re
import shutil
from pathlib import Path

# One Qwen3.6 block in fp16: weights (incl. Petals' 1% metadata eps) plus the
# per-block cache pool at attn_cache_tokens=4096.
GIB_PER_BLOCK = 1.603
# Server._choose_num_blocks() reserves this for rpc_backward, proportional to hidden_size.
AUTOGRAD_GIB = 0.286
# CUDA context plus activation peaks, which _choose_num_blocks() does NOT account for.
# The eager 256-expert MoE needs real room here; this is what the first run OOMed on.
HEADROOM_GIB = 1.5

# Worst case over every contiguous window, from the model's real 26-shard index.
# Servers download whole shard files, so this is what the Hub cache must hold.
WORST_CASE_DISK_GB = {
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
DISK_MARGIN_GB = 5.0  # leave room for logs, the venv and the OS


def blocks_from_vram(free_gib):
    return max(0, math.floor((free_gib - AUTOGRAD_GIB - HEADROOM_GIB) / GIB_PER_BLOCK))


def blocks_from_disk(free_gb, cap_gb):
    budget = min(free_gb - DISK_MARGIN_GB, cap_gb)
    allowed = [n for n, need in WORST_CASE_DISK_GB.items() if need <= budget]
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
    parser.add_argument("--num-layers", type=int, default=40)
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="ceiling per host; sizing to the last layer that fits leaves no\nroom for activation peaks, and fewer layers per host also means less to download",
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    limits = {}
    for node, vram, disk in parse(args.state_dir):
        if vram is None or disk is None:
            limits[node] = (None, None, None)
            continue
        by_vram = blocks_from_vram(vram)
        by_disk = blocks_from_disk(disk, args.max_disk_gb)
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
    print(
        f"\ntotal layer slots: {total} for {args.num_layers} layers " f"({total / args.num_layers:.1f}x coverage)"
        if total
        else ""
    )
    if total < args.num_layers:
        print(f"NOT ENOUGH: {args.num_layers - total} layer slots short. Free GPU memory or add hosts.")
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
