"""Split a prompt set across chains: every dataset evenly, datasets interleaved.

  python3 examples/workload_split.py task/workload/prompts.jsonl            # chains A and B
  python3 examples/workload_split.py task/workload/prompts.jsonl --chains A B C

Writes chain.<TAG>.jsonl next to the input. Within each dataset, prompts are dealt in file
order: the i-th goes to chain i mod N, so every chain gets the same share of every dataset
and no prompt runs twice. Each chain's file then takes one prompt from each dataset in turn
(gsm8k, mbpp, norobots, gsm8k, ...), so a slowdown halfway through a run lands on every
dataset alike instead of on whichever happened to run last.

This is exactly how the 2026-10-05 runs were split: 150 prompts, 75 per chain, 25 of each
dataset per chain.
"""
import argparse
import collections
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompts")
    parser.add_argument("--chains", nargs="+", default=["A", "B"])
    parser.add_argument("--out-dir", default=None, help="default: the directory of the input")
    args = parser.parse_args()

    prompts = [json.loads(line) for line in open(args.prompts, encoding="utf-8") if line.strip()]
    by_dataset = collections.defaultdict(list)
    for prompt in prompts:
        by_dataset[prompt["dataset"]].append(prompt)
    count = len(args.chains)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.prompts))

    seen = set()
    for k, tag in enumerate(args.chains):
        groups = [items[k::count] for _, items in sorted(by_dataset.items())]
        rows = [group[i] for i in range(max(map(len, groups), default=0)) for group in groups if i < len(group)]
        ids = {row["id"] for row in rows}
        if ids & seen:
            sys.exit(f"chain {tag} would repeat prompts already assigned elsewhere")
        seen |= ids
        path = os.path.join(out_dir, f"chain.{tag}.jsonl")
        with open(path, "w", encoding="utf-8") as file:
            file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        counts = collections.Counter(row["dataset"] for row in rows)
        print(f"{path}: {len(rows)} prompts  " + "  ".join(f"{name} {n}" for name, n in sorted(counts.items())))
    if len(seen) != len({p["id"] for p in prompts}):
        sys.exit("some prompts were not assigned to any chain")


if __name__ == "__main__":
    main()
