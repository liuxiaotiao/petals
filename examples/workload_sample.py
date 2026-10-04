"""Draw a fixed, reproducible set of real prompts for workload_bench.py.

  python3 examples/workload_sample.py --out task/workload/prompts.jsonl
  python3 examples/workload_sample.py --per-dataset 50 --datasets gsm8k mbpp lmsys --seed 0

Standard library only, so it runs on the control node or on any host with Hub access.
Rows come from the Hugging Face dataset viewer API, one random offset at a time, so nothing
large is downloaded: lmsys-chat-1m alone is a million conversations.

lmsys/lmsys-chat-1m is gated. Accept its license on the Hub with the account whose token
is in HF_TOKEN, or leave it out with --datasets gsm8k mbpp. A dataset that fails is
reported and skipped; the others are still written.

Each output line is one prompt:
  {"id": "gsm8k/test/17", "dataset": "gsm8k", "messages": [{"role": "user", "content": ...}],
   "reference": ...}
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co/rows"


def fetch_row(dataset, config, split, offset, token, attempts=5):
    query = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split, offset=offset, length=1))
    request = urllib.request.Request(f"{API}?{query}")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404):
                raise  # access problems do not get better by retrying
            if attempt + 1 == attempts:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise
        time.sleep(2**attempt)  # 429 / 5xx / transient network


def gsm8k(row):
    return dict(
        messages=[{"role": "user", "content": row["question"]}],
        reference=row["answer"].split("####")[-1].strip(),
    )


def mbpp(row):
    tests = "\n".join(row["test_list"])
    prompt = (
        f"You are an expert Python programmer, and here is your task: {row['text']} "
        f"Your code should pass these tests:\n\n{tests}"
    )
    return dict(messages=[{"role": "user", "content": prompt}], reference=tests)


def lmsys(row, max_chars):
    """First user turn of the conversation; None to draw again."""
    if any(entry.get("flagged") for entry in row.get("openai_moderation") or []):
        return None  # keep flagged content out of a file that lives in the repo
    first = next((turn for turn in row["conversation"] if turn.get("role") == "user"), None)
    if not first or not first.get("content", "").strip() or len(first["content"]) > max_chars:
        return None
    return dict(
        messages=[{"role": "user", "content": first["content"]}],
        reference=None,
        meta=dict(language=row.get("language"), source_model=row.get("model")),
    )


SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "test", gsm8k),
    "mbpp": ("google-research-datasets/mbpp", "full", "test", mbpp),
    "lmsys": ("lmsys/lmsys-chat-1m", "default", "train", lmsys),
}


def sample(name, count, rng, token, max_chars):
    dataset, config, split, convert = SOURCES[name]
    total = fetch_row(dataset, config, split, 0, token)["num_rows_total"]
    chosen, seen, tries = [], set(), 0
    while len(chosen) < count:
        tries += 1
        if tries > 20 * count:
            raise RuntimeError(f"only {len(chosen)} usable rows after {tries - 1} draws")
        offset = rng.randrange(total)
        if offset in seen:
            continue
        seen.add(offset)
        page = fetch_row(dataset, config, split, offset, token)
        if not page.get("rows") or page["rows"][0].get("truncated_cells"):
            continue  # an oversized row comes back cut short; it would not be the real prompt
        row = page["rows"][0]["row"]
        item = convert(row, max_chars) if name == "lmsys" else convert(row)
        if item is None:
            continue
        chosen.append(dict(id=f"{name}/{split}/{offset}", dataset=name, **item))
        print(f"  {name}: {len(chosen)}/{count}", end="\r", file=sys.stderr, flush=True)
    print(f"  {name}: {count} drawn from {total} rows ({tries} draws)", file=sys.stderr)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="task/workload/prompts.jsonl")
    parser.add_argument("--datasets", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument("--per-dataset", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="lmsys prompts longer than this are drawn again (the runner also caps by tokens)",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    prompts, failed = [], []
    for name in args.datasets:
        try:
            # One generator per dataset: dropping or adding a dataset leaves the others' draws unchanged.
            rng = random.Random(f"{args.seed}/{name}")
            prompts += sample(name, args.per_dataset, rng, token, args.max_chars)
        except urllib.error.HTTPError as error:
            hint = ""
            if name == "lmsys" and error.code in (401, 403):
                hint = (" -- the dataset is gated: accept its terms at "
                        "https://huggingface.co/datasets/lmsys/lmsys-chat-1m and export HF_TOKEN")
            print(f"  {name}: FAILED, HTTP {error.code}{hint}", file=sys.stderr)
            failed.append(name)
        except Exception as error:
            print(f"  {name}: FAILED, {type(error).__name__}: {error}", file=sys.stderr)
            failed.append(name)

    if not prompts:
        sys.exit("nothing was sampled; nothing written")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        for prompt in prompts:
            file.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    print(f"wrote {len(prompts)} prompts to {args.out}" + (f"; skipped {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
