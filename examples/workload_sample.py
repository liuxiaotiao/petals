"""Draw a fixed, reproducible set of real prompts for workload_bench.py.

  python3 examples/workload_sample.py --out task/workload/prompts.jsonl
  python3 examples/workload_sample.py --per-dataset 50 --datasets gsm8k mbpp norobots --seed 0
  python3 examples/workload_sample.py --datasets lmsys --keep   # redo one, keep the others

Standard library only, so it runs on the control node or on any host with Hub access.
Rows come from the Hugging Face dataset viewer API, so nothing large is downloaded. Small
splits (gsm8k, mbpp) are read whole in pages of 100 and sampled locally; lmsys-chat-1m, a
million conversations, is sampled one random row per request. Requests are paced
(--pause) and 429s wait as long as the API asks, because it does rate-limit.

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


class Fetcher:
    """Viewer API pages, paced, with 429 handled the way the API asks.

    The API rate-limits per client: drawing ~50 rows back to back was enough to get 429 on
    the second dataset. So every request waits --pause after the previous one, and a 429
    waits for Retry-After (or a backoff up to a minute) instead of giving up after 15 s.
    """

    def __init__(self, token, pause, attempts=8):
        self.token, self.pause, self.attempts, self.last = token, pause, attempts, 0.0

    def __call__(self, dataset, config, split, offset, length=1):
        query = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split, offset=offset, length=length))
        request = urllib.request.Request(f"{API}?{query}")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        for attempt in range(self.attempts):
            time.sleep(max(0.0, self.last + self.pause - time.monotonic()))
            self.last = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt + 1 == self.attempts:
                    raise  # 401/403/404 are access problems: retrying does not help
                wait = retry_after(error) or min(60, 4 * 2**attempt)
                reason = f"HTTP {error.code}"
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt + 1 == self.attempts:
                    raise
                wait, reason = min(60, 4 * 2**attempt), type(error).__name__
            print(f"\n  {dataset}: {reason}, waiting {wait:.0f}s ({attempt + 1}/{self.attempts - 1})",
                  file=sys.stderr, flush=True)
            time.sleep(wait)


def retry_after(error):
    try:
        return min(300, float(error.headers.get("Retry-After")))
    except (TypeError, ValueError, AttributeError):
        return None


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


def norobots(row, max_chars):
    """Single-turn No Robots conversations only: one user message, one assistant answer
    (a system message, when present, is kept as part of the prompt). None to skip."""
    messages = row.get("messages") or []
    turns = [m for m in messages if m.get("role") != "system"]
    if [m.get("role") for m in turns] != ["user", "assistant"]:
        return None  # multi-turn chat, or malformed
    user = turns[0].get("content", "")
    if not user.strip() or len(user) > max_chars:
        return None
    system = [{"role": "system", "content": m["content"]} for m in messages if m.get("role") == "system"][:1]
    return dict(
        messages=system + [{"role": "user", "content": user}],
        reference=turns[1].get("content"),
        meta=dict(category=row.get("category"), has_system=bool(system)),
    )


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
    # Human-written instructions and answers (HuggingFaceH4/no_robots, test split: 500 rows).
    "norobots": ("HuggingFaceH4/no_robots", "default", "test", norobots),
    "lmsys": ("lmsys/lmsys-chat-1m", "default", "train", lmsys),
}


PAGE = 100  # the viewer API's maximum rows per request


TAKES_MAX_CHARS = {"lmsys", "norobots"}
DEFAULT_DATASETS = ["gsm8k", "mbpp", "norobots"]  # lmsys-chat-1m still selectable with --datasets


def sample(name, count, rng, fetch, max_chars, full_scan_rows):
    dataset, config, split, convert = SOURCES[name]
    first = fetch(dataset, config, split, 0, PAGE)
    total = first["num_rows_total"]

    def usable(entry):
        if entry.get("truncated_cells"):
            return None  # an oversized row comes back cut short; it would not be the real prompt
        return convert(entry["row"], max_chars) if name in TAKES_MAX_CHARS else convert(entry["row"])

    chosen = []
    if total <= full_scan_rows:
        # Small split (gsm8k test: 1319, mbpp test: 500): read it whole in a few pages and
        # draw locally. 14 requests instead of 50+.
        entries = list(first["rows"])
        while len(entries) < total:
            entries += fetch(dataset, config, split, len(entries), PAGE)["rows"]
            print(f"  {name}: read {len(entries)}/{total}", end="\r", file=sys.stderr, flush=True)
        for entry in rng.sample(entries, len(entries)):
            item = usable(entry)
            if item is not None:
                chosen.append(dict(id=f"{name}/{split}/{entry['row_idx']}", dataset=name, **item))
                if len(chosen) == count:
                    break
        if len(chosen) < count:
            raise RuntimeError(f"only {len(chosen)} usable rows in {total}")
        print(f"  {name}: {count} drawn from {total} rows (read whole split)        ", file=sys.stderr)
        return chosen

    # Large split (lmsys: 1M): one random row per request, so the sample is spread over all of it.
    seen, tries = set(), 0
    while len(chosen) < count:
        tries += 1
        if tries > 20 * count:
            raise RuntimeError(f"only {len(chosen)} usable rows after {tries - 1} draws")
        offset = rng.randrange(total)
        if offset in seen:
            continue
        seen.add(offset)
        rows = fetch(dataset, config, split, offset)["rows"]
        item = usable(rows[0]) if rows else None
        if item is None:
            continue
        chosen.append(dict(id=f"{name}/{split}/{offset}", dataset=name, **item))
        print(f"  {name}: {len(chosen)}/{count}", end="\r", file=sys.stderr, flush=True)
    print(f"  {name}: {count} drawn from {total} rows ({tries} draws)", file=sys.stderr)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="task/workload/prompts.jsonl")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=list(SOURCES))
    parser.add_argument("--per-dataset", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="lmsys / norobots prompts longer than this are drawn again (the runner also caps by tokens)",
    )
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between API requests")
    parser.add_argument("--full-scan-rows", type=int, default=5000,
                        help="splits up to this size are read whole and sampled locally")
    parser.add_argument("--keep", action="store_true",
                        help="keep prompts already in --out for datasets not being sampled now")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("  (no HF_TOKEN: anonymous requests get the lowest rate limit; lmsys, if asked for, will fail)", file=sys.stderr)
    fetch = Fetcher(token, args.pause)
    prompts, failed = [], []
    if args.keep and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as file:
            kept = [json.loads(line) for line in file if line.strip()]
        prompts = [p for p in kept if p["dataset"] not in args.datasets]
        if prompts:
            print(f"  keeping {len(prompts)} prompts from {args.out}: "
                  f"{', '.join(sorted({p['dataset'] for p in prompts}))}", file=sys.stderr)
    for name in args.datasets:
        try:
            # One generator per dataset: dropping or adding a dataset leaves the others' draws unchanged.
            rng = random.Random(f"{args.seed}/{name}")
            prompts += sample(name, args.per_dataset, rng, fetch, args.max_chars, args.full_scan_rows)
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
    order = {name: i for i, name in enumerate(SOURCES)}
    prompts.sort(key=lambda p: order.get(p["dataset"], len(order)))  # stable: draw order kept per dataset
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        for prompt in prompts:
            file.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    print(f"wrote {len(prompts)} prompts to {args.out}" + (f"; skipped {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
