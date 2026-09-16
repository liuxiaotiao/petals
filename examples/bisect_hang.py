"""Find which argument to generate() stops this swarm from answering.

The client works and the benchmark does not, so something between them is responsible. Trying
one difference per run costs a round trip each time and has already cost a day, so this runs
every variant against one model object in one process and prints a table.

Each variant gets its own alarm: a hang has to be survivable, or the first failure hides the
rest. SIGALRM interrupts the blocking call in the main thread, which a watchdog on another
thread cannot do.

  python examples/bisect_hang.py --initial-peers "$BOOTSTRAP_PEER" --timeout 60
"""
import argparse
import signal
from time import perf_counter

import torch
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM


class Timeout(Exception):
    pass


def run_variant(name, model, ids, new_tokens, seconds):
    def fire(signum, frame):
        raise Timeout()

    previous = signal.signal(signal.SIGALRM, fire)
    signal.alarm(int(seconds))
    start = perf_counter()
    try:
        with torch.inference_mode():
            model.generate(ids, max_new_tokens=new_tokens, do_sample=False)
        outcome = f"ok in {perf_counter() - start:.1f}s"
    except Timeout:
        outcome = f"HUNG (>{seconds:.0f}s)"
    except Exception as error:
        outcome = f"error {type(error).__name__}: {str(error)[:60]}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    print(f"{name:<44} {outcome}", flush=True)
    return outcome.startswith("ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-peers", nargs="+", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model,
        initial_peers=args.initial_peers,
        torch_dtype=torch.float32,
        max_retries=3,
    )

    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "你好"}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    filler = "分布式推理把一个大模型的层拆分到多台机器上顺序执行。" * 40
    plain8 = tokenizer(filler, return_tensors="pt")["input_ids"][:, :8]
    plain128 = tokenizer(filler, return_tensors="pt")["input_ids"][:, :128]

    print(f"\nchat template ids: {chat.shape[1]} tokens, plain ids: 8 and 128\n")
    print(f"{'variant':<44} result")
    print("-" * 70)
    # Ordered so the known-good case runs first: if that fails, the swarm changed and nothing
    # below it means anything.
    run_variant("chat ids,  max_new_tokens=8   (the client)", model, chat, 8, args.timeout)
    run_variant("chat ids,  max_new_tokens=1", model, chat, 1, args.timeout)
    run_variant("plain ids 8,   max_new_tokens=8", model, plain8, 8, args.timeout)
    run_variant("plain ids 8,   max_new_tokens=1", model, plain8, 1, args.timeout)
    run_variant("plain ids 128, max_new_tokens=8", model, plain128, 8, args.timeout)
    print("\nThe first row that hangs names the culprit; rows above it are the controls.")


if __name__ == "__main__":
    main()
