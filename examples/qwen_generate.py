"""Text-only client for the experimental private Qwen swarm."""
import argparse

import torch
from transformers import AutoTokenizer

from petals import AutoDistributedModelForCausalLM  # Registers the backported Qwen config/tokenizer.


def load_tokenizer(model, revision):
    """Load the tokenizer, falling back to the slow one when the fast parser refuses.

    Qwen3's tokenizer.json is written by a newer `tokenizers` than this fork's pinned
    Transformers 4.43.1 permits (>=0.19,<0.20); the old Rust deserializer rejects it with
    "data did not match any variant of untagged enum ModelWrapper". The Hub repo also ships
    vocab.json and merges.txt, so the pure-Python BPE loads the same vocabulary without that
    parser. Encoding a prompt costs milliseconds against hundreds per generated token, so the
    slow path is not worth engineering around.
    """
    try:
        return AutoTokenizer.from_pretrained(model, revision=revision)
    except Exception as error:
        print(f"fast tokenizer unavailable ({type(error).__name__}), using the slow one", flush=True)
        return AutoTokenizer.from_pretrained(model, revision=revision, use_fast=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-peers", nargs="+", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--revision", default="main")
    # Left unset, this matches the prefix the servers derive from the repo name.
    parser.add_argument("--dht-prefix", default=None)
    parser.add_argument("--prompt", default="请用中文简单介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.model, args.revision)
    # FP32 avoids slow CPU BF16 emulation. Only embeddings, norm and LM head are loaded locally.
    model = AutoDistributedModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        initial_peers=args.initial_peers,
        dht_prefix=args.dht_prefix,
        torch_dtype=torch.float32,
        max_retries=3,
    )
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    if ids.shape[1] + args.max_new_tokens > 2048:
        raise ValueError("This example reserves at most 2048 tokens; shorten the input or change all server limits")
    with torch.inference_mode():
        outputs = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tokenizer.decode(outputs[0, ids.shape[1] :], skip_special_tokens=True))


if __name__ == "__main__":
    main()
