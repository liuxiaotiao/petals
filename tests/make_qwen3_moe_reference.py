"""Regenerate tiny numerical fixtures from the unmodified upstream Qwen3-MoE equations.

Usage (from repo root):
  python tests/make_qwen3_moe_reference.py /path/to/transformers-4.51.0/models/qwen3_moe/modeling_qwen3_moe.py

Use Transformers tag v4.51.0: it is the release that introduced Qwen3-MoE and the
version Qwen3-30B-A3B's own config.json names, so its parameter layout is the one
the Hub checkpoint uses -- one module per expert, which is what this adapter keeps.
Only imports, decorators and annotations are shimmed; attention, normalization,
routing and MoE math run exactly as upstream wrote them.

The fixture is committed so the test suite can check the port on the pinned
Transformers 4.43.1, which predates Qwen3 and cannot build the reference itself.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# head_dim is deliberately not hidden_size // num_attention_heads: Qwen3-30B-A3B has
# hidden_size 2048 with 32 heads of 128, and code that assumes otherwise halves the cache.
TINY_QWEN3_MOE = dict(
    hidden_size=64,
    vocab_size=128,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=128,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    norm_topk_prob=True,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    max_position_embeddings=512,
    tie_word_embeddings=False,
    bos_token_id=1,
    eos_token_id=2,
)

UPSTREAM_NAMES = {
    "rotate_half",
    "apply_rotary_pos_emb",
    "repeat_kv",
    "eager_attention_forward",
    "Qwen3MoeAttention",
    "Qwen3MoeMLP",
    "Qwen3MoeSparseMoeBlock",
    "Qwen3MoeRMSNorm",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeRotaryEmbedding",
}


def default_rope_parameters(config, device=None, **kwargs):
    """Transformers' own default RoPE init, inlined.

    4.43 keeps it in ROPE_INIT_FUNCTIONS["default"]; 5.x moved it onto the rotary class
    and reads rope_parameters instead of rope_theta, so neither key is reliably there to
    borrow. The formula itself has not changed, and tests/test_qwen3_moe.py cross-checks
    the resulting angles against the adapter's own table.
    """
    dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, 1.0


def upstream_namespace(path: Path) -> dict:
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in UPSTREAM_NAMES:
            node.decorator_list = []
            nodes.append(node)
    missing = UPSTREAM_NAMES - {node.name for node in nodes}
    assert not missing, f"{path} does not define {sorted(missing)}"
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes,
        type_ignores=[],
    )
    namespace = dict(
        torch=torch,
        nn=torch.nn,
        F=torch.nn.functional,
        ACT2FN=ACT2FN,
        ROPE_INIT_FUNCTIONS={**ROPE_INIT_FUNCTIONS, "default": default_rope_parameters},
        dynamic_rope_update=lambda f: f,
        logger=SimpleNamespace(warning_once=lambda *args: None),
        ALL_ATTENTION_FUNCTIONS={},
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def upstream_config():
    return SimpleNamespace(
        **TINY_QWEN3_MOE,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        rope_scaling=None,
        sliding_window=None,
        use_sliding_window=False,
        max_window_layers=TINY_QWEN3_MOE["num_hidden_layers"],
        _attn_implementation="eager",
    )


def main():
    path = Path(sys.argv[1])
    namespace = upstream_namespace(path)
    config = upstream_config()
    rotary = namespace["Qwen3MoeRotaryEmbedding"](config)

    arrays = {}
    with torch.inference_mode():
        torch.manual_seed(700)
        block = namespace["Qwen3MoeDecoderLayer"](config, 0).eval()
        for name, parameter in block.named_parameters():
            parameter.uniform_(-0.15, 0.15)
            arrays[f"weight/{name}"] = parameter.numpy().copy()
        for length in (7, 67):
            x = torch.randn(2, length, config.hidden_size)
            position = torch.arange(length)[None]
            mask = torch.full((length, length), torch.finfo(x.dtype).min).triu(1)[None, None]
            out = block(x, attention_mask=mask, position_embeddings=rotary(x, position))
            arrays[f"{length}/input"] = x.numpy()
            arrays[f"{length}/output"] = (out[0] if isinstance(out, tuple) else out).numpy()

    directory = Path(__file__).parent / "data"
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory / "qwen3_moe_reference.npz", **arrays)
    (directory / "qwen3_moe_reference.json").write_text(
        json.dumps(
            dict(
                source="https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                torch_version=torch.__version__,
                dtype="float32",
                seed=700,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
