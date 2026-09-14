"""Regenerate tiny numerical fixtures from the unmodified upstream equations.

Usage (from repo root):
  PYTHONPATH=src:tests python tests/make_qwen_reference.py /path/to/modeling_qwen3_5_moe.py

Use Transformers tag v5.5.0. Only imports, decorators, and the checkpointing base
are shimmed for 4.43.1; attention, normalization, routing and MoE math are upstream.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from transformers.activations import ACT2FN

from test_qwen3_5_moe import tiny_config


def main():
    path = Path(sys.argv[1])
    source = path.read_text()
    names = {
        "Qwen3_5MoeTextRotaryEmbedding",
        "Qwen3_5MoeRMSNormGated",
        "apply_mask_to_padding_states",
        "torch_causal_conv1d_update",
        "l2norm",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "Qwen3_5MoeGatedDeltaNet",
        "rotate_half",
        "apply_rotary_pos_emb",
        "repeat_kv",
        "eager_attention_forward",
        "Qwen3_5MoeAttention",
        "Qwen3_5MoeMLP",
        "Qwen3_5MoeExperts",
        "Qwen3_5MoeTopKRouter",
        "Qwen3_5MoeSparseMoeBlock",
        "Qwen3_5MoeRMSNorm",
        "Qwen3_5MoeDecoderLayer",
    }
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes,
        type_ignores=[],
    )
    namespace = dict(
        torch=torch,
        nn=torch.nn,
        F=torch.nn.functional,
        ACT2FN=ACT2FN,
        GradientCheckpointingLayer=torch.nn.Module,
        dynamic_rope_update=lambda f: f,
        maybe_autocast=torch.autocast,
        ROPE_INIT_FUNCTIONS={},
        causal_conv1d_fn=None,
        causal_conv1d_update=None,
        chunk_gated_delta_rule=None,
        fused_recurrent_gated_delta_rule=None,
        FusedRMSNormGated=None,
        is_fast_path_available=False,
        logger=SimpleNamespace(warning_once=lambda *args: None),
        ALL_ATTENTION_FUNCTIONS=SimpleNamespace(get_interface=lambda name, default: default),
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    c = tiny_config()
    c._attn_implementation = "eager"
    rotary = namespace["Qwen3_5MoeTextRotaryEmbedding"](c)
    arrays = {}
    with torch.inference_mode():
        for index in (0, 3):
            torch.manual_seed(700 + index)
            block = namespace["Qwen3_5MoeDecoderLayer"](c, index).eval()
            for name, p in block.named_parameters():
                p.uniform_(-0.15, 0.15)
                arrays[f"{index}/weight/{name}"] = p.numpy().copy()
            for length in (7, 67):
                x = torch.randn(2, length, c.hidden_size)
                position = torch.arange(length)[None]
                mask = torch.full((length, length), torch.finfo(x.dtype).min).triu(1)[None, None]
                out = block(x, position_embeddings=rotary(x, position), attention_mask=mask if index == 3 else None)
                arrays[f"{index}/{length}/input"] = x.numpy()
                arrays[f"{index}/{length}/output"] = out.numpy()
    directory = Path(__file__).parent / "data"
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory / "qwen3_5_moe_reference.npz", **arrays)
    (directory / "qwen3_5_moe_reference.json").write_text(
        json.dumps(
            dict(
                source="https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                torch_version=torch.__version__,
                dtype="float32",
                seed="700 + layer_index",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
