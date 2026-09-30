"""Small deterministic Qwen3-MoE checks; no pretrained weights or public swarm required."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from make_qwen3_moe_reference import TINY_QWEN3_MOE
from transformers import AutoConfig

from petals import AutoDistributedConfig
from petals.models.qwen3_moe.block import WrappedQwen3MoeBlock, cache_specs
from petals.models.qwen3_moe.config import DistributedQwen3MoeConfig, default_dht_prefix
from petals.server.block_utils import get_distinct_block_indices, get_model_block
from petals.server.server import custom_cache_accounting
from petals.utils.misc import get_size_in_bytes

REFERENCE = Path(__file__).parent / "data" / "qwen3_moe_reference.npz"


def tiny_config(**kwargs):
    return DistributedQwen3MoeConfig(**{**TINY_QWEN3_MOE, **kwargs}, dht_prefix="tiny-qwen3-moe")


def reference_block(config):
    """The fixture's weights loaded into this adapter's block, by name and strictly."""
    arrays = np.load(REFERENCE)
    block = get_model_block(config, 0).eval().float()
    block.load_state_dict(
        {name[len("weight/") :]: torch.from_numpy(arrays[name]) for name in arrays.files if "weight/" in name}
    )
    return block, arrays


def allocate(config, batch_size, max_length, dtype=torch.float32):
    return [
        torch.zeros(shape, dtype=spec)
        for shape, spec in cache_specs(config, "full_attention", batch_size, max_length, dtype)
    ]


def test_parameter_names_match_the_hub_checkpoint():
    """Petals loads blocks by exact parameter name, with no conversion step.

    Qwen3-30B-A3B stores one tensor per expert, so this adapter keeps one module per
    expert too. Packing them the way newer Transformers does would make every name here
    miss its shard entry and the server would refuse to load a single block.
    """
    config = tiny_config()
    names = sorted(name for name, _ in get_model_block(config, 0).named_parameters())
    expected = sorted(
        ["input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate.weight"]
        + [f"self_attn.{part}.weight" for part in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
        + [
            f"mlp.experts.{index}.{part}.weight"
            for index in range(config.num_experts)
            for part in ("gate_proj", "up_proj", "down_proj")
        ]
    )
    assert names == expected
    # Buffers would have to be excluded from every checkpoint the server reads.
    assert not list(get_model_block(config, 0).named_buffers())


@pytest.mark.parametrize("length", [7, 67])
@torch.inference_mode()
def test_matches_transformers_reference(length):
    """Fixtures come from unmodified Transformers v4.51.0; see make_qwen3_moe_reference.py."""
    block, arrays = reference_block(tiny_config())
    (output,) = block(torch.from_numpy(arrays[f"{length}/input"]))
    expected = torch.from_numpy(arrays[f"{length}/output"])
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_incremental_decoding_matches_one_pass():
    config = tiny_config()
    block, arrays = reference_block(config)
    x = torch.from_numpy(arrays["67/input"])
    expected = torch.from_numpy(arrays["67/output"])

    cache = allocate(config, x.shape[0], x.shape[1])
    pieces, prefix = [], 0
    for step in (40, 1, 1, 25):  # A prefill, two single tokens, then a second prefill.
        pieces.append(block.inference_with_cache(x[:, prefix : prefix + step], cache, prefix, 16)[0])
        prefix += step
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_rewind_ignores_the_tokens_it_replaces():
    """Unlike the hybrid Qwen3.5 blocks, a rewind here only has to re-slice the K/V."""
    config = tiny_config()
    block, arrays = reference_block(config)
    x = torch.from_numpy(arrays["67/input"])
    expected = torch.from_numpy(arrays["67/output"])

    cache = allocate(config, x.shape[0], x.shape[1])
    block.inference_with_cache(x[:, :30], cache, 0, 16)
    block.inference_with_cache(torch.randn_like(x[:, 30:50]), cache, 30, 16)  # A branch that is then abandoned.
    replayed = block.inference_with_cache(x[:, 30:], cache, 30, 16)[0]
    torch.testing.assert_close(replayed, expected[:, 30:], atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_rotary_table_grows_without_changing_the_result():
    config = tiny_config(max_position_embeddings=1024)
    block, _ = reference_block(config)
    length = 300  # Longer than ROPE_CACHE_MIN_LENGTH, so the table has to grow at least once.
    x = torch.randn(2, length, config.hidden_size)
    expected = block(x)[0]

    cache = allocate(config, 2, length)
    parts = [
        block.inference_with_cache(x[:, :250], cache, 0, 64)[0],
        block.inference_with_cache(x[:, 250:], cache, 250, 64)[0],
    ]
    torch.testing.assert_close(torch.cat(parts, dim=1), expected, atol=2e-6, rtol=2e-5)


def test_cache_is_sized_from_head_dim_not_hidden_size():
    """Qwen3-MoE sets head_dim independently of hidden_size / num_attention_heads.

    Qwen3-30B-A3B is hidden_size 2048 over 32 heads of 128, so the generic Petals cache,
    which derives head_dim as hidden_size // num_attention_heads, would allocate half the
    K/V a session needs. This adapter's own cache_specs is the reason it does not.
    """
    config = tiny_config()
    assert config.head_dim != config.hidden_size // config.num_attention_heads
    specs = cache_specs(config, "full_attention", 3, 11, torch.float16)
    assert [shape for shape, _ in specs] == [(3, config.num_key_value_heads, 11, config.head_dim)] * 2
    assert [dtype for _, dtype in specs] == [torch.float16] * 2

    cache = allocate(config, 2, 12)
    block, arrays = reference_block(config)
    with torch.inference_mode():  # The block must write into exactly these tensors.
        block.inference_with_cache(torch.from_numpy(arrays["7/input"]), cache, 0, 4)
    assert [tuple(t.shape) for t in cache] == [(2, config.num_key_value_heads, 12, config.head_dim)] * 2


@pytest.mark.parametrize("num_blocks", [1, 3, 8])
@pytest.mark.parametrize("length", [1, 17, 2048])
def test_cache_accounting_is_exact(num_blocks, length):
    """What the server advertises has to cover what a session of this length allocates."""
    config = tiny_config(num_hidden_layers=num_blocks)
    indices = list(range(num_blocks))
    fixed, bytes_per_token = custom_cache_accounting(config, indices, torch.float16)

    # No linear-attention state here, so the whole cost is per token and nothing is fixed.
    assert fixed == 0
    per_block = 2 * config.num_key_value_heads * config.head_dim * get_size_in_bytes(torch.float16)
    assert bytes_per_token * 2 == per_block

    budget = fixed + bytes_per_token * 2 * num_blocks * length
    actual = sum(
        np.prod(shape) * get_size_in_bytes(dtype)
        for _ in indices
        for shape, dtype in cache_specs(config, "full_attention", 1, length, torch.float16)
    )
    assert budget == actual


def test_config_registration_and_round_trip(tmp_path):
    config = tiny_config()
    config.save_pretrained(tmp_path)
    reloaded = AutoDistributedConfig.from_pretrained(tmp_path)
    assert isinstance(AutoConfig.from_pretrained(tmp_path), DistributedQwen3MoeConfig)
    assert reloaded.model_type == "qwen3_moe"
    assert reloaded.block_prefix == "model.layers"
    assert reloaded.head_dim == TINY_QWEN3_MOE["head_dim"]
    assert reloaded.num_key_value_groups == config.num_attention_heads // config.num_key_value_heads
    assert reloaded.petals_custom_cache and reloaded.cache_specs is cache_specs

    config.save_pretrained(tmp_path / "Qwen3-30B-A3B-local")
    saved = AutoDistributedConfig.from_pretrained(tmp_path / "Qwen3-30B-A3B-local")
    assert saved.dht_prefix == default_dht_prefix(tmp_path / "Qwen3-30B-A3B-local")
    assert saved.dht_prefix != default_dht_prefix("Qwen/Qwen3.6-35B-A3B")  # Never joins the Qwen3.5 swarm.


def test_rope_theta_survives_either_spelling():
    """Transformers 4 keeps rope_theta flat; Transformers 5 folds it into rope_parameters."""
    flat = tiny_config(rope_theta=777.0)
    assert flat.rope_theta == 777.0 and flat.rope_parameters["rope_theta"] == 777.0
    nested = tiny_config(rope_parameters={"rope_type": "default", "rope_theta": 777.0})
    assert nested.rope_theta == 777.0
    assert get_model_block(flat, 0).self_attn.rope_theta == 777.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rope_scaling={"rope_type": "yarn", "factor": 4.0}),
        dict(rope_parameters={"rope_type": "yarn", "rope_theta": 1e6}),
        dict(use_sliding_window=True, sliding_window=512),
        dict(mlp_only_layers=[0]),  # Two block sizes; Petals would size the swarm from one.
        dict(decoder_sparse_step=2),
        dict(hidden_act="gelu"),
        dict(attention_dropout=0.1),
        dict(num_attention_heads=8, num_key_value_heads=3),
    ],
)
def test_rejects_checkpoints_it_cannot_serve(kwargs):
    with pytest.raises(ValueError):
        tiny_config(**kwargs)


def test_every_block_is_the_same_variant():
    config = tiny_config()
    assert get_distinct_block_indices(config) == [0]
    assert set(config.layer_types) == {"full_attention"}
    assert get_distinct_block_indices(SimpleNamespace(layer_types=["full_attention"] * 4)) == [0]
    assert isinstance(get_model_block(config, 1), WrappedQwen3MoeBlock)
