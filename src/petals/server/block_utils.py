from typing import List, Optional, Union

import torch
from accelerate import init_empty_weights
from transformers import PretrainedConfig, PreTrainedModel

from petals.models.mixtral.block import WrappedMixtralBlock
from petals.utils.convert_block import QuantType
from petals.utils.misc import get_size_in_bytes


def resolve_block_dtype(config: PretrainedConfig, dtype: Union[str, torch.dtype]) -> torch.dtype:
    """If dtype is "auto", resolves it using BloomConfig. Returns `dtype` intact otherwise."""
    if dtype not in ("auto", None):
        return dtype
    if config.torch_dtype not in ("auto", None, torch.float32):
        # If config specifies float32, we override it to the default dtype below
        return config.torch_dtype
    return torch.bfloat16


def get_block_size(
    config: PretrainedConfig,
    location: str,
    *,
    dtype: Optional[Union[str, torch.dtype]] = None,
    quant_type: QuantType = QuantType.NONE,
    eps: float = 0.01,  # eps accounts for ~1% of metainfo for tensor descriptions, quantization tables, etc.
) -> int:
    if location == "memory":
        assert (
            dtype is not None and quant_type is not None
        ), 'get_block_size(..., location="memory") requires to specify dtype and quant_type for calculations'

    with init_empty_weights(include_buffers=False):
        n_params = max(
            sum(param.numel() for param in get_model_block(config, index).parameters())
            for index in get_distinct_block_indices(config)
        )

    if location == "memory":
        if quant_type == QuantType.NONE:
            dtype = resolve_block_dtype(config, dtype)
            bytes_per_value = get_size_in_bytes(dtype)
        elif quant_type == QuantType.INT8:
            bytes_per_value = 1
        elif quant_type == QuantType.NF4:
            bytes_per_value = 4.25 / 8  # Bitness of NF4 with this config (measured empirically)
        else:
            raise ValueError(f"Unsupported quant_type={quant_type}")
    elif location == "disk":
        dtype = resolve_block_dtype(config, "auto")
        bytes_per_value = get_size_in_bytes(dtype)

    return round(n_params * bytes_per_value * (1 + eps))


def get_distinct_block_indices(config: PretrainedConfig) -> List[int]:
    """Return the first layer index of each distinct block variant.

    Most models build every block from one class with identical shapes, so index 0
    speaks for all of them. Hybrid models interleave variants (e.g. Qwen alternates
    linear and full attention) and need one index per variant, or size and speed
    estimates silently describe whichever variant happens to be first.
    """
    layer_types = getattr(config, "layer_types", None)
    if not layer_types:
        return [0]
    return sorted({layer_types.index(kind) for kind in set(layer_types)})


def get_model_block(config, layer_idx: int = 0):
    """
    The function to create a model block based on the block class
    kwargs argument **only** is necessary for specific classes, like Mixtral.
    They will not be passed to other block constructors.
    """
    if getattr(config, "block_uses_layer_index", False):
        return config.block_class(config, layer_idx)
    if config.block_class == WrappedMixtralBlock:
        config = PreTrainedModel._autoset_attn_implementation(config)
        return config.block_class(config, layer_idx)
    return config.block_class(config)
