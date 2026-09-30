"""Qwen3-30B-A3B and its siblings use the Qwen3 MoE text architecture."""
import os

from hivemind import get_logger
from transformers import PretrainedConfig

from petals.client.config import ClientConfig
from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig
from petals.models.qwen3_moe.block import Qwen3MoeAttention, WrappedQwen3MoeBlock, cache_specs

logger = get_logger(__name__)

# Separates this adapter from swarms serving other Qwen variants, including Qwen3.5.
DHT_PREFIX_SUFFIX = "-petals-qwen3-moe-v1"


def default_dht_prefix(model_name_or_path) -> str:
    """Derive the swarm namespace from the repository name alone.

    Dropping the account prefix lets blocks served from different Hub accounts or
    from local copies of the same checkpoint join one swarm, the same rule Llama
    uses. Dots are replaced because they delimit block indices in DHT keys.
    """
    name = str(model_name_or_path)
    if os.path.isdir(name):
        name = os.path.basename(os.path.abspath(name))
    return name.split("/")[-1].replace(".", "-") + DHT_PREFIX_SUFFIX


class DistributedQwen3MoeConfig(PretrainedConfig, ClientConfig, LMHeadConfig, PTuneConfig):
    model_type = "qwen3_moe"
    block_class = WrappedQwen3MoeBlock
    attn_class = Qwen3MoeAttention
    block_prefix = "model.layers"
    # Every layer is full attention, but the generic Petals cache derives head_dim as
    # hidden_size / num_attention_heads, which is 64 here against a real head_dim of 128.
    # Sizing the cache from this adapter's own cache_specs keeps the shapes honest.
    petals_custom_cache = True
    cache_specs = staticmethod(cache_specs)
    block_uses_layer_index = True
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, **kwargs):
        defaults = dict(
            hidden_size=2048,
            vocab_size=151936,
            num_hidden_layers=48,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            intermediate_size=6144,
            num_experts=128,
            num_experts_per_tok=8,
            moe_intermediate_size=768,
            norm_topk_prob=True,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            attention_bias=False,
            attention_dropout=0.0,
            sliding_window=None,
            use_sliding_window=False,
            max_position_embeddings=40960,
            initializer_range=0.02,
            tie_word_embeddings=False,
            bos_token_id=151643,
            eos_token_id=151645,
            use_cache=True,
        )
        defaults.update(kwargs)
        defaults.pop("model_type", None)
        # Transformers 5 renamed torch_dtype to dtype; accept either, but never drop one silently.
        alias = defaults.pop("dtype", None)
        if alias is not None:
            existing = defaults.setdefault("torch_dtype", alias)
            if str(existing).replace("torch.", "") != str(alias).replace("torch.", ""):
                raise ValueError(f"Conflicting dtype={alias!r} and torch_dtype={existing!r} in the Qwen config")
        defaults.setdefault("torch_dtype", "bfloat16")
        # A checkpoint may describe RoPE three ways: rope_theta alone (Transformers 4),
        # rope_scaling (the old override), or rope_parameters (Transformers 5). Settle it
        # here, before PretrainedConfig gets a chance to rewrite one into another, and then
        # publish a single normalized pair the block can read on either version.
        rope_parameters = dict(defaults.pop("rope_parameters", None) or {})
        rope_scaling = dict(defaults.pop("rope_scaling", None) or {})
        rope_type = rope_parameters.get("rope_type") or rope_scaling.get("rope_type") or "default"
        if rope_type != "default":
            raise ValueError(f"The Qwen3-MoE adapter does not support {rope_type} RoPE, only the default")
        rope_theta = rope_parameters.get("rope_theta") or defaults["rope_theta"]
        defaults["rope_theta"] = rope_theta
        defaults["rope_parameters"] = dict(rope_type="default", rope_theta=rope_theta)
        defaults.setdefault("head_dim", defaults["hidden_size"] // defaults["num_attention_heads"])
        defaults.setdefault("layer_types", ["full_attention"] * defaults["num_hidden_layers"])
        super().__init__(**defaults)

        if self.hidden_act != "silu" or self.attention_dropout != 0:
            raise ValueError("The Qwen3-MoE adapter requires silu and zero attention dropout")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Qwen3-MoE attention head counts must be divisible")
        if self.use_sliding_window and self.sliding_window:
            raise ValueError("The Qwen3-MoE adapter does not support sliding-window attention")
        # A checkpoint that mixes dense and MoE layers has two block sizes, which Petals would
        # silently size from whichever it sees first. Refuse rather than mis-plan the swarm.
        if self.mlp_only_layers or self.decoder_sparse_step != 1:
            raise ValueError("The Qwen3-MoE adapter requires a checkpoint where every layer is MoE")
        if len(self.layer_types) != self.num_hidden_layers or set(self.layer_types) != {"full_attention"}:
            raise ValueError("Every Qwen3-MoE layer is full attention")
        if getattr(self, "quantization_config", None):
            raise ValueError("Use the original BF16 checkpoint; prequantized Qwen checkpoints are not supported")

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, dht_prefix=None, **kwargs):
        if dht_prefix is None and model_name_or_path is not None:
            dht_prefix = default_dht_prefix(model_name_or_path)
            logger.info(f"Using DHT prefix: {dht_prefix}")
        return super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
