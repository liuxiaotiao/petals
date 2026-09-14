"""Qwen3.6-35B-A3B uses the Qwen3.5 MoE text architecture."""
import os

from hivemind import get_logger
from transformers import PretrainedConfig

from petals.client.config import ClientConfig
from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig
from petals.models.qwen3_5_moe.block import QwenAttention, WrappedQwen3_5MoeBlock

logger = get_logger(__name__)

# Separates this experimental text-only adapter from swarms serving other Qwen variants.
DHT_PREFIX_SUFFIX = "-petals-qwen-v1"


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


class DistributedQwen3_5MoeConfig(PretrainedConfig, ClientConfig, LMHeadConfig, PTuneConfig):
    model_type = "qwen3_5_moe"
    block_class = WrappedQwen3_5MoeBlock
    attn_class = QwenAttention
    block_prefix = "model.language_model.layers"
    petals_custom_cache = True
    block_uses_layer_index = True
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, text_config=None, **kwargs):
        defaults = dict(
            hidden_size=2048,
            vocab_size=248320,
            num_hidden_layers=40,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=256,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            num_experts=256,
            num_experts_per_tok=8,
            moe_intermediate_size=512,
            shared_expert_intermediate_size=512,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
            max_position_embeddings=262144,
            initializer_range=0.02,
            tie_word_embeddings=False,
            bos_token_id=248044,
            eos_token_id=248044,
            use_cache=True,
        )
        text = dict(text_config or {})
        text.pop("model_type", None)
        defaults.update(text)
        defaults.update(kwargs)
        defaults.pop("model_type", None)
        # Transformers 5 renamed torch_dtype to dtype; accept either, but never drop one silently.
        alias = defaults.pop("dtype", None)
        if alias is not None:
            existing = defaults.setdefault("torch_dtype", alias)
            if str(existing).replace("torch.", "") != str(alias).replace("torch.", ""):
                raise ValueError(f"Conflicting dtype={alias!r} and torch_dtype={existing!r} in the Qwen config")
        defaults.setdefault("torch_dtype", "bfloat16")
        defaults.setdefault(
            "layer_types",
            [
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(defaults["num_hidden_layers"])
            ],
        )
        defaults.setdefault(
            "rope_parameters",
            dict(
                rope_type="default",
                rope_theta=10000000.0,
                partial_rotary_factor=0.25,
            ),
        )
        defaults["rope_parameters"] = dict(defaults["rope_parameters"])
        # The Hub stores this outside rope_parameters; modern HF migrates it internally.
        defaults["rope_parameters"].setdefault("partial_rotary_factor", defaults.get("partial_rotary_factor", 0.25))
        super().__init__(**defaults)
        # Keep the Hub's nested configuration, while exposing text dimensions to Petals.
        self.text_config = text_config
        if len(self.layer_types) != self.num_hidden_layers or any(
            kind not in ("linear_attention", "full_attention") for kind in self.layer_types
        ):
            raise ValueError("layer_types must specify each Qwen text layer")
        if self.rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("The experimental Qwen adapter only supports default text RoPE")
        if self.hidden_act != "silu" or self.attention_dropout != 0:
            raise ValueError("The Qwen adapter requires silu and zero attention dropout")
        if (
            self.num_attention_heads % self.num_key_value_heads
            or self.linear_num_value_heads % self.linear_num_key_heads
        ):
            raise ValueError("Qwen attention head counts must be divisible")
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
