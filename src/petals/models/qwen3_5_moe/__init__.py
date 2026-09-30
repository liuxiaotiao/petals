from transformers import AutoConfig, AutoTokenizer, Qwen2Tokenizer, Qwen2TokenizerFast

from petals.models.qwen3_5_moe.block import WrappedQwen3_5MoeBlock
from petals.models.qwen3_5_moe.config import DistributedQwen3_5MoeConfig
from petals.models.qwen3_5_moe.model import DistributedQwen3_5MoeForCausalLM
from petals.utils.auto_config import register_model_classes

# Register before AutoDistributedConfig reads config.json. exist_ok keeps this working once
# Transformers ships its own qwen3_5_moe, which it does from v5.5; without it the import
# would start raising the day someone upgrades. By keyword because Transformers 5 inserted a
# `tokenizer_class` parameter ahead of these two, so positionally the fast one lands in the
# slow slot.
AutoConfig.register("qwen3_5_moe", DistributedQwen3_5MoeConfig, exist_ok=True)
AutoTokenizer.register(
    DistributedQwen3_5MoeConfig,
    slow_tokenizer_class=Qwen2Tokenizer,
    fast_tokenizer_class=Qwen2TokenizerFast,
    exist_ok=True,
)
register_model_classes(config=DistributedQwen3_5MoeConfig, model_for_causal_lm=DistributedQwen3_5MoeForCausalLM)

__all__ = ["WrappedQwen3_5MoeBlock", "DistributedQwen3_5MoeConfig", "DistributedQwen3_5MoeForCausalLM"]
