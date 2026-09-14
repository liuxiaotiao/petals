from transformers import AutoConfig, AutoTokenizer, Qwen2Tokenizer, Qwen2TokenizerFast

from petals.models.qwen3_5_moe.block import WrappedQwen3_5MoeBlock
from petals.models.qwen3_5_moe.config import DistributedQwen3_5MoeConfig
from petals.models.qwen3_5_moe.model import DistributedQwen3_5MoeForCausalLM
from petals.utils.auto_config import register_model_classes

# Register with the pinned Transformers version before AutoDistributedConfig reads config.json.
AutoConfig.register("qwen3_5_moe", DistributedQwen3_5MoeConfig)
AutoTokenizer.register(DistributedQwen3_5MoeConfig, Qwen2Tokenizer, Qwen2TokenizerFast)
register_model_classes(config=DistributedQwen3_5MoeConfig, model_for_causal_lm=DistributedQwen3_5MoeForCausalLM)

__all__ = ["WrappedQwen3_5MoeBlock", "DistributedQwen3_5MoeConfig", "DistributedQwen3_5MoeForCausalLM"]
