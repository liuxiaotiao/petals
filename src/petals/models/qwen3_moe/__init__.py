from transformers import AutoConfig, AutoTokenizer, Qwen2Tokenizer, Qwen2TokenizerFast

from petals.models.qwen3_moe.block import WrappedQwen3MoeBlock
from petals.models.qwen3_moe.config import DistributedQwen3MoeConfig
from petals.models.qwen3_moe.model import DistributedQwen3MoeForCausalLM
from petals.utils.auto_config import register_model_classes

# Register before AutoDistributedConfig reads config.json. The pinned Transformers 4.43.1
# predates Qwen3 entirely, while 4.51 and newer ship their own qwen3_moe; exist_ok makes this
# the mapping in either case, so AutoConfig returns the distributed config on every version.
AutoConfig.register("qwen3_moe", DistributedQwen3MoeConfig, exist_ok=True)
# By keyword: Transformers 5 inserted a `tokenizer_class` parameter ahead of these two,
# so passing them positionally would register the fast tokenizer as the slow one.
AutoTokenizer.register(
    DistributedQwen3MoeConfig,
    slow_tokenizer_class=Qwen2Tokenizer,
    fast_tokenizer_class=Qwen2TokenizerFast,
    exist_ok=True,
)
register_model_classes(config=DistributedQwen3MoeConfig, model_for_causal_lm=DistributedQwen3MoeForCausalLM)

__all__ = ["WrappedQwen3MoeBlock", "DistributedQwen3MoeConfig", "DistributedQwen3MoeForCausalLM"]
