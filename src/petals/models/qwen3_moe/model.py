"""Client for Qwen3-MoE: embeddings, the final norm and the LM head run locally."""
from torch import nn
from transformers import LlamaForCausalLM, LlamaPreTrainedModel

from petals.client.from_pretrained import FromPretrainedMixin
from petals.client.lm_head import LMHead
from petals.client.remote_generation import RemoteGenerationMixin
from petals.client.remote_sequential import RemoteSequential
from petals.models.llama.model import DistributedLlamaModel
from petals.models.qwen3_moe.config import DistributedQwen3MoeConfig
from petals.models.qwen3_moe.ops import Qwen3MoeRMSNorm


class DistributedQwen3MoeModel(DistributedLlamaModel):
    config_class = DistributedQwen3MoeConfig

    def __init__(self, config, *, dht=None):
        # Build the client-side modules directly instead of through LlamaModel.__init__,
        # which would also construct a rotary embedding from Llama's own RoPE fields.
        LlamaPreTrainedModel.__init__(self, config)
        if config.tuning_mode or config.active_adapter:
            raise ValueError("The Qwen3-MoE adapter supports inference without prompt tuning or LoRA")
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = RemoteSequential(config, dht=dht)
        self.requires_grad_(False)


class DistributedQwen3MoeForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, LlamaForCausalLM):
    config_class = DistributedQwen3MoeConfig
    _keys_to_ignore_on_load_missing = [r".*\.position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    def __init__(self, config):
        # Llama's generation/LM-head interface is reusable; none of its layers are instantiated.
        config.pretraining_tp = 1
        LlamaPreTrainedModel.__init__(self, config)
        self.model = DistributedQwen3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config)
        self.post_init()

    @property
    def transformer(self) -> DistributedQwen3MoeModel:  # For compatibility with RemoteGenerationMixin
        return self.model

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def generate(self, *args, **kwargs):
        config = kwargs.get("generation_config", self.generation_config)
        for name in ("num_beams", "num_beam_groups"):
            if kwargs.get(name, getattr(config, name, 1)) != 1:
                raise ValueError("The Qwen3-MoE client supports greedy and sampling generation, not beam search")
        if kwargs.get("assistant_model") is not None or kwargs.get("prompt_lookup_num_tokens") is not None:
            raise ValueError("Speculative generation is not supported by the Qwen3-MoE client")
        return super().generate(*args, **kwargs)
