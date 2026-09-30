"""Eager Qwen3-MoE blocks with an explicit per-session KV cache.

The equations and checkpoint layout follow Transformers v4.51.0, the version that
introduced Qwen3-MoE. See ops.py for the upstream license. Every layer is full
attention, so a session's cache is two plain K/V tensors and nothing is carried on
the shared block between calls.
"""
import torch
from hivemind.utils.logging import get_logger
from torch import nn
from torch.nn import functional as F

from petals.models.qwen3_moe.ops import Qwen3MoeRMSNorm, Qwen3MoeSparseMoeBlock, apply_rotary_pos_emb, repeat_kv

logger = get_logger(__name__)

# Smallest rotary table we bother to build, so short sessions still grow it rarely.
ROPE_CACHE_MIN_LENGTH = 256


class Qwen3MoeAttention(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_key_value_groups
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        # Qwen3 normalizes over the head dimension only, after the projection and before RoPE.
        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, config.rms_norm_eps)
        # Transformers 4 keeps rope_theta on the config; Transformers 5 folds it into
        # rope_parameters and drops the flat attribute. Accept whichever this config has.
        parameters = getattr(config, "rope_parameters", None) or {}
        self.rope_theta = parameters.get("rope_theta") or config.rope_theta
        # Plain dict, not a buffer: buffers land on the meta device during block loading
        # and would have to be excluded from every checkpoint the server reads.
        self._rope_cache = {}

    def rotary(self, prefix, length, device, dtype):
        """Cached cos/sin rows for positions [prefix, prefix + length).

        The table is rebuilt if it was first created under inference mode and is now
        needed with autograd enabled, since inference tensors cannot be saved for backward.
        """
        dim = self.head_dim
        key = (device, dtype, dim)
        cached = self._rope_cache.get(key)
        end = prefix + length
        if cached is not None:
            stale = cached[0].is_inference() and not torch.is_inference_mode_enabled()
            if cached[0].shape[0] >= end and not stale:
                return cached[0][prefix:end][None], cached[1][prefix:end][None]
            size = max(end, 2 * cached[0].shape[0], ROPE_CACHE_MIN_LENGTH)
        else:
            size = max(end, ROPE_CACHE_MIN_LENGTH)
        inv = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        freqs = torch.arange(size, device=device).float()[:, None] * inv[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        self._rope_cache[key] = cached = (emb.cos().to(dtype), emb.sin().to(dtype))
        return cached[0][prefix:end][None], cached[1][prefix:end][None]

    def forward(self, x, past=None):
        batch, length, _ = x.shape
        prefix = 0 if past is None else past[0].shape[2]
        q = self.q_norm(self.q_proj(x).view(batch, length, -1, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(batch, length, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, -1, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *self.rotary(prefix, length, x.device, x.dtype))
        if past is not None:
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        scores = (q @ repeat_kv(k, self.num_key_value_groups).transpose(-1, -2)) * self.head_dim**-0.5
        positions = torch.arange(prefix, prefix + length, device=x.device)
        mask = torch.arange(k.shape[2], device=x.device)[None, :] > positions[:, None]
        scores = scores.masked_fill(mask[None, None], torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        result = (probs @ repeat_kv(v, self.num_key_value_groups)).transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(result), (k, v)


def cache_specs(config, layer_type, batch_size, max_length, dtype):
    """Tensor shapes/dtypes; shared by allocation and the server's memory estimate.

    `layer_type` is accepted for one signature across adapters; every Qwen3-MoE layer
    is full attention. head_dim is read from the config rather than derived from
    hidden_size / num_attention_heads, which this model does not satisfy: 2048 / 32
    is 64, while the real head_dim is 128.
    """
    shape = (batch_size, config.num_key_value_heads, max_length, config.head_dim)
    return [(shape, dtype), (shape, dtype)]


class WrappedQwen3MoeBlock(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = "full_attention"
        self.self_attn = Qwen3MoeAttention(config, layer_idx)
        self.mlp = Qwen3MoeSparseMoeBlock(config)
        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, *, layer_past=None, use_cache=False):
        out, state = self.self_attn(self.input_layernorm(hidden_states), layer_past)
        out = out + hidden_states
        out = out + self.mlp(self.post_attention_layernorm(out))
        return (out, state) if use_cache else (out,)

    def inference_with_cache(self, hidden_states, cache, prefix_length, chunk_length):
        """Commit per-session state only after a successful forward.

        A rewind is free here: the K/V tensors are sliced to `prefix_length` and the
        rest is simply overwritten, so unlike the hybrid Qwen3.5 adapter nothing has
        to be replayed and no block inputs have to be retained.
        """
        length = hidden_states.shape[1]
        state = (cache[0][:, :, :prefix_length], cache[1][:, :, :prefix_length])
        outputs = []
        for offset in range(0, length, chunk_length):
            out, state = self(hidden_states[:, offset : offset + chunk_length], layer_past=state, use_cache=True)
            outputs.append(out)
        end = prefix_length + length
        for target, source in zip(cache, state):
            target[:, :, prefix_length:end].copy_(source[:, :, prefix_length:end])
        return (torch.cat(outputs, dim=1),)
