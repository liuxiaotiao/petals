"""Eager, text-only Qwen blocks with explicit per-session hybrid state.

The equations and checkpoint layout follow Transformers v5.5.0. See ops.py for
the upstream license. No model state is stored on a shared block between calls.
"""
import torch
from hivemind.utils.logging import get_logger
from torch import nn
from torch.nn import functional as F

from petals.models.qwen3_5_moe.ops import (
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRMSNormGated,
    Qwen3_5MoeSparseMoeBlock,
    apply_rotary_pos_emb,
    repeat_kv,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

logger = get_logger(__name__)

# Smallest rotary table we bother to build, so short sessions still grow it rarely.
ROPE_CACHE_MIN_LENGTH = 256
# Above this many replayed tokens a rewind costs a visible prefill, so say so out loud.
REPLAY_WARN_TOKENS = 512


class QwenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_key_value_groups
        for name, size in (
            ("q_proj", self.num_heads * self.head_dim * 2),
            ("k_proj", config.num_key_value_heads * self.head_dim),
            ("v_proj", config.num_key_value_heads * self.head_dim),
        ):
            setattr(self, name, nn.Linear(config.hidden_size, size, bias=config.attention_bias))
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3_5MoeRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3_5MoeRMSNorm(self.head_dim, config.rms_norm_eps)
        # Plain dict, not a buffer: buffers land on the meta device during block loading
        # and would have to be excluded from every checkpoint the server reads.
        self._rope_cache = {}

    def rotary(self, prefix, length, device, dtype):
        """Cached cos/sin rows. For text, all three MRoPE position axes are equal.

        The table is rebuilt if it was first created under inference mode and is now
        needed with autograd enabled, since inference tensors cannot be saved for backward.
        """
        rope = self.config.rope_parameters
        dim = int(self.head_dim * rope.get("partial_rotary_factor", 1.0))
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
        inv = 1.0 / (rope["rope_theta"] ** (torch.arange(0, dim, 2, device=device).float() / dim))
        freqs = torch.arange(size, device=device).float()[:, None] * inv[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        self._rope_cache[key] = cached = (emb.cos().to(dtype), emb.sin().to(dtype))
        return cached[0][prefix:end][None], cached[1][prefix:end][None]

    def forward(self, x, past=None):
        batch, length, _ = x.shape
        prefix = 0 if past is None else past[0].shape[2]
        q, gate = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim * 2).chunk(2, dim=-1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(batch, length, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, -1, self.head_dim).transpose(1, 2)
        positions = torch.arange(prefix, prefix + length, device=x.device)
        q, k = apply_rotary_pos_emb(q, k, *self.rotary(prefix, length, x.device, x.dtype))
        if past is not None:
            k, v = torch.cat((past[0], k), dim=2), torch.cat((past[1], v), dim=2)
        scores = (q @ repeat_kv(k, self.num_key_value_groups).transpose(-1, -2)) * self.head_dim**-0.5
        mask = torch.arange(k.shape[2], device=x.device)[None, :] > positions[:, None]
        scores = scores.masked_fill(mask[None, None], torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        result = (probs @ repeat_kv(v, self.num_key_value_groups)).transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(result * gate.reshape(batch, length, -1).sigmoid()), (k, v)


class QwenGatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        self.value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.kernel_size = config.linear_conv_kernel_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.kernel_size, groups=self.conv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(config.linear_num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(config.linear_num_value_heads))
        self.norm = Qwen3_5MoeRMSNormGated(config.linear_value_head_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        for name, size in (
            ("in_proj_qkv", self.conv_dim),
            ("in_proj_z", self.value_dim),
            ("in_proj_b", config.linear_num_value_heads),
            ("in_proj_a", config.linear_num_value_heads),
        ):
            setattr(self, name, nn.Linear(config.hidden_size, size, bias=False))

    def forward(self, x, past=None):
        c = self.config
        batch, length, _ = x.shape
        mixed = self.in_proj_qkv(x).transpose(1, 2)
        previous = mixed.new_zeros(batch, self.conv_dim, self.kernel_size) if past is None else past[0]
        mixed_history = torch.cat((previous, mixed), dim=-1)
        conv_state = mixed_history[:, :, -self.kernel_size :].contiguous()
        mixed = F.silu(self.conv1d(mixed_history)[:, :, -length:]).transpose(1, 2)
        q, k, v = mixed.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
        q = q.reshape(batch, length, -1, c.linear_key_head_dim)
        k = k.reshape_as(q)
        v = v.reshape(batch, length, -1, c.linear_value_head_dim)
        groups = c.linear_num_value_heads // c.linear_num_key_heads
        q, k = q.repeat_interleave(groups, dim=2), k.repeat_interleave(groups, dim=2)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)
        rule = torch_recurrent_gated_delta_rule if length == 1 else torch_chunk_gated_delta_rule
        out, recurrent = rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=None if past is None else past[1],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        z = self.in_proj_z(x).reshape(-1, c.linear_value_head_dim)
        out = self.norm(out.reshape_as(z), z).reshape(batch, length, self.value_dim)
        return self.out_proj(out), (conv_state, recurrent)


def cache_specs(config, layer_type, batch_size, max_length, dtype):
    """Tensor shapes/dtypes; shared by allocation and the server's memory estimate."""
    if layer_type == "full_attention":
        shape = (batch_size, config.num_key_value_heads, max_length, config.head_dim)
        return [(shape, dtype), (shape, dtype)]
    conv_dim = 2 * config.linear_num_key_heads * config.linear_key_head_dim
    conv_dim += config.linear_num_value_heads * config.linear_value_head_dim
    return [
        ((batch_size, max_length, config.hidden_size), dtype),
        ((batch_size, conv_dim, config.linear_conv_kernel_dim), dtype),
        (
            (batch_size, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim),
            torch.float32,
        ),
        ((batch_size, 1), torch.int64),
    ]


class WrappedQwen3_5MoeBlock(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNet(config)
        else:
            self.self_attn = QwenAttention(config)
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        # Petals benchmarks standalone blocks without HF PreTrainedModel.post_init().
        # Initialize packed Parameters too; otherwise throughput probes read empty memory.
        nn.init.normal_(self.mlp.experts.gate_up_proj, std=config.initializer_range)
        nn.init.normal_(self.mlp.experts.down_proj, std=config.initializer_range)
        nn.init.normal_(self.mlp.gate.weight, std=config.initializer_range)
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, *, layer_past=None, use_cache=False):
        mixer = self.linear_attn if self.layer_type == "linear_attention" else self.self_attn
        out, state = mixer(self.input_layernorm(hidden_states), layer_past)
        out = out + hidden_states
        out = out + self.mlp(self.post_attention_layernorm(out))
        return (out, state) if use_cache else (out,)

    def inference_with_cache(self, hidden_states, cache, prefix_length, chunk_length):
        """Commit per-session state only after a successful forward.

        Linear state cannot be sliced like KV tensors. Retain block inputs and
        replay the mixer if Petals rewinds a session; normal decoding uses the
        constant-size recurrent state. Replays do not recompute the MoE.
        """
        length = hidden_states.shape[1]
        if self.layer_type == "full_attention":
            state = (cache[0][:, :, :prefix_length], cache[1][:, :, :prefix_length])
        else:
            history, conv, recurrent, position = cache
            if prefix_length == 0:
                state = None  # Fresh allocations are uninitialized.
            elif (position == prefix_length).all():
                state = (conv, recurrent)
            else:
                if (position < prefix_length).any():
                    raise ValueError("Cannot advance Qwen recurrent state past its cached history")
                state = None
                replay = f"replaying {prefix_length} tokens through the linear mixer after a session rewind"
                if prefix_length >= REPLAY_WARN_TOKENS:
                    logger.warning(f"Qwen {replay}; this costs one prefill over the rewound prefix")
                else:
                    logger.debug(f"Qwen {replay}")
                for offset in range(0, prefix_length, chunk_length):
                    inputs = history[:, offset : min(prefix_length, offset + chunk_length)]
                    _, state = self.linear_attn(self.input_layernorm(inputs), state)
        outputs = []
        for offset in range(0, length, chunk_length):
            out, state = self(hidden_states[:, offset : offset + chunk_length], layer_past=state, use_cache=True)
            outputs.append(out)
        end = prefix_length + length
        if self.layer_type == "full_attention":
            for target, source in zip(cache, state):
                target[:, :, prefix_length:end].copy_(source[:, :, prefix_length:end])
        else:
            history[:, prefix_length:end].copy_(hidden_states)
            conv.copy_(state[0])
            recurrent.copy_(state[1])
            position.fill_(end)
        return (torch.cat(outputs, dim=1),)
