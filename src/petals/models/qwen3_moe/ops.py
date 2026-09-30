# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text math backported from Transformers modeling_qwen3_moe.py.

Changes: eager PyTorch only, with the modern Transformers integrations and type
dependencies removed. The sparse MoE keeps one module per expert rather than the
packed 3-D parameters newer Transformers uses, because Petals loads block weights
by exact parameter name and the Hub checkpoint stores each expert separately as
`mlp.experts.<i>.{gate,up,down}_proj.weight`. The routed arithmetic is unchanged.
Upstream: https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Qwen3MoeRMSNorm(nn.Module):
    """Note the difference from Qwen3.5: this scales by `weight`, not by `1 + weight`."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states, n_rep: int):
    """(batch, num_key_value_heads, length, head_dim) -> (batch, num_attention_heads, length, head_dim)."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, length, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, length, head_dim)


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Top-k routing over `num_experts` independent MLPs; Qwen3-MoE has no shared expert."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)
        )

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)
        # (tokens, top_k, experts) -> (experts, top_k, tokens), so one slice selects an expert's work.
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_index in torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero():
            expert_index = int(expert_index[0])
            top_k_position, token_index = torch.where(expert_mask[expert_index])
            current = self.experts[expert_index](hidden_states[token_index])
            current = current * routing_weights[token_index, top_k_position, None]
            final_hidden_states.index_add_(0, token_index, current.to(final_hidden_states.dtype))
        return final_hidden_states.view(batch_size, sequence_length, hidden_dim)
