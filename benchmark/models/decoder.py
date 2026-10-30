"""Trusted GPT-2 causal decoder (architecture id ``gpt2_decoder``).

Same layout as Hugging Face GPT2LMHeadModel: learned absolute positions, pre-LayerNorm blocks,
fused QKV projection, gelu_new (tanh) MLP, biases on all Linear/LayerNorm layers, tied LM head.
Attention is "math" attention with FP32 intermediates: Q·Kᵀ, softmax and ·V run in float32 even
under BF16 autocast (the contract's math_attention_fp32_intermediates assumption). No remote code.
"""
import math

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, H, A, I, p_attn, p_resid):
        super().__init__()
        self.H, self.A = H, A
        self.ln_1 = nn.LayerNorm(H)
        self.c_attn = nn.Linear(H, 3 * H)
        self.c_proj = nn.Linear(H, H)
        self.ln_2 = nn.LayerNorm(H)
        self.c_fc = nn.Linear(H, I)
        self.mlp_proj = nn.Linear(I, H)
        self.act = nn.GELU(approximate='tanh')
        self.attn_drop = nn.Dropout(p_attn)
        self.resid_drop = nn.Dropout(p_resid)

    def forward(self, x, mask):
        B, S, H = x.shape
        q, k, v = self.c_attn(self.ln_1(x)).split(H, dim=2)
        q, k, v = (t.view(B, S, self.A, H // self.A).transpose(1, 2) for t in (q, k, v))
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(H // self.A)
            scores = scores.masked_fill(mask[:S, :S], float('-inf'))
            y = self.attn_drop(torch.softmax(scores, dim=-1)) @ v.float()
        y = y.to(q.dtype).transpose(1, 2).reshape(B, S, H)
        x = x + self.resid_drop(self.c_proj(y))
        return x + self.resid_drop(self.mlp_proj(self.act(self.c_fc(self.ln_2(x)))))


class GPT2Decoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        H, A, L = m['hidden_size'], m['num_attention_heads'], m['num_layers']
        I = m['intermediate_size'] or 4 * H
        C = m['max_position_embeddings']
        self.wte = nn.Embedding(m['vocab_size'], H)
        self.wpe = nn.Embedding(C, H)
        self.drop = nn.Dropout(m['embedding_dropout'])
        self.h = nn.ModuleList(Block(H, A, I, m['attention_dropout'], m['residual_dropout']) for _ in range(L))
        self.ln_f = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, m['vocab_size'], bias=False)
        self.lm_head.weight = self.wte.weight  # tied output embedding
        self.register_buffer('causal_mask', torch.ones(C, C, dtype=torch.bool).triu(1), persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.drop(self.wte(ids) + self.wpe(positions))
        for block in self.h:
            x = block(x, self.causal_mask)
        return self.lm_head(self.ln_f(x))
