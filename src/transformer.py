"""
Decoder-only transformer (GPT-style) designed for RLHF / RLVR experimentation.

Design choices:
  - RMSNorm  (pre-norm)  — more training-stable than post LayerNorm
  - RoPE positional embeddings — generalises beyond training length, no learned positions
  - SwiGLU feed-forward — matches LLaMA / Mistral style
  - torch.nn.functional.scaled_dot_product_attention — uses FlashAttention when available
  - Causal (autoregressive) masking built into the attention call
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    vocab_size: int = 32_000      # e.g. SentencePiece / tiktoken vocabulary
    max_seq_len: int = 4_096      # context window
    d_model: int = 512            # embedding / hidden dimension
    n_heads: int = 8              # attention heads
    n_kv_heads: int = 8           # key/value heads (set < n_heads for GQA)
    n_layers: int = 6             # transformer blocks
    d_ff: int | None = None       # feed-forward inner dim; defaults to ~8/3 * d_model (SwiGLU)
    dropout: float = 0.0          # attention + residual dropout
    rope_base: float = 10_000.0   # RoPE frequency base
    tie_embeddings: bool = True   # share input/output embedding weights

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        if self.d_ff is None:
            # SwiGLU: 2/3 * 4 * d_model, rounded to nearest multiple of 256
            raw = int(2 / 3 * 4 * self.d_model)
            self.d_ff = (raw + 255) // 256 * 256


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no bias, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------

def _rope_freqs(dim: int, base: float, device: torch.device) -> torch.Tensor:
    """Precompute inverse frequencies for RoPE: shape (dim/2,)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    return inv_freq


def _apply_rope(x: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to query or key tensor.
    x: (batch, n_heads, seq_len, head_dim)
    """
    seq_len = x.size(2)
    t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)          # (seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)   # (seq_len, head_dim)
    cos = emb.cos()[None, None, :, :]         # (1, 1, seq_len, head_dim)
    sin = emb.sin()[None, None, :, :]

    def rotate_half(v: torch.Tensor) -> torch.Tensor:
        half = v.shape[-1] // 2
        return torch.cat([-v[..., half:], v[..., :half]], dim=-1)

    return (x * cos) + (rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# Causal Self-Attention (with optional Grouped Query Attention)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads  # repetitions for GQA

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.attn_dropout = cfg.dropout
        self.register_buffer(
            "_inv_freq",
            _rope_freqs(self.head_dim, cfg.rope_base, torch.device("cpu")),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Move inv_freq to the right device (lazy)
        inv_freq = self._inv_freq.to(x.device)
        q = _apply_rope(q, inv_freq)
        k = _apply_rope(k, inv_freq)

        # Expand KV heads to match Q heads (GQA)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Uses FlashAttention-2 kernel when available (torch >= 2.0)
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU FFN: uses two gate projections and element-wise SiLU gating."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up   = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# Full Decoder-Only Transformer
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    Causal language model.

    forward() returns raw logits of shape (batch, seq_len, vocab_size).
    Pair with F.cross_entropy for SFT pre-training, or strip the lm_head
    and attach a scalar head for a reward model (see reward_model.py).
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)  — token indices, seq_len <= max_seq_len
        returns:   (batch, seq_len, vocab_size)  — next-token logits
        """
        assert input_ids.size(1) <= self.cfg.max_seq_len, (
            f"Sequence length {input_ids.size(1)} exceeds max_seq_len {self.cfg.max_seq_len}"
        )
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Simple autoregressive generation (no KV-cache — for clarity).
        Returns the full sequence including the prompt.
        """
        for _ in range(max_new_tokens):
            # Truncate context to max_seq_len
            ctx = input_ids[:, -self.cfg.max_seq_len:]
            logits = self(ctx)[:, -1, :]  # (batch, vocab_size)

            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                topk_vals = torch.topk(logits, top_k).values[:, [-1]]
                logits = logits.masked_fill(logits < topk_vals, float("-inf"))

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
