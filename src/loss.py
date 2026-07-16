"""
RL loss functions shared by PPO (RLHF) and GRPO (RLVR).

Glossary
--------
log_probs   : log π_θ(a|s)  — log-probabilities under the *current* policy
old_log_probs: log π_old(a|s) — log-probabilities under the policy that collected rollouts
advantages  : A(s, a)       — estimated advantage (how much better than baseline)
values      : V_θ(s)        — critic's value estimate
returns     : G_t            — discounted cumulative rewards
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Token-level log-probabilities
# ---------------------------------------------------------------------------

def sequence_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token log π(token | context) for each response token.

    logits:        (batch, seq_len, vocab_size) — model output
    input_ids:     (batch, seq_len)             — target token ids
    response_mask: (batch, seq_len)             — 1 for response tokens, 0 for prompt/pad

    Returns: (batch, seq_len) — log-probs, zeroed outside the response mask
    """
    # Shift: logits at position t predict token at position t+1
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)  # (B, T-1, V)
    target    = input_ids[:, 1:]                        # (B, T-1)
    mask      = response_mask[:, 1:]                    # (B, T-1)

    token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return token_log_probs * mask


# ---------------------------------------------------------------------------
# PPO Clipped Surrogate Loss
# ---------------------------------------------------------------------------

def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """
    PPO-Clip objective (Schulman et al., 2017).

    Maximises E[min(r·A, clip(r, 1-ε, 1+ε)·A)]  where r = π/π_old.

    All tensors: (batch, seq_len).
    Returns scalar loss (negated for gradient *descent*).
    """
    ratio = torch.exp(log_probs - old_log_probs)   # importance-sampling ratio
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    surrogate = torch.min(ratio * advantages, clipped * advantages)

    # Average over unmasked tokens
    loss = -(surrogate * mask).sum() / mask.sum().clamp(min=1)
    return loss


# ---------------------------------------------------------------------------
# Value Function Loss
# ---------------------------------------------------------------------------

def value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """
    Clipped value loss — prevents the critic from making large updates.

    values, old_values, returns, mask: (batch, seq_len)
    """
    values_clipped = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
    loss1 = (values - returns).pow(2)
    loss2 = (values_clipped - returns).pow(2)
    loss  = 0.5 * (torch.max(loss1, loss2) * mask).sum() / mask.sum().clamp(min=1)
    return loss


# ---------------------------------------------------------------------------
# KL Divergence Penalty (token-level)
# ---------------------------------------------------------------------------

def kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sequence mean KL: KL(π_θ || π_ref) ≈ log π_θ - log π_ref.

    Used to prevent the policy from drifting too far from the reference SFT model.
    Returns scalar.
    """
    kl = (log_probs - ref_log_probs) * mask
    return kl.sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Advantage Normalisation (helper)
# ---------------------------------------------------------------------------

def normalise_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalise advantages to zero mean, unit variance over unmasked tokens.
    Stabilises policy gradient updates.
    """
    flat = advantages[mask.bool()]
    mean = flat.mean()
    std  = flat.std().clamp(min=eps)
    return (advantages - mean) / std
