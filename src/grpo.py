"""
GRPO Trainer for RLVR (Reinforcement Learning from Verifiable Rewards).

GRPO = Group Relative Policy Optimisation (DeepSeek-R1, Shao et al. 2024).

Key differences from PPO / RLHF
---------------------------------
1. **No critic / value function.** Advantages are estimated purely from
   reward comparisons *within a group* of G completions for the same prompt.
   This eliminates the need to train a separate value head.

2. **Verifiable rewards.** Instead of a learned reward model, a deterministic
   verifier function checks correctness (e.g. math answer == ground truth,
   code passes unit tests).  Reward is binary (or structured) and not gameable.

3. **Group-relative baseline.** For each prompt x, generate G completions.
   The baseline is the mean reward of the group.  The advantage for completion i is:
       A_i = (r_i - mean(r)) / (std(r) + ε)

4. **KL penalty** vs. a frozen reference policy is still kept to prevent collapse.

Algorithm (one step)
--------------------
For each batch of prompts:
  a) Sample G completions per prompt from π_θ.
  b) Score each completion with the verifier → r_i ∈ {0, 1} (or float).
  c) Compute group-normalised advantages A_i.
  d) Run K PPO-style update epochs using the clipped surrogate objective.

References
----------
- Shao et al. 2024 — "DeepSeekMath: Pushing the Limits of Mathematical
  Reasoning in Open Language Models"  (GRPO introduction)
- DeepSeek-AI 2025 — "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
  via Reinforcement Learning"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer, TransformerConfig
from .loss import sequence_log_probs, ppo_policy_loss, kl_penalty


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    # Rollout
    rollout_batch_size: int = 4    # number of distinct prompts per step
    group_size: int = 8            # G: completions sampled per prompt
    max_new_tokens: int = 512      # allow longer reasoning chains
    temperature: float = 0.8

    # GRPO update
    grpo_epochs: int = 1           # passes over each rollout batch (often 1)
    clip_eps: float = 0.2          # PPO-clip ε
    kl_coef: float = 0.04          # KL penalty coefficient

    # Optimisation
    lr: float = 1e-6               # typically very low — policy already capable
    max_grad_norm: float = 1.0


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """
    GRPO trainer — no critic, verifiable reward signal.

    Parameters
    ----------
    policy     : the model being trained (Transformer)
    ref_model  : frozen SFT reference — used only for KL penalty
    verifier   : callable(input_ids, prompt_len) -> Tensor of shape (batch,)
                 Returns a scalar reward for each completion.
                 Example verifiers: exact-match math, code execution pass/fail.
    cfg        : GRPOConfig
    """

    def __init__(
        self,
        policy: Transformer,
        ref_model: Transformer,
        verifier: Callable[[torch.Tensor, int], torch.Tensor],
        cfg: GRPOConfig,
    ):
        self.policy    = policy
        self.ref_model = ref_model
        self.verifier  = verifier
        self.cfg       = cfg

        self.optim = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)

        for p in ref_model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(self, prompt_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Sample G completions per prompt and compute group-relative advantages.

        prompt_ids: (P, prompt_len)  where P = rollout_batch_size

        Returns dict with tensors of shape (P*G, full_len) (or scalars).
        """
        cfg        = self.cfg
        P, L       = prompt_ids.shape
        G          = cfg.group_size
        prompt_len = L

        # Repeat each prompt G times → (P*G, L)
        expanded = prompt_ids.repeat_interleave(G, dim=0)

        # Generate completions
        input_ids = self.policy.generate(
            expanded,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )  # (P*G, full_len)

        full_len = input_ids.size(1)
        response_mask = torch.zeros(P * G, full_len, device=input_ids.device)
        response_mask[:, prompt_len:] = 1.0

        # Scores from verifier
        rewards = self.verifier(input_ids, prompt_len)   # (P*G,)

        # Group-relative advantage normalisation
        rewards_grouped = rewards.view(P, G)             # (P, G)
        mean_r = rewards_grouped.mean(dim=1, keepdim=True)
        std_r  = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages_grouped = (rewards_grouped - mean_r) / std_r  # (P, G)
        advantages = advantages_grouped.view(P * G)      # (P*G,)

        # Broadcast token-level: same advantage for every token in the completion
        advantages_token = advantages[:, None].expand(-1, full_len) * response_mask

        # Log-probs under current policy and reference
        policy_logits  = self.policy(input_ids)
        log_probs      = sequence_log_probs(policy_logits, input_ids, response_mask)

        ref_logits     = self.ref_model(input_ids)
        ref_log_probs  = sequence_log_probs(ref_logits, input_ids, response_mask)

        return {
            "input_ids":        input_ids,
            "response_mask":    response_mask,
            "log_probs":        log_probs.detach(),
            "ref_log_probs":    ref_log_probs.detach(),
            "advantages":       advantages_token.detach(),
            "rewards":          rewards.detach(),
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        """Run GRPO update epochs on the collected rollout."""
        cfg = self.cfg

        input_ids     = rollout["input_ids"]
        response_mask = rollout["response_mask"]
        old_log_probs = rollout["log_probs"]
        ref_log_probs = rollout["ref_log_probs"]
        advantages    = rollout["advantages"]

        total_p_loss = 0.0
        total_kl     = 0.0
        n_steps      = 0

        for _ in range(cfg.grpo_epochs):
            logits       = self.policy(input_ids)
            log_probs    = sequence_log_probs(logits, input_ids, response_mask)

            p_loss = ppo_policy_loss(
                log_probs, old_log_probs, advantages, response_mask, cfg.clip_eps
            )
            kl = kl_penalty(log_probs, ref_log_probs, response_mask)

            loss = p_loss + cfg.kl_coef * kl

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            self.optim.step()

            total_p_loss += p_loss.item()
            total_kl     += kl.item()
            n_steps      += 1

        return {
            "grpo/policy_loss":  total_p_loss / n_steps,
            "grpo/kl":           total_kl     / n_steps,
            "grpo/mean_reward":  rollout["rewards"].mean().item(),
            "grpo/reward_std":   rollout["rewards"].std().item(),
        }


# ---------------------------------------------------------------------------
# Example verifier builders
# ---------------------------------------------------------------------------

def exact_match_verifier(ground_truth: list[str], tokenizer_decode_fn: Callable) -> Callable:
    """
    Returns a verifier that gives reward 1.0 if the decoded completion
    contains the expected answer, 0.0 otherwise.

    tokenizer_decode_fn: callable(input_ids) -> list[str]
    """
    def verify(input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        completion_ids = input_ids[:, prompt_len:]
        texts = tokenizer_decode_fn(completion_ids)
        # ground_truth is repeated G times, matching the expanded batch
        scores = torch.tensor(
            [1.0 if gt.strip() in text else 0.0
             for gt, text in zip(ground_truth, texts)],
            device=input_ids.device,
        )
        return scores
    return verify
