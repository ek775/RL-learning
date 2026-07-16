"""
PPO Trainer for RLHF (Reinforcement Learning from Human Feedback).

High-level algorithm
--------------------
1. Rollout  — the *actor* (policy) generates completions for a batch of prompts.
2. Score    — the *reward model* assigns a scalar reward to each completion.
              A KL penalty vs. a frozen reference policy is subtracted to stop
              the actor from exploiting the reward model (reward hacking).
3. Estimate — GAE (Generalised Advantage Estimation) computes per-token advantages
              using the *critic* (value function).
4. Update   — multiple epochs of PPO-Clip updates on the actor + critic.

References
----------
- Schulman et al. 2017 — "Proximal Policy Optimization Algorithms"
- Stiennon et al. 2020 — "Learning to summarize from human feedback"
- Ziegler  et al. 2019 — "Fine-Tuning Language Models from Human Preferences"
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Transformer, TransformerConfig
from reward_model import RewardModel
from loss import (
    sequence_log_probs,
    ppo_policy_loss,
    value_loss,
    kl_penalty,
    normalise_advantages,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    # Rollout
    rollout_batch_size: int = 8        # prompts per rollout step
    max_new_tokens: int = 256          # max completion length
    temperature: float = 1.0

    # PPO update
    ppo_epochs: int = 4                # passes over each rollout batch
    mini_batch_size: int = 4
    clip_eps: float = 0.2              # PPO clipping ε
    vf_coef: float = 0.1               # value loss coefficient
    entropy_coef: float = 0.01         # entropy bonus coefficient
    kl_coef: float = 0.04              # KL penalty coefficient (adaptive or fixed)

    # GAE
    gamma: float = 1.0                 # discount (1.0 = no discounting for bandit RLHF)
    lam: float = 0.95                  # GAE λ

    # Optimisation
    lr_actor: float = 1e-5
    lr_critic: float = 1e-4
    max_grad_norm: float = 1.0


# ---------------------------------------------------------------------------
# Critic (value function)
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    Per-token value function V(s_t).
    Shares the transformer backbone with the actor but has its own value head.
    In practice you often initialise from the same SFT checkpoint.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.backbone = Transformer(cfg)
        self.backbone.lm_head = nn.Identity()          # type: ignore[assignment]
        self.value_head = nn.Linear(cfg.d_model, 1, bias=False)
        nn.init.normal_(self.value_head.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns per-token value estimates, shape (batch, seq_len)."""
        hidden = self.backbone(input_ids)    # (B, T, d_model)
        return self.value_head(hidden).squeeze(-1)   # (B, T)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,   # (batch, seq_len) — reward only at last token
    values: torch.Tensor,    # (batch, seq_len)
    mask: torch.Tensor,      # (batch, seq_len) — 1 = real token
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages via GAE and returns via discounted sum.
    Returns (advantages, returns), both (batch, seq_len).
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae   = torch.zeros(B, device=rewards.device)

    # Bootstrap value after the last token is 0 (episode ends)
    next_value = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        m          = mask[:, t]
        delta      = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae   = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae * m
        next_value = values[:, t] * m   # carry value forward only for real tokens

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Minimal PPO trainer for a single-GPU setup.

    Parameters
    ----------
    actor     : the policy being trained (Transformer)
    ref_model : frozen SFT reference policy (Transformer) — used for KL penalty
    reward_model: trained RewardModel
    critic    : value function (Critic)
    cfg       : PPOConfig
    """

    def __init__(
        self,
        actor: Transformer,
        ref_model: Transformer,
        reward_model: RewardModel,
        critic: Critic,
        cfg: PPOConfig,
    ):
        self.actor        = actor
        self.ref_model    = ref_model
        self.reward_model = reward_model
        self.critic       = critic
        self.cfg          = cfg

        self.actor_optim  = torch.optim.AdamW(actor.parameters(),  lr=cfg.lr_actor)
        self.critic_optim = torch.optim.AdamW(critic.parameters(), lr=cfg.lr_critic)

        # Freeze reference model — never updated
        for p in ref_model.parameters():
            p.requires_grad_(False)
        for p in reward_model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Rollout phase
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(self, prompt_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Generate completions and collect all quantities needed for the PPO update.

        prompt_ids: (batch, prompt_len)

        Returns a dict with:
            input_ids      (B, full_len)
            response_mask  (B, full_len)  — 1 for generated tokens
            log_probs      (B, full_len)
            ref_log_probs  (B, full_len)
            values         (B, full_len)
            rewards        (B, full_len)  — non-zero only at final token
        """
        cfg = self.cfg
        prompt_len = prompt_ids.size(1)

        # 1. Generate completions
        input_ids = self.actor.generate(
            prompt_ids,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )  # (B, prompt_len + completion_len)

        full_len = input_ids.size(1)
        response_mask = torch.zeros_like(input_ids, dtype=torch.float)
        response_mask[:, prompt_len:] = 1.0

        # 2. Log-probs under current actor and reference model
        actor_logits = self.actor(input_ids)
        log_probs    = sequence_log_probs(actor_logits, input_ids, response_mask)

        ref_logits    = self.ref_model(input_ids)
        ref_log_probs = sequence_log_probs(ref_logits, input_ids, response_mask)

        # 3. Critic values
        values = self.critic(input_ids)  # (B, T)

        # 4. Reward from reward model (scalar per sequence)
        seq_reward = self.reward_model(input_ids)  # (B,)

        # 5. KL penalty (per-token), subtracted from reward
        kl = (log_probs - ref_log_probs).detach()  # positive means policy moved away

        # Place shaped rewards at each response token: r_t = -kl_coef * KL_t
        # Add terminal reward at the last response token
        shaped = -cfg.kl_coef * kl * response_mask
        last_idx = response_mask.sum(dim=1).long() + prompt_len - 1   # idx of last gen token
        last_idx = last_idx.clamp(0, full_len - 1)
        shaped[torch.arange(input_ids.size(0)), last_idx] += seq_reward

        return {
            "input_ids":     input_ids,
            "response_mask": response_mask,
            "log_probs":     log_probs.detach(),
            "ref_log_probs": ref_log_probs.detach(),
            "values":        values.detach(),
            "rewards":       shaped.detach(),
        }

    # ------------------------------------------------------------------
    # Update phase
    # ------------------------------------------------------------------

    def update(self, rollout: dict[str, torch.Tensor]) -> dict[str, float]:
        """Run PPO-Clip update epochs on the collected rollout."""
        cfg = self.cfg

        input_ids     = rollout["input_ids"]
        response_mask = rollout["response_mask"]
        old_log_probs = rollout["log_probs"]
        old_values    = rollout["values"]
        rewards       = rollout["rewards"]

        # Advantages & returns (computed once from rollout data)
        advantages, returns = compute_gae(
            rewards, old_values, response_mask, cfg.gamma, cfg.lam
        )
        advantages = normalise_advantages(advantages, response_mask)

        B = input_ids.size(0)
        metrics_accum: dict[str, float] = {}

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(B, device=input_ids.device)
            for start in range(0, B, cfg.mini_batch_size):
                idx = perm[start : start + cfg.mini_batch_size]

                mb_ids   = input_ids[idx]
                mb_mask  = response_mask[idx]
                mb_adv   = advantages[idx]
                mb_ret   = returns[idx]
                mb_old_lp = old_log_probs[idx]
                mb_old_v  = old_values[idx]

                # Forward passes
                logits  = self.actor(mb_ids)
                log_probs_new = sequence_log_probs(logits, mb_ids, mb_mask)
                values_new    = self.critic(mb_ids)

                # Entropy bonus (encourage exploration)
                probs   = F.softmax(logits[:, :-1], dim=-1)
                entropy = -(probs * probs.log().clamp(min=-100)).sum(-1)
                entropy_loss = -(entropy * mb_mask[:, 1:]).sum() / mb_mask[:, 1:].sum().clamp(min=1)

                p_loss = ppo_policy_loss(log_probs_new, mb_old_lp, mb_adv, mb_mask, cfg.clip_eps)
                v_loss = value_loss(values_new, mb_old_v, mb_ret, mb_mask, cfg.clip_eps)

                actor_loss = p_loss + cfg.vf_coef * v_loss + cfg.entropy_coef * entropy_loss

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),  cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                for k, v in {
                    "ppo/policy_loss":  p_loss.item(),
                    "ppo/value_loss":   v_loss.item(),
                    "ppo/entropy":     -entropy_loss.item(),
                }.items():
                    metrics_accum[k] = metrics_accum.get(k, 0.0) + v

        # Average metrics over all mini-batch steps
        n_steps = cfg.ppo_epochs * max(1, B // cfg.mini_batch_size)
        return {k: v / n_steps for k, v in metrics_accum.items()}
