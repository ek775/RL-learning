"""
Reward Model for RLHF.

Wraps the base Transformer and replaces the language-model head with a
scalar value head that outputs a single reward score per sequence.

Training objective (Bradley-Terry preference model):
    loss = -log σ(r_chosen - r_rejected)

where r_chosen and r_rejected are the scalar outputs for the preferred
and dis-preferred completions respectively.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer, TransformerConfig


class RewardModel(nn.Module):
    """
    Scalar reward head on top of a frozen or fine-tuned transformer backbone.

    Usage
    -----
    cfg = TransformerConfig()
    rm  = RewardModel(cfg)

    # During RLHF rollouts
    rewards = rm(input_ids)   # (batch,)

    # During reward model training (Bradley-Terry)
    loss = rm.preference_loss(chosen_ids, rejected_ids)
    """

    def __init__(self, cfg: TransformerConfig, pretrained: Transformer | None = None):
        super().__init__()
        if pretrained is not None:
            self.backbone = pretrained
        else:
            self.backbone = Transformer(cfg)

        # Remove the language-model head (not needed for a reward model)
        self.backbone.lm_head = nn.Identity()  # type: ignore[assignment]

        self.value_head = nn.Linear(cfg.d_model, 1, bias=False)
        nn.init.normal_(self.value_head.weight, std=0.02)

    def _last_token_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run the backbone and pool the hidden state at the last *real* token.
        input_ids: (batch, seq_len)
        returns:   (batch, d_model)
        """
        hidden = self.backbone(input_ids)   # (batch, seq_len, d_model) — lm_head is Identity
        # Find the index of the last non-padding token (assumes 0 = pad)
        seq_lens = (input_ids != 0).sum(dim=1) - 1   # (batch,)
        seq_lens = seq_lens.clamp(min=0)
        idx = seq_lens[:, None, None].expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)      # (batch, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns scalar reward scores, shape (batch,).
        Higher score = model judges the response as better.
        """
        h = self._last_token_hidden(input_ids)
        return self.value_head(h).squeeze(-1)

    def preference_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Bradley-Terry pairwise ranking loss.

        chosen_ids:   (batch, seq_len_c)
        rejected_ids: (batch, seq_len_r)

        Returns (loss, metrics_dict).
        """
        r_chosen   = self(chosen_ids)    # (batch,)
        r_rejected = self(rejected_ids)  # (batch,)

        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        with torch.no_grad():
            accuracy = (r_chosen > r_rejected).float().mean().item()

        metrics = {
            "rm/loss": loss.item(),
            "rm/accuracy": accuracy,
            "rm/reward_chosen": r_chosen.mean().item(),
            "rm/reward_rejected": r_rejected.mean().item(),
            "rm/reward_margin": (r_chosen - r_rejected).mean().item(),
        }
        return loss, metrics
