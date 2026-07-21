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

    def __init__(
        self,
        cfg: TransformerConfig,
        pretrained: Transformer | None = None,
        pad_token_id: int = 0,
    ):
        super().__init__()
        if pretrained is not None:
            self.backbone = pretrained
        else:
            self.backbone = Transformer(cfg)

        # Remove the language-model head (not needed for a reward model)
        self.backbone.lm_head = nn.Identity()  # type: ignore[assignment]

        self.value_head = nn.Linear(cfg.d_model, 1, bias=False)
        nn.init.normal_(self.value_head.weight, std=0.02)

        # Token id used for padding — GPT-2's tokenizer sets pad == eos (50256),
        # not 0, so this must be configurable rather than hard-coded.
        self.pad_token_id = pad_token_id

    def _last_token_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the backbone and pool the hidden state at the last *real* token.
        input_ids:      (batch, seq_len)
        attention_mask: (batch, seq_len), optional — 1 for real tokens, 0 for padding.
        returns:        (batch, d_model)
        """
        hidden = self.backbone(input_ids)   # (batch, seq_len, d_model) — lm_head is Identity
        # Find the index of the last non-padding token. Prefer the attention
        # mask (accurate even if a real token happens to equal pad_token_id);
        # fall back to comparing against pad_token_id when no mask is given.
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1) - 1   # (batch,)
        else:
            seq_lens = (input_ids != self.pad_token_id).sum(dim=1) - 1   # (batch,)
        seq_lens = seq_lens.clamp(min=0)
        idx = seq_lens[:, None, None].expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)      # (batch, d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns scalar reward scores, shape (batch,).
        Higher score = model judges the response as better.
        """
        h = self._last_token_hidden(input_ids, attention_mask)
        return self.value_head(h).squeeze(-1)

    def preference_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: torch.Tensor | None = None,
        rejected_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Bradley-Terry pairwise ranking loss.

        chosen_ids:   (batch, seq_len_c)
        rejected_ids: (batch, seq_len_r)
        chosen_mask, rejected_mask: optional attention masks matching the ids above.

        Returns (loss, metrics_dict).
        """
        r_chosen   = self(chosen_ids, chosen_mask)      # (batch,)
        r_rejected = self(rejected_ids, rejected_mask)  # (batch,)

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
