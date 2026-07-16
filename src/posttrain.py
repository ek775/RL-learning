"""
Post-training: DPO (Direct Preference Optimisation) on Intel/orca_dpo_pairs.

Why DPO instead of PPO for this dataset?
-----------------------------------------
Intel/orca_dpo_pairs already provides (prompt, chosen, rejected) triples —
exactly what DPO needs.  DPO *directly* optimises the policy without a
separate reward model or value function, making it simpler and more stable
than PPO for this setting.

DPO objective (Rafailov et al., 2023)
--------------------------------------
L_DPO = -E[ log σ( β · (log π_θ(y_w|x) - log π_ref(y_w|x))
                     - β · (log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]

where:
  y_w = chosen response,  y_l = rejected response
  β   = temperature (controls divergence from reference)

The log-ratio log π / π_ref for a response is the *sum* of per-token
log-probabilities over the response tokens only (not the prompt).

Dataset fields (Intel/orca_dpo_pairs)
--------------------------------------
  system   : system instruction
  question : user query
  chosen   : preferred assistant response
  rejected : dis-preferred assistant response

Run
---
    # Fine-tune from a pre-trained checkpoint:
    python posttrain.py --pretrain_ckpt checkpoints/pretrain_final.pt

    # Or train from scratch (for experimentation):
    python posttrain.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from transformer import Transformer, TransformerConfig


# ---------------------------------------------------------------------------
# DPO Dataset
# ---------------------------------------------------------------------------

class DPODataset(Dataset):
    """
    Tokenises (prompt, chosen, rejected) triples for DPO training.

    Each item returns two complete sequences:
        chosen_ids   = tokenise(prompt + chosen)
        rejected_ids = tokenise(prompt + rejected)
    and the corresponding response masks (1 for response tokens, 0 for prompt).
    """

    def __init__(self, hf_split, tokenizer, max_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.examples: list[dict] = []

        for row in hf_split:
            system   = (row.get("system")   or "").strip()
            question = (row.get("question") or "").strip()
            chosen   = (row.get("chosen")   or "").strip()
            rejected = (row.get("rejected") or "").strip()

            # Format: "[system]\nUser: {question}\nAssistant: {response}"
            prompt = (f"{system}\n" if system else "") + f"User: {question}\nAssistant: "

            chosen_full   = prompt + chosen
            rejected_full = prompt + rejected

            chosen_ids   = tokenizer.encode(chosen_full,   add_special_tokens=True)
            rejected_ids = tokenizer.encode(rejected_full, add_special_tokens=True)
            prompt_ids   = tokenizer.encode(prompt,        add_special_tokens=True)
            prompt_len   = len(prompt_ids)

            # Truncate to max_len
            chosen_ids   = chosen_ids[:max_len]
            rejected_ids = rejected_ids[:max_len]

            # Skip examples where the response was truncated away entirely
            if len(chosen_ids) <= prompt_len or len(rejected_ids) <= prompt_len:
                continue

            self.examples.append({
                "chosen_ids":   chosen_ids,
                "rejected_ids": rejected_ids,
                "prompt_len":   prompt_len,
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "chosen_ids":   torch.tensor(ex["chosen_ids"],   dtype=torch.long),
            "rejected_ids": torch.tensor(ex["rejected_ids"], dtype=torch.long),
            "prompt_len":   ex["prompt_len"],
        }


def dpo_collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad chosen and rejected sequences to the longest in the batch."""
    def pad_seq(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(s.size(0) for s in seqs)
        padded  = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        mask    = torch.zeros(len(seqs), max_len, dtype=torch.float)
        for i, s in enumerate(seqs):
            padded[i, :s.size(0)] = s
            mask[i,   :s.size(0)] = 1.0
        return padded, mask

    chosen_ids,   chosen_pad_mask   = pad_seq([b["chosen_ids"]   for b in batch])
    rejected_ids, rejected_pad_mask = pad_seq([b["rejected_ids"] for b in batch])
    prompt_lens = torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long)

    return {
        "chosen_ids":        chosen_ids,
        "rejected_ids":      rejected_ids,
        "chosen_pad_mask":   chosen_pad_mask,
        "rejected_pad_mask": rejected_pad_mask,
        "prompt_lens":       prompt_lens,
    }


# ---------------------------------------------------------------------------
# Log-probability helpers
# ---------------------------------------------------------------------------

def response_log_probs(
    model: Transformer,
    input_ids: torch.Tensor,    # (B, T)
    pad_mask: torch.Tensor,     # (B, T) — 0 for pad tokens
    prompt_lens: torch.Tensor,  # (B,)   — number of prompt tokens
) -> torch.Tensor:
    """
    Sum of log π(token | context) over *response* tokens only.
    Returns shape (B,).
    """
    logits = model(input_ids)                                   # (B, T, V)
    log_p  = F.log_softmax(logits[:, :-1], dim=-1)             # (B, T-1, V)
    target = input_ids[:, 1:]                                   # (B, T-1)

    token_lp = log_p.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # Build response mask: 1 only for response tokens (after prompt, before padding)
    B, T_minus1 = token_lp.shape
    positions = torch.arange(T_minus1, device=input_ids.device).unsqueeze(0)  # (1, T-1)
    # The target at position t corresponds to token t+1, which is a response token
    # if (t+1) >= prompt_len, i.e. t >= prompt_len - 1
    response_mask = (positions >= (prompt_lens - 1).unsqueeze(1)).float()
    response_mask = response_mask * pad_mask[:, 1:]             # also zero out padding

    return (token_lp * response_mask).sum(dim=1)                # (B,)


# ---------------------------------------------------------------------------
# DPO Loss
# ---------------------------------------------------------------------------

def dpo_loss(
    policy: Transformer,
    ref_model: Transformer,
    batch: dict[str, torch.Tensor],
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Computes the DPO loss for a batch.

    β controls how strongly the policy stays near the reference.
    Typical values: 0.1 – 0.5.
    """
    chosen_ids   = batch["chosen_ids"]
    rejected_ids = batch["rejected_ids"]
    prompt_lens  = batch["prompt_lens"]

    # Policy log-probs
    pi_chosen   = response_log_probs(policy, chosen_ids,   batch["chosen_pad_mask"],   prompt_lens)
    pi_rejected = response_log_probs(policy, rejected_ids, batch["rejected_pad_mask"], prompt_lens)

    # Reference log-probs (no grad)
    with torch.no_grad():
        ref_chosen   = response_log_probs(ref_model, chosen_ids,   batch["chosen_pad_mask"],   prompt_lens)
        ref_rejected = response_log_probs(ref_model, rejected_ids, batch["rejected_pad_mask"], prompt_lens)

    # DPO implicit reward
    logits_w = beta * (pi_chosen   - ref_chosen)
    logits_l = beta * (pi_rejected - ref_rejected)

    loss = -F.logsigmoid(logits_w - logits_l).mean()

    with torch.no_grad():
        accuracy       = (logits_w > logits_l).float().mean().item()
        reward_margin  = (logits_w - logits_l).mean().item()

    metrics = {
        "dpo/loss":            loss.item(),
        "dpo/accuracy":        accuracy,
        "dpo/reward_margin":   reward_margin,
        "dpo/reward_chosen":   logits_w.mean().item(),
        "dpo/reward_rejected": logits_l.mean().item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # --- Tokenizer ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_id     = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    # --- Model --------------------------------------------------------------
    if args.pretrain_ckpt and Path(args.pretrain_ckpt).exists():
        ckpt = torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=False)
        cfg  = ckpt["cfg"]
        policy = Transformer(cfg).to(device)
        policy.load_state_dict(ckpt["model"])
        print(f"Loaded pre-trained weights from {args.pretrain_ckpt}")
    else:
        cfg = TransformerConfig(
            vocab_size  = vocab_size,
            max_seq_len = args.max_seq_len,
            d_model     = args.d_model,
            n_heads     = args.n_heads,
            n_layers    = args.n_layers,
        )
        policy = Transformer(cfg).to(device)
        print("No checkpoint found — training from random init.")

    # Reference model = frozen copy of the starting policy
    ref_model = Transformer(cfg).to(device)
    ref_model.load_state_dict(policy.state_dict())
    for p in ref_model.parameters():
        p.requires_grad_(False)

    print(f"Model parameters: {policy.num_params() / 1e6:.1f}M")

    # --- Dataset ------------------------------------------------------------
    print("Loading Intel/orca_dpo_pairs ...")
    raw = load_dataset("Intel/orca_dpo_pairs")

    train_split = raw["train"]
    # Use a small fraction for validation (dataset has no pre-split val set)
    val_size    = min(500, int(0.02 * len(train_split)))
    val_split   = train_split.select(range(val_size))
    train_split = train_split.select(range(val_size, len(train_split)))

    train_ds = DPODataset(train_split, tokenizer, max_len=args.max_seq_len)
    val_ds   = DPODataset(val_split,   tokenizer, max_len=args.max_seq_len)
    print(f"Train pairs: {len(train_ds):,}  |  Val pairs: {len(val_ds):,}")

    collate = lambda b: dpo_collate(b, pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=args.num_workers)

    # --- Optimiser ----------------------------------------------------------
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)

    # --- Training -----------------------------------------------------------
    step = 0
    t0   = time.time()

    for epoch in range(args.epochs):
        for batch in train_loader:
            if step >= args.max_steps:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            loss, metrics = dpo_loss(policy, ref_model, batch, beta=args.beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            if step % args.log_interval == 0:
                elapsed = time.time() - t0
                print(f"step {step:5d}  "
                      f"loss {metrics['dpo/loss']:.4f}  "
                      f"acc {metrics['dpo/accuracy']:.3f}  "
                      f"margin {metrics['dpo/reward_margin']:.3f}  "
                      f"{elapsed:.0f}s")
                t0 = time.time()

            if step % args.eval_interval == 0 and step > 0:
                val_metrics = evaluate(policy, ref_model, val_loader, device, args.beta)
                print(f"  [val]  loss {val_metrics['dpo/loss']:.4f}  "
                      f"acc {val_metrics['dpo/accuracy']:.3f}")

            if step % args.save_interval == 0 and step > 0:
                save_path = Path(args.out_dir) / f"dpo_step{step}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"step": step, "model": policy.state_dict(), "cfg": cfg}, save_path)
                print(f"  Saved: {save_path}")

            step += 1

    save_path = Path(args.out_dir) / "dpo_final.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": policy.state_dict(), "cfg": cfg}, save_path)
    print(f"Training complete. Final checkpoint: {save_path}")


@torch.no_grad()
def evaluate(
    policy: Transformer,
    ref_model: Transformer,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    max_batches: int = 50,
) -> dict[str, float]:
    policy.eval()
    accum: dict[str, float] = {}
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        _, metrics = dpo_loss(policy, ref_model, batch, beta)
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        count += 1
    policy.train()
    return {k: v / max(count, 1) for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO post-training on Intel/orca_dpo_pairs")
    p.add_argument("--pretrain_ckpt",  type=str,   default="checkpoints/pretrain_final.pt",
                   help="Path to pre-trained checkpoint (optional)")
    p.add_argument("--max_seq_len",    type=int,   default=1024,
                   help="Max tokens per sequence (prompt+response). Lower = faster.")
    p.add_argument("--d_model",        type=int,   default=512)
    p.add_argument("--n_heads",        type=int,   default=8)
    p.add_argument("--n_layers",       type=int,   default=6)
    p.add_argument("--beta",           type=float, default=0.1,
                   help="DPO temperature β — higher = stay closer to reference")
    p.add_argument("--epochs",         type=int,   default=1)
    p.add_argument("--max_steps",      type=int,   default=10_000)
    p.add_argument("--batch_size",     type=int,   default=2)
    p.add_argument("--lr",             type=float, default=5e-6)
    p.add_argument("--max_grad_norm",  type=float, default=1.0)
    p.add_argument("--log_interval",   type=int,   default=50)
    p.add_argument("--eval_interval",  type=int,   default=500)
    p.add_argument("--save_interval",  type=int,   default=2000)
    p.add_argument("--num_workers",    type=int,   default=2)
    p.add_argument("--out_dir",        type=str,   default="checkpoints")
    p.add_argument("--device",         type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
