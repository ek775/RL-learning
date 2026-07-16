"""
Pre-training: causal language modelling on wikitext-103-raw-v1.

Pipeline
--------
1. Tokenise every article in the dataset (GPT-2 BPE tokenizer, vocab = 50 257).
2. Concatenate all token ids and chunk into non-overlapping blocks of
   `max_seq_len` tokens (packing) — no padding, maximum GPU utilisation.
3. Train the decoder-only Transformer from transformer.py with:
      - AdamW optimiser
      - Linear warmup + cosine decay learning-rate schedule
      - Gradient accumulation (simulate larger batches on small GPUs)
      - Periodic validation-loss logging and checkpoint saving

Run
---
    python pretrain.py
    python pretrain.py --max_steps 5000 --batch_size 4 --device cuda
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

from rl.transformer import Transformer, TransformerConfig


# ---------------------------------------------------------------------------
# Packed sequence dataset
# ---------------------------------------------------------------------------

class PackedTextDataset(Dataset):
    """
    Concatenates all tokenised documents into one long stream and cuts it into
    fixed-length blocks of `block_size` tokens.  The next-token prediction
    target for each block is the same block shifted right by one.
    """

    def __init__(self, hf_split, tokenizer, block_size: int):
        token_ids: list[int] = []
        for example in hf_split:
            text = example.get("text", "") or ""
            if text.strip():
                ids = tokenizer.encode(text, add_special_tokens=False)
                token_ids.extend(ids)
                token_ids.append(tokenizer.eos_token_id)  # document boundary

        # Trim to a multiple of block_size
        total = (len(token_ids) // block_size) * block_size
        self.data = torch.tensor(token_ids[:total], dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) // self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1]
        # If the last block is short, wrap around (shouldn't happen after trimming)
        if len(chunk) < self.block_size + 1:
            chunk = torch.cat([chunk, self.data[: self.block_size + 1 - len(chunk)]])
        x = chunk[:-1]   # input tokens
        y = chunk[1:]    # target tokens (shifted by 1)
        return x, y


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / max_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # --- Tokenizer ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len  # silence GPT-2's 1024-token warning
    vocab_size = tokenizer.vocab_size  # 50 257

    # --- Model --------------------------------------------------------------
    cfg = TransformerConfig(
        vocab_size  = vocab_size,
        max_seq_len = args.max_seq_len,
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        dropout     = args.dropout,
    )
    model = Transformer(cfg).to(device)
    print(f"Model parameters: {model.num_params() / 1e6:.1f}M")

    # --- TensorBoard --------------------------------------------------------
    writer = SummaryWriter(log_dir=args.log_dir)
    writer.add_text("config/model", str(cfg), 0)
    writer.add_scalar("config/num_params_M", model.num_params() / 1e6, 0)

    # --- Dataset ------------------------------------------------------------
    print("Loading wikitext-103-raw-v1 ...")
    raw = load_dataset("iohadrubin/wikitext-103-raw-v1")

    train_ds = PackedTextDataset(raw["train"],      tokenizer, args.max_seq_len)
    val_ds   = PackedTextDataset(raw["validation"], tokenizer, args.max_seq_len)
    print(f"Train blocks: {len(train_ds):,}  |  Val blocks: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # --- Optimiser ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # --- Training -----------------------------------------------------------
    step          = 0
    accum_step    = 0
    running_loss  = 0.0
    t0            = time.time()

    optimizer.zero_grad()

    for x, y in _cycle(train_loader):
        if step >= args.max_steps:
            break

        x, y = x.to(device), y.to(device)

        logits = model(x)                        # (B, T, V)
        loss   = F.cross_entropy(
            logits.view(-1, vocab_size), y.view(-1)
        ) / args.grad_accum_steps

        loss.backward()
        running_loss += loss.item()
        accum_step   += 1

        if accum_step < args.grad_accum_steps:
            continue

        # --- Gradient step --------------------------------------------------
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            ppl     = math.exp(min(running_loss, 20))
            print(f"step {step:6d}  loss {running_loss:.4f}  ppl {ppl:.2f}  "
                  f"lr {lr:.2e}  grad_norm {grad_norm:.3f}  {elapsed:.0f}s")
            writer.add_scalar("train/loss",      running_loss,     step)
            writer.add_scalar("train/ppl",       ppl,              step)
            writer.add_scalar("train/lr",        lr,               step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            running_loss = 0.0
            t0 = time.time()

        # Validation
        if step % args.eval_interval == 0 and step > 0:
            val_loss = evaluate(model, val_loader, device, vocab_size, max_batches=50)
            val_ppl  = math.exp(min(val_loss, 20))
            print(f"  [val]  loss {val_loss:.4f}  ppl {val_ppl:.2f}")
            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("val/ppl",  val_ppl,  step)

        # Checkpoint
        if step % args.save_interval == 0 and step > 0:
            save_path = Path(args.out_dir) / f"pretrain_step{step}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"step": step, "model": model.state_dict(),
                        "cfg": cfg, "optimizer": optimizer.state_dict()}, save_path)
            print(f"  Saved checkpoint: {save_path}")

        step       += 1
        accum_step  = 0

    writer.close()

    # Final model
    save_path = Path(args.out_dir) / "pretrained.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": model.state_dict(), "cfg": cfg}, save_path)
    print(f"Training complete. Model saved: {save_path}")


@torch.no_grad()
def evaluate(model: Transformer, loader: DataLoader,
             device: torch.device, vocab_size: int, max_batches: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


def _cycle(loader: DataLoader):
    """Cycle through a DataLoader indefinitely."""
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train the transformer on wikitext-103")
    p.add_argument("--max_seq_len",     type=int,   default=4096)
    p.add_argument("--d_model",         type=int,   default=512)
    p.add_argument("--n_heads",         type=int,   default=8)
    p.add_argument("--n_layers",        type=int,   default=6)
    p.add_argument("--dropout",         type=float, default=0.0)
    p.add_argument("--batch_size",      type=int,   default=2)
    p.add_argument("--grad_accum_steps",type=int,   default=8,
                   help="Effective batch = batch_size * grad_accum_steps")
    p.add_argument("--max_steps",       type=int,   default=20_000)
    p.add_argument("--warmup_steps",    type=int,   default=500)
    p.add_argument("--max_lr",          type=float, default=3e-4)
    p.add_argument("--min_lr",          type=float, default=3e-5)
    p.add_argument("--weight_decay",    type=float, default=0.1)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)
    p.add_argument("--log_interval",    type=int,   default=50)
    p.add_argument("--eval_interval",   type=int,   default=500)
    p.add_argument("--save_interval",   type=int,   default=2000)
    p.add_argument("--num_workers",     type=int,   default=2)
    p.add_argument("--out_dir",         type=str,   default="models")
    p.add_argument("--log_dir",         type=str,   default="runs/pretrain")
    p.add_argument("--device",          type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
