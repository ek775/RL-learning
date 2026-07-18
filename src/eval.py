"""
Evaluation: pretrained vs DPO fine-tuned model.

Metrics
-------
1. Perplexity          — WikiText-103 validation (base LM quality).
2. Preference accuracy — % of pairs where log π(chosen) > log π(rejected).
3. Log-prob margin     — mean (log π(chosen) − log π(rejected)).
4. Implicit reward     — mean (log π_dpo − log π_ref) on chosen and rejected;
                          the quantity DPO directly optimises.
5. Generation samples  — side-by-side qualitative comparison.

Interpretation guide
--------------------
  Preference accuracy > 50%          → model ranks chosen above rejected more
                                        than chance; higher is better.
  Log-prob margin (DPO > pretrained) → DPO widened the gap between chosen and
                                        rejected log-likelihoods.
  Implicit reward on chosen  > 0     → DPO moved chosen *above* the reference
                                        policy (correct direction).
  Implicit reward on rejected < 0    → DPO suppressed rejected responses
                                        relative to the reference (correct).
  Perplexity delta ≈ 0               → DPO did not cause catastrophic forgetting.

Run
---
    python -m rl.eval
    python -m rl.eval --pretrain_ckpt models/pretrained.pt \\
                      --dpo_ckpt      models/dpo/finetuned.pt \\
                      --device        cuda
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from rl.transformer import Transformer, TransformerConfig


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device) -> Transformer:
    """Load a Transformer from a .pt checkpoint (pretrain or DPO format)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: TransformerConfig = ckpt["cfg"]
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Metric 1 — Perplexity on WikiText-103
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: Transformer,
    tokenizer,
    device: torch.device,
    num_blocks: int = 200,
    block_size: int = 512,
) -> float:
    """
    Perplexity on `num_blocks` consecutive 512-token blocks from
    the WikiText-103 validation split (~100K tokens with the default settings).
    """
    raw = load_dataset("iohadrubin/wikitext-103-raw-v1", split="validation")
    eos = tokenizer.eos_token_id
    bt  = tokenizer.backend_tokenizer

    token_ids: list[int] = []
    for row in raw:
        ids = bt.encode(row["text"]).ids
        if ids:
            token_ids.extend(ids)
            token_ids.append(eos)

    data = torch.tensor(token_ids, dtype=torch.long)
    n    = min(num_blocks, (len(data) - 1) // block_size)

    total_nll = 0.0
    for i in range(n):
        start = i * block_size
        chunk = data[start : start + block_size + 1].unsqueeze(0).to(device)
        logits = model(chunk[:, :-1])
        nll = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_nll += nll.item()

    return math.exp(total_nll / (n * block_size))


# ---------------------------------------------------------------------------
# Helper — response log-probability
# ---------------------------------------------------------------------------

@torch.no_grad()
def response_log_prob(
    model: Transformer,
    full_ids: torch.Tensor,
    prompt_len: int,
    device: torch.device,
) -> float:
    """
    Sum of log π(token | context) over response tokens only.

    full_ids  : (1, prompt_len + response_len) — prompt + response token ids.
    prompt_len: number of prompt tokens (excluded from the log-prob sum).

    How the indexing works
    ~~~~~~~~~~~~~~~~~~~~~~
    logits[t] predicts token at position t+1 in input_ids.
    The first response token sits at index `prompt_len` in input_ids, so it is
    predicted by logits[prompt_len - 1], i.e. index prompt_len-1 in the shifted
    token_lps tensor.
    """
    x         = full_ids.to(device)
    logits    = model(x)                                        # (1, T, V)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)          # (T-1, V)
    targets   = x[0, 1:]                                       # (T-1,)
    token_lps = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (T-1,)
    return token_lps[prompt_len - 1 :].sum().item()


# ---------------------------------------------------------------------------
# Metrics 2–4 — Preference evaluation on Intel/orca_dpo_pairs
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_preferences(
    pretrain_model: Transformer,
    dpo_model: Transformer,
    tokenizer,
    device: torch.device,
    num_pairs: int = 200,
) -> dict:
    """
    Evaluate both models on held-out DPO preference pairs.

    Uses the same validation split as posttrain.py:
        first min(500, 2% of train) rows of Intel/orca_dpo_pairs["train"].

    Returns
    -------
    n_pairs                  : number of pairs actually evaluated
    pretrain_acc / dpo_acc   : preference accuracy in percent
    pretrain_margin / dpo_margin : mean (log π(chosen) − log π(rejected))
    implicit_reward_chosen   : mean (log π_dpo − log π_ref) on chosen responses
    implicit_reward_rejected : mean (log π_dpo − log π_ref) on rejected responses
    """
    raw      = load_dataset("Intel/orca_dpo_pairs")["train"]
    val_size = min(500, int(0.02 * len(raw)))
    pairs    = raw.select(range(min(num_pairs, val_size)))
    max_len  = pretrain_model.cfg.max_seq_len

    pre_correct = 0
    dpo_correct = 0
    pre_margins:   list[float] = []
    dpo_margins:   list[float] = []
    ir_chosen:     list[float] = []
    ir_rejected:   list[float] = []
    skipped = 0

    for i, row in enumerate(pairs):
        system   = (row.get("system")   or "").strip()
        question = (row.get("question") or "").strip()
        chosen   = (row.get("chosen")   or "").strip()
        rejected = (row.get("rejected") or "").strip()

        # Replicate the prompt/response format used in posttrain.py exactly
        prompt_str   = (f"{system}\n" if system else "") + f"User: {question}\nAssistant:"
        prompt_ids   = tokenizer(prompt_str,      add_special_tokens=False, return_tensors="pt").input_ids
        chosen_ids   = tokenizer(" " + chosen,    add_special_tokens=False, return_tensors="pt").input_ids
        rejected_ids = tokenizer(" " + rejected,  add_special_tokens=False, return_tensors="pt").input_ids

        p_len  = prompt_ids.size(1)
        full_c = torch.cat([prompt_ids, chosen_ids],   dim=1)[:, :max_len]
        full_r = torch.cat([prompt_ids, rejected_ids], dim=1)[:, :max_len]

        # Skip if the prompt fills the entire context window
        if full_c.size(1) <= p_len or full_r.size(1) <= p_len:
            skipped += 1
            continue

        pre_lp_c = response_log_prob(pretrain_model, full_c, p_len, device)
        pre_lp_r = response_log_prob(pretrain_model, full_r, p_len, device)
        dpo_lp_c = response_log_prob(dpo_model,      full_c, p_len, device)
        dpo_lp_r = response_log_prob(dpo_model,      full_r, p_len, device)

        pre_correct += int(pre_lp_c > pre_lp_r)
        dpo_correct += int(dpo_lp_c > dpo_lp_r)
        pre_margins.append(pre_lp_c - pre_lp_r)
        dpo_margins.append(dpo_lp_c - dpo_lp_r)
        ir_chosen.append(dpo_lp_c - pre_lp_c)
        ir_rejected.append(dpo_lp_r - pre_lp_r)

        if (i + 1) % 50 == 0:
            n_so_far = len(pre_margins)
            print(f"  [{i + 1:>3}/{len(pairs)}]  "
                  f"pretrain_acc={pre_correct / n_so_far:.1%}  "
                  f"dpo_acc={dpo_correct / n_so_far:.1%}")

    n = len(pre_margins)
    if n == 0:
        raise RuntimeError(
            "All pairs were skipped — sequences too long for max_seq_len. "
            "Try reducing --num_pairs or increasing max_seq_len."
        )

    if skipped:
        print(f"  Skipped {skipped} pairs (prompt exceeded context window).")

    return {
        "n_pairs":                  n,
        "pretrain_acc":             100 * pre_correct / n,
        "dpo_acc":                  100 * dpo_correct / n,
        "pretrain_margin":          sum(pre_margins) / n,
        "dpo_margin":               sum(dpo_margins) / n,
        "implicit_reward_chosen":   sum(ir_chosen) / n,
        "implicit_reward_rejected": sum(ir_rejected) / n,
    }


# ---------------------------------------------------------------------------
# Metric 5 — Qualitative generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def print_generation_samples(
    pretrain_model: Transformer,
    dpo_model: Transformer,
    tokenizer,
    device: torch.device,
    num_samples: int = 3,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_k: int = 50,
) -> None:
    """Print side-by-side greedy/sampled generations from both models."""
    raw      = load_dataset("Intel/orca_dpo_pairs")["train"]
    val_size = min(500, int(0.02 * len(raw)))
    step     = max(1, val_size // num_samples)

    for s in range(num_samples):
        row      = raw[s * step]
        system   = (row.get("system")   or "").strip()
        question = (row.get("question") or "").strip()
        chosen   = (row.get("chosen")   or "").strip()

        prompt_str = (f"{system}\n" if system else "") + f"User: {question}\nAssistant:"
        prompt_ids = tokenizer(
            prompt_str, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)
        p_len = prompt_ids.size(1)

        pre_out = pretrain_model.generate(
            prompt_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
        )
        dpo_out = dpo_model.generate(
            prompt_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
        )

        pre_text = tokenizer.decode(pre_out[0, p_len:], skip_special_tokens=True).strip()
        dpo_text = tokenizer.decode(dpo_out[0, p_len:], skip_special_tokens=True).strip()
        ref_text = chosen[:300]

        print(f"\n{'─' * 70}")
        print(f"PROMPT     : {question[:120]}")
        print(f"\nPRETRAINED ▶ {pre_text}")
        print(f"\nDPO        ▶ {dpo_text}")
        print(f"\nREFERENCE  ▶ {ref_text}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate pretrained vs DPO model")
    p.add_argument("--pretrain_ckpt", default="models/pretrained.pt",
                   help="Path to pre-training checkpoint.")
    p.add_argument("--dpo_ckpt",      default="models/dpo/finetuned.pt",
                   help="Path to DPO fine-tuned checkpoint.")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_pairs",     type=int, default=200,
                   help="Preference pairs to evaluate (max 500, the val-set size).")
    p.add_argument("--num_blocks",    type=int, default=200,
                   help="512-token blocks for perplexity evaluation (~100K tokens).")
    p.add_argument("--num_samples",   type=int, default=3,
                   help="Qualitative generation examples to print.")
    p.add_argument("--skip_ppl",      action="store_true",
                   help="Skip perplexity (saves ~1 min on CPU).")
    p.add_argument("--skip_gen",      action="store_true",
                   help="Skip qualitative generation.")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Device : {device}")
    print(f"Loading pretrained  → {args.pretrain_ckpt}")
    pretrain_model = load_model(args.pretrain_ckpt, device)
    print(f"Loading DPO         → {args.dpo_ckpt}")
    dpo_model      = load_model(args.dpo_ckpt, device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token           = tokenizer.eos_token
    tokenizer.model_max_length    = pretrain_model.cfg.max_seq_len

    # ── 1. Perplexity ──────────────────────────────────────────────────────
    if not args.skip_ppl:
        print(f"\n── Perplexity  (WikiText-103 val, {args.num_blocks} × 512 tokens) ────────")
        print("  pretrained ...")
        pre_ppl = compute_perplexity(pretrain_model, tokenizer, device, num_blocks=args.num_blocks)
        print("  DPO ...")
        dpo_ppl = compute_perplexity(dpo_model,      tokenizer, device, num_blocks=args.num_blocks)
        ppl_delta = dpo_ppl - pre_ppl
    else:
        pre_ppl = dpo_ppl = ppl_delta = None

    # ── 2–4. Preference metrics ────────────────────────────────────────────
    print(f"\n── Preference evaluation  (Intel/orca_dpo_pairs, {args.num_pairs} pairs) ──")
    metrics = evaluate_preferences(
        pretrain_model, dpo_model, tokenizer, device, num_pairs=args.num_pairs,
    )

    # ── Results table ──────────────────────────────────────────────────────
    W = 70
    print("\n" + "═" * W)
    print(f"{'METRIC':<42} {'PRETRAINED':>10} {'DPO':>10}")
    print("─" * W)

    if not args.skip_ppl:
        delta_str = f"  Δ {'+' if ppl_delta >= 0 else ''}{ppl_delta:.1f}"
        print(f"{'Perplexity (WikiText-103 val)':<42} {pre_ppl:>10.2f} {dpo_ppl:>10.2f}{delta_str}")

    print(f"{'Preference accuracy (%)':<42} {metrics['pretrain_acc']:>9.1f}% {metrics['dpo_acc']:>9.1f}%")
    print(f"{'Log-prob margin (chosen − rejected)':<42} {metrics['pretrain_margin']:>10.2f} {metrics['dpo_margin']:>10.2f}")
    print(f"{'Implicit reward  chosen   (log π_dpo − π_ref)':<42} {'—':>10} {metrics['implicit_reward_chosen']:>10.2f}")
    print(f"{'Implicit reward  rejected (log π_dpo − π_ref)':<42} {'—':>10} {metrics['implicit_reward_rejected']:>10.2f}")
    ir_gap = metrics["implicit_reward_chosen"] - metrics["implicit_reward_rejected"]
    print(f"{'Implicit reward gap (chosen − rejected)':<42} {'—':>10} {ir_gap:>10.2f}")
    print("─" * W)
    print(f"  Evaluated on {metrics['n_pairs']} preference pairs.\n")

    # Diagnostics
    if not args.skip_ppl:
        if ppl_delta < 5:
            print("  ✓ DPO preserved base LM quality (perplexity stable).")
        else:
            print("  ⚠ DPO raised perplexity — possible catastrophic forgetting.")

    if metrics["dpo_acc"] > metrics["pretrain_acc"]:
        print("  ✓ DPO improved preference accuracy.")
    else:
        print("  ⚠ DPO did not improve preference accuracy — inspect training.")

    ir_c = metrics["implicit_reward_chosen"]
    ir_r = metrics["implicit_reward_rejected"]
    if ir_c > 0 and ir_r < 0:
        print("  ✓ DPO correctly up-weights chosen and down-weights rejected.")
    elif ir_c > ir_r:
        print("  ~ Implicit reward gap points in the right direction but margin is small.")
    else:
        print("  ⚠ Implicit reward direction unexpected — inspect DPO training.")

    print("═" * W)

    # ── 5. Generation samples ──────────────────────────────────────────────
    if not args.skip_gen:
        print(f"\n── Generation samples  ({args.num_samples} examples, temp=0.7, top-k=50) ──")
        print_generation_samples(
            pretrain_model, dpo_model, tokenizer, device,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()
