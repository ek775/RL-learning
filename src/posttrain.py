"""
Post-training: DPO (Direct Preference Optimisation) via TRL's DPOTrainer
on Intel/orca_dpo_pairs.

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

Dataset fields (Intel/orca_dpo_pairs)
--------------------------------------
  system   : system instruction
  question : user query
  chosen   : preferred assistant response
  rejected : dis-preferred assistant response

Run
---
    python -m rl.posttrain --pretrain_ckpt models/pretrained.pt
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from trl import DPOConfig, DPOTrainer

from rl.transformer import Transformer, TransformerConfig


# ---------------------------------------------------------------------------
# HuggingFace-compatible wrapper
# ---------------------------------------------------------------------------

class TransformerHFConfig(PretrainedConfig):
    """Serialisable config bridge between TransformerConfig and HuggingFace."""

    model_type = "custom_transformer"

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 8,
        n_layers: int = 6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.n_kv_heads  = n_kv_heads
        self.n_layers    = n_layers


class TransformerForCausalLM(PreTrainedModel):
    """
    Thin HuggingFace-compatible wrapper around the custom Transformer.

    Exposes the interface expected by TRL's DPOTrainer:
      - forward() returns CausalLMOutputWithPast with a .logits field
      - get/set_input_embeddings and get/set_output_embeddings
    """

    config_class = TransformerHFConfig

    def __init__(self, config: TransformerHFConfig):
        super().__init__(config)
        transformer_cfg = TransformerConfig(
            vocab_size  = config.vocab_size,
            max_seq_len = config.max_seq_len,
            d_model     = config.d_model,
            n_heads     = config.n_heads,
            n_kv_heads  = config.n_kv_heads,
            n_layers    = config.n_layers,
        )
        self.model = Transformer(transformer_cfg)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.model.embed

    def set_input_embeddings(self, value: torch.nn.Embedding) -> None:
        self.model.embed = value

    def get_output_embeddings(self) -> torch.nn.Linear:
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings: torch.nn.Linear) -> None:
        self.model.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # attention_mask is accepted for API compatibility but unused —
        # the inner Transformer uses built-in causal masking via SDPA.
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_from_pretrain_ckpt(ckpt_path: str) -> TransformerForCausalLM:
    """
    Load a TransformerForCausalLM from a pre-training checkpoint.
    The checkpoint is expected to contain {"model": state_dict, "cfg": TransformerConfig}.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: TransformerConfig = ckpt["cfg"]
    hf_config = TransformerHFConfig(
        vocab_size  = cfg.vocab_size,
        max_seq_len = cfg.max_seq_len,
        d_model     = cfg.d_model,
        n_heads     = cfg.n_heads,
        n_kv_heads  = cfg.n_kv_heads,
        n_layers    = cfg.n_layers,
    )
    model = TransformerForCausalLM(hf_config)
    model.model.load_state_dict(ckpt["model"])
    return model


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def build_dpo_dataset(hf_split) -> Dataset:
    """
    Convert Intel/orca_dpo_pairs rows into the plain-text format that
    TRL's DPOTrainer expects: {prompt, chosen, rejected}.

    TRL handles tokenisation internally, so we only need raw strings.
    """
    rows: dict[str, list[str]] = {"prompt": [], "chosen": [], "rejected": []}
    for row in hf_split:
        system   = (row.get("system")   or "").strip()
        question = (row.get("question") or "").strip()
        chosen   = (row.get("chosen")   or "").strip()
        rejected = (row.get("rejected") or "").strip()

        prompt = (f"{system}\n" if system else "") + f"User: {question}\nAssistant: "
        rows["prompt"].append(prompt)
        rows["chosen"].append(chosen)
        rows["rejected"].append(rejected)

    return Dataset.from_dict(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO post-training via TRL on Intel/orca_dpo_pairs")
    p.add_argument("--pretrain_ckpt", type=str,   default="models/pretrained.pt",
                   help="Path to pre-trained checkpoint (optional)")
    p.add_argument("--max_seq_len",   type=int,   default=1024,
                   help="Max tokens per sequence (prompt+response). Lower = faster.")
    p.add_argument("--d_model",       type=int,   default=512)
    p.add_argument("--n_heads",       type=int,   default=8)
    p.add_argument("--n_layers",      type=int,   default=6)
    p.add_argument("--beta",          type=float, default=0.1,
                   help="DPO temperature β — higher = stay closer to reference")
    p.add_argument("--epochs",        type=int,   default=1)
    p.add_argument("--max_steps",     type=int,   default=10_000)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--lr",            type=float, default=5e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_interval",  type=int,   default=50)
    p.add_argument("--eval_interval", type=int,   default=500)
    p.add_argument("--save_interval", type=int,   default=2000)
    p.add_argument("--out_dir",       type=str,   default="models/dpo")
    p.add_argument("--log_dir",       type=str,   default="runs/dpo")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    # --- Tokenizer ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # --- Model --------------------------------------------------------------
    if args.pretrain_ckpt and Path(args.pretrain_ckpt).exists():
        policy = load_from_pretrain_ckpt(args.pretrain_ckpt)
        print(f"Loaded pre-trained weights from {args.pretrain_ckpt}")
    else:
        hf_config = TransformerHFConfig(
            vocab_size  = tokenizer.vocab_size,
            max_seq_len = args.max_seq_len,
            d_model     = args.d_model,
            n_heads     = args.n_heads,
            n_layers    = args.n_layers,
        )
        policy = TransformerForCausalLM(hf_config)
        print("No checkpoint found — training from random init.")

    # Reference model: frozen copy of the starting policy.
    # Passed explicitly so TRL does not try to reload it from a Hub path.
    ref_model = copy.deepcopy(policy)
    for param in ref_model.parameters():
        param.requires_grad_(False)

    print(f"Model parameters: {policy.num_params() / 1e6:.1f}M")

    # --- Dataset ------------------------------------------------------------
    print("Loading Intel/orca_dpo_pairs ...")
    raw = load_dataset("Intel/orca_dpo_pairs")

    train_hf    = raw["train"]
    val_size    = min(500, int(0.02 * len(train_hf)))
    val_hf      = train_hf.select(range(val_size))
    train_hf    = train_hf.select(range(val_size, len(train_hf)))

    train_ds = build_dpo_dataset(train_hf)
    val_ds   = build_dpo_dataset(val_hf)
    print(f"Train pairs: {len(train_ds):,}  |  Val pairs: {len(val_ds):,}")

    # --- TRL DPOConfig ------------------------------------------------------
    # DPOConfig inherits from TrainingArguments; all standard HF training
    # knobs are available here.  TRL's DPOTrainer handles collation, the
    # reference-model forward pass, and logging — no manual DataLoader needed.
    dpo_config = DPOConfig(
        output_dir                   = args.out_dir,
        beta                         = args.beta,
        num_train_epochs             = args.epochs,
        max_steps                    = args.max_steps,
        per_device_train_batch_size  = args.batch_size,
        per_device_eval_batch_size   = args.batch_size,
        learning_rate                = args.lr,
        max_grad_norm                = args.max_grad_norm,
        logging_steps                = args.log_interval,
        eval_steps                   = args.eval_interval,
        save_steps                   = args.save_interval,
        eval_strategy                = "steps",
        save_strategy                = "steps",
        report_to                    = "tensorboard",
        logging_dir                  = args.log_dir,
        seed                         = args.seed,
        max_length                   = args.max_seq_len,
        max_prompt_length            = args.max_seq_len // 2,
        # Single-process data loading avoids multiprocessing pickling issues
        # that arise from Python 3.14's forkserver start method.
        dataloader_num_workers       = 0,
        remove_unused_columns        = False,
    )

    # --- Trainer ------------------------------------------------------------
    trainer = DPOTrainer(
        model            = policy,
        ref_model        = ref_model,
        args             = dpo_config,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        processing_class = tokenizer,
    )

    trainer.train()

    # Save final checkpoint in the original .pt format for compatibility
    # with other scripts (pretrain.py, ppo.py, etc.) that load via torch.load.
    final_path = Path(args.out_dir) / "finetuned.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step":  trainer.state.global_step,
            "model": policy.model.state_dict(),
            "cfg":   policy.model.cfg,
        },
        final_path,
    )
    print(f"Training complete. Model saved: {final_path}")


if __name__ == "__main__":
    train(parse_args())
