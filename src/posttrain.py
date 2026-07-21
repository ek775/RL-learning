"""
Post-training via TRL: DPO, reward-model training, GRPO, and RLOO on
Intel/orca_dpo_pairs.

Methods
-------
dpo (default)
    Direct Preference Optimisation via TRL's DPOTrainer. Intel/orca_dpo_pairs
    already provides (prompt, chosen, rejected) triples — exactly what DPO
    needs. DPO *directly* optimises the policy without a separate reward
    model or value function, making it simpler and more stable than PPO for
    this setting.

    L_DPO = -E[ log σ( β · (log π_θ(y_w|x) - log π_ref(y_w|x))
                         - β · (log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]

    where y_w = chosen response, y_l = rejected response, β = temperature.

reward_model
    Trains the scalar `RewardModel` (see reward_model.py) on the same
    (chosen, rejected) pairs using the Bradley-Terry pairwise loss:
        loss = -log σ(r_chosen - r_rejected)
    This checkpoint is a prerequisite for `grpo` and `rloo` below, since both
    are online RL methods that need a reward signal for sampled completions.

grpo
    Group Relative Policy Optimisation via TRL's GRPOTrainer. Samples a
    group of completions per prompt, scores them with the trained reward
    model, and uses the group-normalised reward as the advantage — no
    critic/value model needed.

rloo
    REINFORCE Leave-One-Out via TRL's RLOOTrainer. Similar to GRPO (group
    sampling + reward model scoring), but uses a leave-one-out baseline
    instead of normalising by the group's standard deviation.

Dataset fields (Intel/orca_dpo_pairs)
--------------------------------------
  system   : system instruction
  question : user query
  chosen   : preferred assistant response
  rejected : dis-preferred assistant response

Run
---
    python -m rl.posttrain --method dpo           --pretrain_ckpt models/pretrained.pt
    python -m rl.posttrain --method reward_model   --pretrain_ckpt models/pretrained.pt
    python -m rl.posttrain --method grpo           --pretrain_ckpt models/pretrained.pt --reward_model_ckpt models/reward_model/reward_model.pt
    python -m rl.posttrain --method rloo           --pretrain_ckpt models/pretrained.pt --reward_model_ckpt models/reward_model/reward_model.pt
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutput
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, RLOOConfig, RLOOTrainer

from rl.reward_model import RewardModel
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


class TransformerForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Thin HuggingFace-compatible wrapper around the custom Transformer.

    Exposes the interface expected by TRL's trainers:
      - forward() returns CausalLMOutputWithPast with a .logits field
      - get/set_input_embeddings and get/set_output_embeddings
      - generate() (via GenerationMixin) for GRPO/RLOO rollout sampling

    This model has no KV-cache support (the inner Transformer always
    recomputes over the full sequence — see Transformer.forward), so callers
    that use generate() (GRPOTrainer/RLOOTrainer) must disable caching via
    generation_kwargs={"use_cache": False}. With caching off, HF's default
    `prepare_inputs_for_generation` always forwards the full growing
    input_ids each step, which matches this model's stateless forward.
    """

    config_class       = TransformerHFConfig
    # In this version of transformers, _get_tied_weight_keys iterates
    # tied.keys(), so _tied_weights_keys must be a dict, not a list.
    # Key = the weight path to drop from the saved state dict (the dependent copy).
    _tied_weights_keys = {"model.lm_head.weight": None}

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

    def tie_weights(self) -> None:
        # embed and lm_head share the same weight tensor (see TransformerConfig.tie_embeddings).
        # Declaring this explicitly lets HuggingFace save only one copy instead of
        # raising a RuntimeError about "shared tensors not properly defined".
        self.model.lm_head.weight = self.model.embed.weight

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


def load_reward_model(ckpt_path: str, pad_token_id: int) -> RewardModel:
    """
    Load a RewardModel from a checkpoint produced by `train_reward_model()`.
    The checkpoint is expected to contain {"model": state_dict, "cfg": TransformerConfig}.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg: TransformerConfig = ckpt["cfg"]
    reward_model = RewardModel(cfg, pad_token_id=pad_token_id)
    reward_model.load_state_dict(ckpt["model"])
    return reward_model


class RewardModelWrapper(nn.Module):
    """
    Adapts `RewardModel` to the `nn.Module` reward-function interface expected
    by TRL's GRPOTrainer/RLOOTrainer: forward(input_ids, attention_mask) must
    return an object with a `.logits` field of shape (batch, 1).

    Passed as an `nn.Module` (rather than a `PreTrainedModel`) so the trainer
    does not try to auto-derive a tokenizer from it — the caller must supply
    `reward_processing_classes=tokenizer` explicitly.
    """

    def __init__(self, reward_model: RewardModel, name: str = "custom_reward_model"):
        super().__init__()
        self.reward_model = reward_model
        # `.config` only needs a `_name_or_path` attribute — used by TRL purely
        # for logging/naming the reward function, never for model loading.
        self.config = SimpleNamespace(_name_or_path=name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        rewards = self.reward_model(input_ids, attention_mask)   # (batch,)
        return SequenceClassifierOutput(logits=rewards.unsqueeze(-1))


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

        # No trailing space on the prompt: BPE tokenizers (e.g. GPT-2) merge a
        # trailing space with the first word of the response into a single token
        # (e.g. " Hello"), so tokenize(prompt) ≠ tokenize(prompt+chosen)[:n].
        # Stripping the space here and prepending it to each response keeps the
        # tokenization boundary unambiguous.
        prompt = (f"{system}\n" if system else "") + f"User: {question}\nAssistant:"
        rows["prompt"].append(prompt)
        rows["chosen"].append(" " + chosen)
        rows["rejected"].append(" " + rejected)

    return Dataset.from_dict(rows)


def build_prompt_dataset(hf_split) -> Dataset:
    """
    Convert Intel/orca_dpo_pairs rows into a prompt-only dataset for online RL
    (GRPO/RLOO): {prompt}. The chosen/rejected completions aren't needed here
    — the policy generates its own completions, which are then scored by the
    reward model.
    """
    rows: dict[str, list[str]] = {"prompt": []}
    for row in hf_split:
        system   = (row.get("system")   or "").strip()
        question = (row.get("question") or "").strip()
        prompt = (f"{system}\n" if system else "") + f"User: {question}\nAssistant:"
        rows["prompt"].append(prompt)

    return Dataset.from_dict(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-training via TRL: DPO, reward-model training, GRPO, and RLOO "
                    "on Intel/orca_dpo_pairs"
    )
    p.add_argument("--method", type=str, default="dpo",
                   choices=["dpo", "reward_model", "grpo", "rloo"],
                   help="Post-training method to run")

    # --- Shared model / optimisation args ------------------------------------
    p.add_argument("--pretrain_ckpt", type=str,   default="models/pretrained.pt",
                   help="Path to pre-trained checkpoint (optional)")
    p.add_argument("--max_seq_len",   type=int,   default=1024,
                   help="Max tokens per sequence (prompt+response). Lower = faster.")
    p.add_argument("--d_model",       type=int,   default=512)
    p.add_argument("--n_heads",       type=int,   default=8)
    p.add_argument("--n_layers",      type=int,   default=6)
    p.add_argument("--epochs",        type=int,   default=1)
    p.add_argument("--max_steps",     type=int,   default=10_000)
    p.add_argument("--batch_size",    type=int,   default=2)
    p.add_argument("--lr",            type=float, default=5e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_interval",  type=int,   default=50)
    p.add_argument("--eval_interval", type=int,   default=500)
    p.add_argument("--save_interval", type=int,   default=2000)
    p.add_argument("--gpu",           type=int,   default=0,
                   help="CUDA/ROCm device index for the dedicated GPU (default: 0)")
    p.add_argument("--out_dir",       type=str,   default=None,
                   help="Checkpoint output directory. Defaults to models/<method>.")
    p.add_argument("--log_dir",       type=str,   default=None,
                   help="TensorBoard log directory. Defaults to runs/<method>.")
    p.add_argument("--seed",          type=int,   default=42)

    # --- DPO / GRPO / RLOO --------------------------------------------------
    p.add_argument("--beta",          type=float, default=0.1,
                   help="DPO temperature β, or GRPO/RLOO KL-penalty coefficient β")

    # --- GRPO / RLOO (online RL) --------------------------------------------
    p.add_argument("--reward_model_ckpt", type=str, default="models/reward_model/reward_model.pt",
                   help="Path to a trained reward-model checkpoint (see --method reward_model)")
    p.add_argument("--num_generations", type=int, default=4,
                   help="Number of completions sampled per prompt")
    p.add_argument("--max_completion_length", type=int, default=128,
                   help="Max new tokens generated per completion")
    p.add_argument("--temperature",   type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--top_p",         type=float, default=1.0)
    p.add_argument("--top_k",         type=int,   default=None)

    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = f"models/{args.method}"
    if args.log_dir is None:
        args.log_dir = f"runs/{args.method}"
    return args


# ---------------------------------------------------------------------------
# Method: dpo
# ---------------------------------------------------------------------------

def train_dpo(args: argparse.Namespace) -> None:
    # --- Device restriction -------------------------------------------------
    # On multi-GPU servers TRL/accelerate will spread policy and ref_model
    # across all visible devices.  Pin to the dedicated GPU before any
    # CUDA/ROCm context is created so accelerate only ever sees one device.
    # HIP_VISIBLE_DEVICES is the AMD/ROCm equivalent of CUDA_VISIBLE_DEVICES.
    gpu = str(args.gpu)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    os.environ.setdefault("HIP_VISIBLE_DEVICES",  gpu)

    # --- Tokenizer ----------------------------------------------------------
    # Must match pretrain.py exactly so the token IDs embedded in the
    # pre-trained weights are interpreted consistently:
    #   BOS = EOS = PAD = <|endoftext|> (id 50256)
    #   add_bos_token = False  (GPT-2 default — no BOS prepended)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len  # suppress the >1024-token warning

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
#        max_prompt_length            = args.max_seq_len // 2,
        # Gradient checkpointing requires HF-native module internals; our custom
        # TransformerBlock layers are not compatible, so disable it explicitly.
        gradient_checkpointing       = False,
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


# ---------------------------------------------------------------------------
# Method: reward_model
# ---------------------------------------------------------------------------

def _tokenize_reward_pairs(hf_split, tokenizer, max_len: int) -> Dataset:
    """
    Tokenise (prompt+chosen) and (prompt+rejected) text pairs (from
    `build_dpo_dataset`) into truncated token-id lists for reward-model
    training. Padding is deferred to collation, since sequences in different
    batches need different amounts of it.
    """
    bt = tokenizer.backend_tokenizer

    def _tokenize(batch):
        chosen_texts   = [p + c for p, c in zip(batch["prompt"], batch["chosen"])]
        rejected_texts = [p + r for p, r in zip(batch["prompt"], batch["rejected"])]
        chosen_ids   = [e.ids[:max_len] for e in bt.encode_batch(chosen_texts)]
        rejected_ids = [e.ids[:max_len] for e in bt.encode_batch(rejected_texts)]
        return {"chosen_ids": chosen_ids, "rejected_ids": rejected_ids}

    return hf_split.map(_tokenize, batched=True, remove_columns=hf_split.column_names, desc="Tokenizing")


def _reward_collate_fn(batch: list[dict]) -> dict[str, list[list[int]]]:
    return {
        "chosen_ids":   [item["chosen_ids"]   for item in batch],
        "rejected_ids": [item["rejected_ids"] for item in batch],
    }


def _pad_id_lists(id_lists: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch of variable-length token-id lists into a dense tensor + attention mask."""
    max_len = max(len(ids) for ids in id_lists)
    input_ids      = torch.full((len(id_lists), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(id_lists), max_len), dtype=torch.long)
    for i, ids in enumerate(id_lists):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    return input_ids.to(device), attention_mask.to(device)


def train_reward_model(args: argparse.Namespace) -> None:
    """
    Train the scalar `RewardModel` (Bradley-Terry pairwise loss) on
    Intel/orca_dpo_pairs. This checkpoint is a prerequisite for `--method
    grpo`/`--method rloo`, which need a reward signal for sampled completions.
    """
    gpu = str(args.gpu)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    os.environ.setdefault("HIP_VISIBLE_DEVICES",  gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len

    # --- Backbone -------------------------------------------------------------
    # Initialised from the same pre-trained weights as the policy so the
    # reward model starts from a language-competent representation.
    if args.pretrain_ckpt and Path(args.pretrain_ckpt).exists():
        ckpt = torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=False)
        cfg: TransformerConfig = ckpt["cfg"]
        backbone = Transformer(cfg)
        backbone.load_state_dict(ckpt["model"])
        print(f"Loaded pre-trained backbone from {args.pretrain_ckpt}")
    else:
        cfg = TransformerConfig(
            vocab_size  = tokenizer.vocab_size,
            max_seq_len = args.max_seq_len,
            d_model     = args.d_model,
            n_heads     = args.n_heads,
            n_layers    = args.n_layers,
        )
        backbone = Transformer(cfg)
        print("No checkpoint found — training reward model backbone from random init.")

    reward_model = RewardModel(cfg, pretrained=backbone, pad_token_id=tokenizer.pad_token_id).to(device)
    print(f"Reward model parameters: {sum(p.numel() for p in reward_model.parameters()) / 1e6:.1f}M")

    # --- Dataset ----------------------------------------------------------------
    print("Loading Intel/orca_dpo_pairs ...")
    raw = load_dataset("Intel/orca_dpo_pairs")

    train_hf = raw["train"]
    val_size = min(500, int(0.02 * len(train_hf)))
    val_hf   = train_hf.select(range(val_size))
    train_hf = train_hf.select(range(val_size, len(train_hf)))

    train_ds = _tokenize_reward_pairs(build_dpo_dataset(train_hf), tokenizer, args.max_seq_len)
    val_ds   = _tokenize_reward_pairs(build_dpo_dataset(val_hf),   tokenizer, args.max_seq_len)
    print(f"Train pairs: {len(train_ds):,}  |  Val pairs: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_reward_collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=_reward_collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=args.lr)
    writer    = SummaryWriter(log_dir=args.log_dir)

    # --- Training loop ------------------------------------------------------
    step = 0
    reward_model.train()
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break

            chosen_ids,   chosen_mask   = _pad_id_lists(batch["chosen_ids"],   tokenizer.pad_token_id, device)
            rejected_ids, rejected_mask = _pad_id_lists(batch["rejected_ids"], tokenizer.pad_token_id, device)

            loss, metrics = reward_model.preference_loss(
                chosen_ids, rejected_ids, chosen_mask, rejected_mask
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), args.max_grad_norm)
            optimizer.step()

            if step % args.log_interval == 0:
                print(f"step {step:6d}  loss {metrics['rm/loss']:.4f}  "
                      f"acc {metrics['rm/accuracy']:.3f}  margin {metrics['rm/reward_margin']:.3f}")
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)

            if step > 0 and step % args.eval_interval == 0:
                reward_model.eval()
                val_metrics: dict[str, list[float]] = {}
                with torch.no_grad():
                    for val_batch in val_loader:
                        v_chosen_ids,   v_chosen_mask   = _pad_id_lists(
                            val_batch["chosen_ids"],   tokenizer.pad_token_id, device
                        )
                        v_rejected_ids, v_rejected_mask = _pad_id_lists(
                            val_batch["rejected_ids"], tokenizer.pad_token_id, device
                        )
                        _, v_metrics = reward_model.preference_loss(
                            v_chosen_ids, v_rejected_ids, v_chosen_mask, v_rejected_mask
                        )
                        for name, value in v_metrics.items():
                            val_metrics.setdefault(name, []).append(value)
                for name, values in val_metrics.items():
                    avg = sum(values) / len(values)
                    writer.add_scalar(name.replace("rm/", "rm/val_"), avg, step)
                print(f"  [eval @ step {step}] loss {sum(val_metrics['rm/loss']) / len(val_metrics['rm/loss']):.4f}  "
                      f"acc {sum(val_metrics['rm/accuracy']) / len(val_metrics['rm/accuracy']):.3f}")
                reward_model.train()

            if step > 0 and step % args.save_interval == 0:
                ckpt_path = Path(args.out_dir) / f"reward_model_step{step}.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"step": step, "model": reward_model.state_dict(), "cfg": cfg}, ckpt_path)

            step += 1

    final_path = Path(args.out_dir) / "reward_model.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": reward_model.state_dict(), "cfg": cfg}, final_path)
    print(f"Training complete. Reward model saved: {final_path}")


# ---------------------------------------------------------------------------
# Methods: grpo / rloo (online RL, share setup)
# ---------------------------------------------------------------------------

def _setup_online_rl(
    args: argparse.Namespace,
) -> tuple[TransformerForCausalLM, "AutoTokenizer", RewardModelWrapper, Dataset, Dataset]:
    """Shared setup for GRPO and RLOO: policy, tokenizer, reward function, and prompt datasets."""
    gpu = str(args.gpu)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
    os.environ.setdefault("HIP_VISIBLE_DEVICES",  gpu)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len

    # --- Policy ---------------------------------------------------------------
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
    print(f"Model parameters: {policy.num_params() / 1e6:.1f}M")

    # --- Reward model (frozen) -------------------------------------------------
    if not Path(args.reward_model_ckpt).exists():
        raise FileNotFoundError(
            f"Reward model checkpoint not found at {args.reward_model_ckpt!r}. "
            "Train one first with `--method reward_model`."
        )
    reward_model = load_reward_model(args.reward_model_ckpt, tokenizer.pad_token_id)
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad_(False)
    # TRL only auto-places `PreTrainedModel` reward functions on the training
    # device; a plain `nn.Module` (used here for compiled-model compatibility)
    # must be moved there explicitly before the trainer is constructed.
    if torch.cuda.is_available():
        reward_model = reward_model.to("cuda")
    reward_func = RewardModelWrapper(reward_model)
    print(f"Loaded reward model from {args.reward_model_ckpt}")

    # --- Dataset (prompts only — completions are sampled online) ---------------
    print("Loading Intel/orca_dpo_pairs ...")
    raw = load_dataset("Intel/orca_dpo_pairs")

    train_hf = raw["train"]
    val_size = min(500, int(0.02 * len(train_hf)))
    val_hf   = train_hf.select(range(val_size))
    train_hf = train_hf.select(range(val_size, len(train_hf)))

    train_ds = build_prompt_dataset(train_hf)
    val_ds   = build_prompt_dataset(val_hf)
    print(f"Train prompts: {len(train_ds):,}  |  Val prompts: {len(val_ds):,}")

    return policy, tokenizer, reward_func, train_ds, val_ds


def _save_online_rl_checkpoint(policy: TransformerForCausalLM, out_dir: str, global_step: int) -> None:
    final_path = Path(out_dir) / "finetuned.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step":  global_step,
            "model": policy.model.state_dict(),
            "cfg":   policy.model.cfg,
        },
        final_path,
    )
    print(f"Training complete. Model saved: {final_path}")


def train_grpo(args: argparse.Namespace) -> None:
    """
    Group Relative Policy Optimisation (Shao et al., 2024) via TRL's
    GRPOTrainer. Samples `num_generations` completions per prompt, scores
    them with the reward model, and uses the group-normalised reward as the
    advantage — no critic/value model needed.
    """
    policy, tokenizer, reward_func, train_ds, val_ds = _setup_online_rl(args)

    grpo_config = GRPOConfig(
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
        num_generations              = args.num_generations,
        max_completion_length        = args.max_completion_length,
        temperature                  = args.temperature,
        top_p                        = args.top_p,
        top_k                        = args.top_k,
        # This model has no KV-cache support, so caching must stay disabled
        # (see TransformerForCausalLM docstring); rollouts recompute the full
        # sequence at every generation step.
        generation_kwargs            = {"use_cache": False},
        gradient_checkpointing       = False,
        dataloader_num_workers       = 0,
        remove_unused_columns        = False,
    )

    trainer = GRPOTrainer(
        model                    = policy,
        reward_funcs             = reward_func,
        args                     = grpo_config,
        train_dataset            = train_ds,
        eval_dataset             = val_ds,
        processing_class         = tokenizer,
        reward_processing_classes = tokenizer,
    )

    trainer.train()
    _save_online_rl_checkpoint(policy, args.out_dir, trainer.state.global_step)


def train_rloo(args: argparse.Namespace) -> None:
    """
    REINFORCE Leave-One-Out (Ahmadian et al., 2024) via TRL's RLOOTrainer.
    Like GRPO, samples `num_generations` completions per prompt and scores
    them with the reward model, but uses a leave-one-out baseline (each
    sample's advantage is its reward minus the mean of the *other* samples in
    its group) instead of normalising by the group's standard deviation.
    """
    policy, tokenizer, reward_func, train_ds, val_ds = _setup_online_rl(args)

    rloo_config = RLOOConfig(
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
        num_generations              = args.num_generations,
        max_completion_length        = args.max_completion_length,
        temperature                  = args.temperature,
        top_p                        = args.top_p,
        top_k                        = args.top_k,
        # This model has no KV-cache support, so caching must stay disabled
        # (see TransformerForCausalLM docstring); rollouts recompute the full
        # sequence at every generation step.
        generation_kwargs            = {"use_cache": False},
        gradient_checkpointing       = False,
        dataloader_num_workers       = 0,
        remove_unused_columns        = False,
    )

    trainer = RLOOTrainer(
        model                    = policy,
        reward_funcs             = reward_func,
        args                     = rloo_config,
        train_dataset            = train_ds,
        eval_dataset             = val_ds,
        processing_class         = tokenizer,
        reward_processing_classes = tokenizer,
    )

    trainer.train()
    _save_online_rl_checkpoint(policy, args.out_dir, trainer.state.global_step)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_METHODS = {
    "dpo":          train_dpo,
    "reward_model": train_reward_model,
    "grpo":         train_grpo,
    "rloo":         train_rloo,
}


if __name__ == "__main__":
    cli_args = parse_args()
    _METHODS[cli_args.method](cli_args)

