"""
Part 2: Transformer Scaling Study
Trains decoder-only transformers of 5 sizes on SVG data.
Includes LR sweep on tiny model, then trains all sizes.

Usage:
  # Step 1: LR sweep on tiny model
  python part2_train.py --mode lr_sweep

  # Step 2: Train all 5 models with best LR
  python part2_train.py --mode train_all --lr 3e-3  # replace with best LR from sweep
"""

import os
import math
import time
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
RUNS_DIR   = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"Using device: {DEVICE}")

# Training hyperparameters (fixed across all models)
BATCH_SIZE    = 32          # sequences per batch
BLOCK_SIZE    = 512         # context window (tokens)
GRAD_CLIP     = 1.0
WARMUP_ITERS  = 200
WEIGHT_DECAY  = 0.1
BETA1, BETA2  = 0.9, 0.95
VOCAB_SIZE    = 4096

# LR sweep values to test on tiny model
LR_SWEEP_VALUES = [1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2]

# ── Model Configurations ──────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    name:       str
    n_layer:    int
    n_head:     int
    n_embd:     int
    vocab_size: int = VOCAB_SIZE
    block_size: int = BLOCK_SIZE
    dropout:    float = 0.0
    bias:       bool = False

    @property
    def n_params(self):
        # approximate
        return (
            self.vocab_size * self.n_embd +           # embedding
            self.n_layer * (
                4 * self.n_embd * self.n_embd +       # attn QKV + proj
                2 * self.n_embd * (4 * self.n_embd) + # MLP
                2 * self.n_embd                        # layernorms
            ) +
            self.n_embd * self.vocab_size              # lm head
        )

MODEL_CONFIGS = [
    ModelConfig("tiny",   n_layer=4,  n_head=4,  n_embd=128),   # ~1M
    ModelConfig("small",  n_layer=6,  n_head=6,  n_embd=192),   # ~3M
    ModelConfig("medium", n_layer=6,  n_head=6,  n_embd=384),   # ~10M
    ModelConfig("large",  n_layer=10, n_head=8,  n_embd=512),   # ~30M
    ModelConfig("xl",     n_layer=12, n_head=12, n_embd=768),   # ~88M
]

# ── Transformer Model ─────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj  = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd
        self.dropout = cfg.dropout
        self.register_buffer("bias", torch.tril(
            torch.ones(cfg.block_size, cfg.block_size)
        ).view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Use Flash Attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe  = nn.Embedding(cfg.block_size, cfg.n_embd),
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos  = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

# ── Data Loading ──────────────────────────────────────────────────────────────
class SVGDataset:
    def __init__(self, split: str, block_size: int):
        tokens = np.load(DATA_DIR / split / "tokens.npy", mmap_mode='r')
        self.data = torch.from_numpy(tokens.astype(np.int64))
        self.block_size = block_size
        print(f"  {split}: {len(self.data):,} tokens")

    def __len__(self):
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int, device: str):
        ix = torch.randint(len(self) - 1, (batch_size,))
        x  = torch.stack([self.data[i:i+self.block_size]         for i in ix])
        y  = torch.stack([self.data[i+1:i+self.block_size+1]     for i in ix])
        return x.to(device), y.to(device)

# ── LR Schedule ───────────────────────────────────────────────────────────────
def get_lr(it: int, lr: float, warmup_iters: int, lr_decay_iters: int) -> float:
    if it < warmup_iters:
        return lr * it / warmup_iters
    if it > lr_decay_iters:
        return lr * 0.1
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr * 0.1 + coeff * (lr - lr * 0.1)

# ── Training Loop ─────────────────────────────────────────────────────────────
def train(cfg: ModelConfig, lr: float, max_iters: int, run_name: str,
          eval_interval: int = 200, eval_iters: int = 50):

    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"  Params:     {cfg.count_params() if hasattr(cfg, 'count_params') else 0:,}")
    print(f"  LR:         {lr}")
    print(f"  Max iters:  {max_iters}")
    print(f"{'='*60}")

    # Data
    train_data = SVGDataset("train", BLOCK_SIZE)
    val_data   = SVGDataset("val",   BLOCK_SIZE)

    # Model
    model = GPT(cfg).to(DEVICE)
    n_params = model.count_params()
    print(f"  Actual params: {n_params:,}")

    # Optimizer
    # Separate weight decay params
    decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(BETA1, BETA2))

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = {}
        for split, dataset in [("train", train_data), ("val", val_data)]:
            ls = []
            for _ in range(eval_iters):
                x, y = dataset.get_batch(BATCH_SIZE, DEVICE)
                _, loss = model(x, y)
                ls.append(loss.item())
            losses[split] = float(np.mean(ls))
        model.train()
        return losses

    # Training
    history = {"iter": [], "train_loss": [], "val_loss": [], "lr": []}
    t0 = time.time()
    best_val_loss = float('inf')

    for it in range(max_iters + 1):
        # Eval
        if it % eval_interval == 0:
            losses = estimate_loss()
            elapsed = time.time() - t0
            print(f"  iter {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | {elapsed:.0f}s")
            history["iter"].append(it)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])
            history["lr"].append(get_lr(it, lr, WARMUP_ITERS, max_iters))
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(model.state_dict(), run_dir / "best_model.pt")

        if it == max_iters:
            break

        # LR update
        current_lr = get_lr(it, lr, WARMUP_ITERS, max_iters)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Forward + backward
        x, y = train_data.get_batch(BATCH_SIZE, DEVICE)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    wall_time = time.time() - t0
    tokens_per_sec = (max_iters * BATCH_SIZE * BLOCK_SIZE) / wall_time

    results = {
        "run_name":       run_name,
        "model_name":     cfg.name,
        "n_params":       n_params,
        "lr":             lr,
        "max_iters":      max_iters,
        "best_val_loss":  best_val_loss,
        "wall_time_s":    wall_time,
        "tokens_per_sec": tokens_per_sec,
        "history":        history,
        "n_layer":        cfg.n_layer,
        "n_head":         cfg.n_head,
        "n_embd":         cfg.n_embd,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  [DONE] Done! Best val loss: {best_val_loss:.4f} | "
          f"Time: {wall_time/60:.1f}min | {tokens_per_sec:.0f} tok/s")
    return results

# ── Estimate iters for 1 epoch ────────────────────────────────────────────────
def iters_per_epoch():
    tokens = np.load(DATA_DIR / "train" / "tokens.npy", mmap_mode='r')
    n_tokens = len(tokens)
    return n_tokens // (BATCH_SIZE * BLOCK_SIZE)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lr_sweep", "train_all"], required=True)
    parser.add_argument("--lr",   type=float, default=3e-3,
                        help="Learning rate (used in train_all mode)")
    args = parser.parse_args()

    n_iters = iters_per_epoch()
    print(f"Estimated iters per epoch: {n_iters:,}")

    if args.mode == "lr_sweep":
        print("\n" + "="*60)
        print("LR SWEEP on TINY model")
        print("="*60)
        cfg = MODEL_CONFIGS[0]  # tiny
        sweep_results = []
        # Use 20% of epoch for sweep (much faster)
        sweep_iters = max(500, n_iters // 5)
        print(f"Sweep iters per run: {sweep_iters} (~20% of epoch)")

        for lr in LR_SWEEP_VALUES:
            run_name = f"lr_sweep_tiny_lr{lr:.0e}"
            r = train(cfg, lr, sweep_iters, run_name, eval_interval=100, eval_iters=25)
            sweep_results.append({"lr": lr, "val_loss": r["best_val_loss"]})

        print("\n" + "="*60)
        print("LR SWEEP RESULTS:")
        print("="*60)
        for r in sweep_results:
            marker = " ← BEST" if r["val_loss"] == min(x["val_loss"] for x in sweep_results) else ""
            print(f"  lr={r['lr']:.0e}  val_loss={r['val_loss']:.4f}{marker}")

        best_lr = min(sweep_results, key=lambda x: x["val_loss"])["lr"]
        print(f"\nBest LR: {best_lr}")
        print(f"\nNext step:")
        print(f"  python part2_train.py --mode train_all --lr {best_lr}")

        with open(RUNS_DIR / "lr_sweep_results.json", "w") as f:
            json.dump({"sweep_results": sweep_results, "best_lr": best_lr}, f, indent=2)

    elif args.mode == "train_all":
        lr = args.lr
        print(f"\nTraining all 5 models with lr={lr}")
        print(f"Iters per epoch: {n_iters:,}")

        all_results = []
        for cfg in MODEL_CONFIGS:
            run_name = f"scale_{cfg.name}_lr{lr:.0e}"
            r = train(cfg, lr, n_iters, run_name, eval_interval=500, eval_iters=50)
            all_results.append(r)

        # Print scaling table
        print("\n" + "="*60)
        print("SCALING RESULTS:")
        print("="*60)
        print(f"{'Model':8s} {'Params':>12s} {'Val Loss':>10s} {'Time(min)':>10s}")
        print("-"*45)
        for r in all_results:
            print(f"{r['model_name']:8s} {r['n_params']:>12,} "
                  f"{r['best_val_loss']:>10.4f} {r['wall_time_s']/60:>10.1f}")

        with open(RUNS_DIR / "scaling_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nNext step: python part2_plot.py")

if __name__ == "__main__":
    main()
