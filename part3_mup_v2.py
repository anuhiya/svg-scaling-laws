import math
import time
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from mup import MuReadout, set_base_shapes, MuAdamW

DATA_DIR  = Path("data")
RUNS_DIR  = Path("runs")
PLOTS_DIR = Path("plots")
RUNS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"Using device: {DEVICE}")

BATCH_SIZE   = 32
BLOCK_SIZE   = 512
GRAD_CLIP    = 1.0
WARMUP_ITERS = 200
WEIGHT_DECAY = 0.1
BETA1, BETA2 = 0.9, 0.95
VOCAB_SIZE   = 4096

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

MODEL_CONFIGS = [
    ModelConfig("tiny",   n_layer=4,  n_head=4,  n_embd=128),
    ModelConfig("small",  n_layer=6,  n_head=6,  n_embd=192),
    ModelConfig("medium", n_layer=6,  n_head=6,  n_embd=384),
    ModelConfig("large",  n_layer=10, n_head=8,  n_embd=512),
    ModelConfig("xl",     n_layer=12, n_head=12, n_embd=768),
]

#  muP Transformer
class MuCausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn   = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj   = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.n_head   = cfg.n_head
        self.n_embd   = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.register_buffer("bias", torch.tril(
            torch.ones(cfg.block_size, cfg.block_size)
        ).view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # muP: scale by 1/d instead of 1/sqrt(d)
        att = (q @ k.transpose(-2, -1)) * (1.0 / self.head_dim)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y   = att @ v
        y   = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MuMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class MuBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.attn = MuCausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.mlp  = MuMLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MuGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe  = nn.Embedding(cfg.block_size, cfg.n_embd),
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([MuBlock(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = MuReadout(cfg.n_embd, cfg.vocab_size, bias=False)
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
        pos  = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

def make_mup_model(cfg: ModelConfig):
    """
    KEY FIX: base and delta must have same n_layer as target.
    Only width (n_embd) varies between base and delta.
    """
    base_cfg  = ModelConfig("base",  n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_head * 8)
    delta_cfg = ModelConfig("delta", n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_head * 16)
    base_model  = MuGPT(base_cfg)
    delta_model = MuGPT(delta_cfg)
    model = MuGPT(cfg)
    set_base_shapes(model, base_model, delta=delta_model)
    return model

# Data + LR 
class SVGDataset:
    def __init__(self, split, block_size):
        tokens = np.load(DATA_DIR / split / "tokens.npy", mmap_mode='r')
        self.data = torch.from_numpy(tokens.astype(np.int64))
        self.block_size = block_size
        print(f"  {split}: {len(self.data):,} tokens")

    def __len__(self):
        return len(self.data) - self.block_size

    def get_batch(self, batch_size, device):
        ix = torch.randint(len(self) - 1, (batch_size,))
        x  = torch.stack([self.data[i:i+self.block_size]     for i in ix])
        y  = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x.to(device), y.to(device)

def get_lr(it, lr, warmup_iters, lr_decay_iters):
    if it < warmup_iters:
        return lr * it / warmup_iters
    if it > lr_decay_iters:
        return lr * 0.1
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr * 0.1 + coeff * (lr - lr * 0.1)

# Training 
def train_mup(cfg, lr, max_iters, run_name, eval_interval=500, eval_iters=50):
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(exist_ok=True)

    print(f"\nTraining muP: {run_name}, lr={lr}, n_embd={cfg.n_embd}, n_layer={cfg.n_layer}")

    train_data = SVGDataset("train", BLOCK_SIZE)
    val_data   = SVGDataset("val",   BLOCK_SIZE)

    model = make_mup_model(cfg).to(DEVICE)
    n_params = model.count_params()
    print(f"  Params: {n_params:,}")

    optimizer = MuAdamW(
        model.parameters(), lr=lr,
        betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY,
    )

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = {}
        for split, dataset in [("train", train_data), ("val", val_data)]:
            ls = [model(*dataset.get_batch(BATCH_SIZE, DEVICE))[1].item()
                  for _ in range(eval_iters)]
            losses[split] = float(np.mean(ls))
        model.train()
        return losses

    history = {"iter": [], "train_loss": [], "val_loss": []}
    t0 = time.time()
    best_val_loss = float('inf')

    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss()
            elapsed = time.time() - t0
            print(f"  iter {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | {elapsed:.0f}s")
            history["iter"].append(it)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(model.state_dict(), run_dir / "best_model.pt")

        if it == max_iters:
            break

        current_lr = get_lr(it, lr, WARMUP_ITERS, max_iters)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        x, y = train_data.get_batch(BATCH_SIZE, DEVICE)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    wall_time = time.time() - t0
    results = {
        "run_name": run_name, "model_name": cfg.name,
        "n_params": n_params, "lr": lr,
        "best_val_loss": best_val_loss,
        "wall_time_s": wall_time,
        "history": history,
        "parameterization": "mup",
        "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_embd": cfg.n_embd,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n Done. Best val loss: {best_val_loss:.4f} | Time: {wall_time/60:.1f}min")
    return results

def iters_per_epoch():
    tokens = np.load(DATA_DIR / "train" / "tokens.npy", mmap_mode='r')
    return len(tokens) // (BATCH_SIZE * BLOCK_SIZE)

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train_all", "plot"], required=True)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    n_iters = iters_per_epoch()
    print(f"Iters per epoch: {n_iters:,}")

    if args.mode == "train_all":
        lr = args.lr
        print(f"\nTraining all 5 muP models with lr={lr}")
        all_results = []
        for cfg in MODEL_CONFIGS:
            # Skip if already done
            run_name = f"mup2_scale_{cfg.name}_lr{lr:.0e}"
            run_dir  = RUNS_DIR / run_name
            if (run_dir / "results.json").exists():
                print(f"\nSkipping {cfg.name} — already done")
                with open(run_dir / "results.json") as f:
                    all_results.append(json.load(f))
                continue
            r = train_mup(cfg, lr, n_iters, run_name)
            all_results.append(r)

        print("\nmuP scaling results:")
        for r in all_results:
            print(f"{r['model_name']:8s} {r['n_params']:>12,} "
                  f"{r['best_val_loss']:>10.4f} {r['wall_time_s']/60:>10.1f}")

        with open(RUNS_DIR / "mup_scaling_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\nNext: python part3_mup_v2.py --mode plot")

    elif args.mode == "plot":
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit

        def power_law(N, a, alpha, c):
            return a * N**(-alpha) + c

        fig, ax = plt.subplots(figsize=(9, 5))

        for label, fname, color, marker in [
            ("Standard (SP)", "scaling_results.json",     "steelblue",  "o"),
            ("muP",           "mup_scaling_results.json", "darkorange", "s"),
        ]:
            path = RUNS_DIR / fname
            if not path.exists():
                print(f"Missing {fname} — skipping")
                continue
            with open(path) as f:
                results = json.load(f)
            params = np.array([r["n_params"]      for r in results])
            losses = np.array([r["best_val_loss"] for r in results])
            names  = [r["model_name"] for r in results]

            ax.scatter(params, losses, s=80, color=color, marker=marker, zorder=5)
            for i, name in enumerate(names):
                ax.annotate(name, (params[i], losses[i]),
                            textcoords="offset points", xytext=(8,2), fontsize=8)
            try:
                popt, pcov = curve_fit(power_law, params, losses,
                                       p0=[10,0.07,1.0],
                                       bounds=([0,0,0],[1e6,2,10]), maxfev=10000)
                a, alpha, c = popt
                perr = np.sqrt(np.diag(pcov))
                N_fit = np.logspace(np.log10(params.min()*0.8),
                                    np.log10(params.max()*1.2), 200)
                ax.plot(N_fit, power_law(N_fit, *popt), "--", color=color,
                        linewidth=2, label=f"{label}: α={alpha:.3f}±{perr[1]:.3f}")
                print(f"{label}: α={alpha:.3f} ± {perr[1]:.3f}")
            except Exception as e:
                ax.plot(params, losses, "--", color=color, linewidth=1.5, label=label)
                print(f"Fit failed for {label}: {e}")

        ax.set_xscale("log")
        ax.set_xlabel("Number of Parameters (log scale)", fontsize=12)
        ax.set_ylabel("Validation Loss (after 1 epoch)", fontsize=12)
        ax.set_title("Scaling Laws: Standard Parameterization vs µP", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "sp_vs_mup_scaling.png", dpi=150)
        plt.close()
        print(f"Plot saved → {PLOTS_DIR}/sp_vs_mup_scaling.png")

if __name__ == "__main__":
    main()
