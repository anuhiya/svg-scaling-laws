"""
Part 2: Plot scaling laws from training results.
Run after part2_train.py --mode train_all completes.
    python part2_plot.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

RUNS_DIR  = Path("runs")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────
with open(RUNS_DIR / "scaling_results.json") as f:
    results = json.load(f)

params     = np.array([r["n_params"]      for r in results])
val_losses = np.array([r["best_val_loss"] for r in results])
names      = [r["model_name"] for r in results]

# ── Fit power law: L = a * N^(-alpha) + c ────────────────────────────────────
def power_law(N, a, alpha, c):
    return a * N**(-alpha) + c

try:
    popt, pcov = curve_fit(
        power_law, params, val_losses,
        p0=[10.0, 0.07, 1.0],
        bounds=([0, 0, 0], [1e6, 2.0, 10.0]),
        maxfev=10000
    )
    a, alpha, c = popt
    perr = np.sqrt(np.diag(pcov))
    fit_ok = True
    print(f"Power law fit: L = {a:.3f} * N^(-{alpha:.3f}) + {c:.3f}")
    print(f"Scaling exponent α = {alpha:.3f} ± {perr[1]:.3f}")
except Exception as e:
    print(f"Fit failed: {e}")
    fit_ok = False

# ── Scaling plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(params, val_losses, s=80, zorder=5, color="steelblue", label="Trained models")
for i, name in enumerate(names):
    ax.annotate(name, (params[i], val_losses[i]),
                textcoords="offset points", xytext=(8, 0), fontsize=9)

if fit_ok:
    N_fit = np.logspace(np.log10(params.min()*0.8), np.log10(params.max()*1.2), 200)
    L_fit = power_law(N_fit, *popt)
    ax.plot(N_fit, L_fit, "r--", label=f"Power law fit: α={alpha:.3f}", linewidth=2)

ax.set_xscale("log")
ax.set_xlabel("Number of Parameters (log scale)", fontsize=12)
ax.set_ylabel("Validation Loss (after 1 epoch)", fontsize=12)
ax.set_title("SVG Transformer Scaling Law", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "scaling_law.png", dpi=150)
plt.close()
print(f"Scaling plot saved → {PLOTS_DIR}/scaling_law.png")

# ── Training curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for i, r in enumerate(results):
    hist  = r["history"]
    iters = hist["iter"]
    color = colors[i % len(colors)]
    label = f"{r['model_name']} ({r['n_params']/1e6:.1f}M)"
    axes[0].plot(iters, hist["train_loss"], color=color, label=label, alpha=0.8)
    axes[1].plot(iters, hist["val_loss"],   color=color, label=label, alpha=0.8)

for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Training Curves — All Model Sizes", fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "training_curves.png", dpi=150)
plt.close()
print(f"Training curves saved → {PLOTS_DIR}/training_curves.png")

# ── LR sweep plot ─────────────────────────────────────────────────────────────
lr_sweep_path = RUNS_DIR / "lr_sweep_results.json"
if lr_sweep_path.exists():
    with open(lr_sweep_path) as f:
        sweep = json.load(f)
    lrs    = [r["lr"]       for r in sweep["sweep_results"]]
    losses = [r["val_loss"] for r in sweep["sweep_results"]]
    best   = sweep["best_lr"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lrs, losses, "o-", color="steelblue", linewidth=2, markersize=8)
    best_loss = losses[lrs.index(best)]
    ax.scatter([best], [best_loss], s=150, color="red", zorder=5,
               label=f"Best LR={best:.0e}")
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
    ax.set_ylabel("Best Validation Loss", fontsize=12)
    ax.set_title("LR Sweep — Tiny Model", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lr_sweep.png", dpi=150)
    plt.close()
    print(f"LR sweep plot saved → {PLOTS_DIR}/lr_sweep.png")

# ── Stats table ───────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Model':8s} {'Params':>10s} {'Val Loss':>10s} {'Time(min)':>10s} {'Tok/s':>10s}")
print("-"*55)
for r in results:
    print(f"{r['model_name']:8s} {r['n_params']:>10,} "
          f"{r['best_val_loss']:>10.4f} "
          f"{r['wall_time_s']/60:>10.1f} "
          f"{r['tokens_per_sec']:>10.0f}")
if fit_ok:
    print(f"\nScaling exponent α = {alpha:.3f} ± {perr[1]:.3f}")
    print(f"(Natural language typically: α ≈ 0.07-0.10)")
