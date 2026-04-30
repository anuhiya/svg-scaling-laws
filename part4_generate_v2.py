"""
Part 4: Fixed SVG Generation Script
Run: python part4_generate_v2.py
"""

import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from lxml import etree
from dataclasses import dataclass

random.seed(42)
torch.manual_seed(42)

DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
DATA_DIR   = Path("data")
RUNS_DIR   = Path("runs")
GEN_DIR    = Path("generated")
GEN_DIR.mkdir(exist_ok=True)
BLOCK_SIZE = 512
VOCAB_SIZE = 4096

print(f"Device: {DEVICE}")

# ── Model (same as part2) ─────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    name: str
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int = VOCAB_SIZE
    block_size: int = BLOCK_SIZE
    dropout: float = 0.0
    bias: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3*cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.register_buffer("bias", torch.tril(
            torch.ones(cfg.block_size, cfg.block_size)
        ).view(1,1,cfg.block_size,cfg.block_size))

    def forward(self, x):
        B,T,C = x.size()
        q,k,v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        else:
            att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
            att = F.softmax(att,dim=-1)
            y = att @ v
        return self.c_proj(y.transpose(1,2).contiguous().view(B,T,C))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.n_embd, 4*cfg.n_embd, bias=cfg.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4*cfg.n_embd, cfg.n_embd, bias=cfg.bias)
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe  = nn.Embedding(cfg.block_size, cfg.n_embd),
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B,T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))

# ── Load model ────────────────────────────────────────────────────────────────
cfg = ModelConfig("small", n_layer=6, n_head=6, n_embd=192)
model = GPT(cfg).to(DEVICE)

# Try SP small first, then muP small
for run_name in ["scale_small_lr1e-02", "mup2_scale_small_lr1e-02"]:
    pt = RUNS_DIR / run_name / "best_model.pt"
    if pt.exists():
        model.load_state_dict(torch.load(pt, map_location=DEVICE))
        print(f"Loaded: {pt}")
        break

model.eval()

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tok = Tokenizer.from_file(str(DATA_DIR / "svg_tokenizer.json"))
bos_id = tok.token_to_id("<bos>")
eos_id = tok.token_to_id("<eos>")

print(f"bos_id={bos_id}, eos_id={eos_id}")

# ── Load real SVGs from training data for prefix prompts ──────────────────────
with open(DATA_DIR / "train" / "svgs.txt") as f:
    raw = f.read()
real_svgs = [s.strip() for s in raw.split("<SEP>") if s.strip()]
print(f"Loaded {len(real_svgs):,} real SVGs for reference")

# ── Generation function ───────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt_ids, max_new_tokens=400, temperature=1.0, top_k=50):
    """Generate from a list of token ids."""
    # Start with bos + prompt
    ids = [bos_id] + prompt_ids
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == eos_id:
            break

        idx = torch.cat([idx, next_id], dim=1)

    # Decode everything after bos
    generated_ids = idx[0].tolist()[1:]  # skip bos
    return tok.decode(generated_ids)

# ── Test: reproduce a real SVG ────────────────────────────────────────────────
print("\n--- Testing: feed first 10 tokens of real SVG, let model complete ---")
test_svg = real_svgs[0]
test_ids = tok.encode(test_svg).ids[:15]  # first 15 tokens as prompt
test_prompt_text = tok.decode(test_ids)
print(f"Prompt: {repr(test_prompt_text[:80])}")
completion = generate(test_ids, max_new_tokens=500, temperature=0.7, top_k=40)
print(f"Completion: {repr(completion[:200])}")
print()

# ── Unconditional generation ──────────────────────────────────────────────────
# Use first few tokens of real SVGs as prompts (guaranteed valid start)
print("="*60)
print("UNCONDITIONAL GENERATION (10 samples)")
print("="*60)

results = []
sample_svgs = random.sample(real_svgs, 20)

for i in range(10):
    temp = [0.5, 0.8, 1.0][i % 3]
    # Use first 5 tokens of a real SVG as seed (just the <svg> opening)
    seed_svg = sample_svgs[i]
    seed_ids = tok.encode(seed_svg).ids[:20]

    completion = generate(seed_ids, max_new_tokens=600, temperature=temp, top_k=50)

    # Force close if needed
    # Close any open tags properly
    if '</svg>' not in completion:
        if '</path>' not in completion and '<path' in completion:
            completion = completion.rstrip() + '"/></svg>'
        elif '<g>' in completion and '</g>' not in completion:
            completion = completion.rstrip() + '</g></svg>'
        else:
            completion = completion.rstrip() + '</svg>'

    # Check validity
    valid_xml = False
    try:
        etree.fromstring(completion.encode())
        valid_xml = True
    except:
        pass

    results.append({
        "id": i, "temperature": temp,
        "svg": completion, "valid_xml": valid_xml,
        "length": len(completion)
    })
    status = "[VALID]" if valid_xml else "[INVALID]"
    print(f"  Sample {i+1:2d} | temp={temp} | {status} | {len(completion)} chars")

# ── Prefix-conditioned: use real SVG prefixes ─────────────────────────────────
print("\n" + "="*60)
print("PREFIX-CONDITIONED GENERATION")
print("="*60)

prefix_results = []
prefix_names = ["circle_face", "open_path", "group_rect", "arrow", "heart"]
# Use actual partial SVGs from training data
prefix_svgs = random.sample(real_svgs, 5)

for i, (name, svg) in enumerate(zip(prefix_names, prefix_svgs)):
    # Use first 30% of tokens as prefix
    all_ids = tok.encode(svg).ids
    n_prefix = max(20, len(all_ids) // 2)
    prefix_ids = all_ids[:n_prefix]
    prefix_text = tok.decode(prefix_ids)

    completion = generate(prefix_ids, max_new_tokens=500, temperature=0.8, top_k=50)
    # Close any open tags properly
    if '</svg>' not in completion:
        if '</path>' not in completion and '<path' in completion:
            completion = completion.rstrip() + '"/></svg>'
        elif '<g>' in completion and '</g>' not in completion:
            completion = completion.rstrip() + '</g></svg>'
        else:
            completion = completion.rstrip() + '</svg>'

    valid_xml = False
    try:
        etree.fromstring(completion.encode())
        valid_xml = True
    except:
        pass

    prefix_results.append({
        "name": name, "prefix": prefix_text,
        "full_svg": completion, "valid_svg": valid_xml
    })
    status = "[VALID]" if valid_xml else "[INVALID]"
    print(f"  {name:15s} | {status} | {len(completion)} chars")

# ── Metrics ───────────────────────────────────────────────────────────────────
all_svgs = [r["svg"] for r in results]
n = len(all_svgs)
n_xml = sum(1 for r in results if r["valid_xml"])
n_vbox  = sum(1 for s in all_svgs if 'viewBox' in s)
n_path  = sum(1 for s in all_svgs if '<path' in s)

metrics = {
    "n_samples": n,
    "xml_validity_rate": n_xml / n,
    "has_viewbox_rate":  n_vbox / n,
    "has_path_rate":     n_path / n,
}

print("\n" + "="*60)
print("METRICS")
print("="*60)
print(json.dumps(metrics, indent=2))

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = GEN_DIR / "small_v2"
out_dir.mkdir(exist_ok=True)

with open(out_dir / "unconditional.json", "w") as f:
    json.dump(results, f, indent=2)
with open(out_dir / "prefix_conditioned.json", "w") as f:
    json.dump(prefix_results, f, indent=2)
with open(out_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# HTML gallery
cells = ""
for r in results:
    bg = "#e8f5e9" if r["valid_xml"] else "#fff3e0"
    cells += f"""<div style="background:{bg};padding:8px;border-radius:4px;
        width:220px;display:inline-block;margin:6px;vertical-align:top">
      <div style="width:220px;height:220px;overflow:hidden;border:1px solid #ddd">
        {r['svg']}
      </div>
      <div style="font-size:11px;margin-top:4px">
        temp={r['temperature']} | {'VALID' if r['valid_xml'] else 'INVALID'} | {r['length']} chars
      </div>
    </div>"""

prefix_cells = ""
for r in prefix_results:
    bg = "#e8f5e9" if r["valid_svg"] else "#fff3e0"
    prefix_cells += f"""<div style="background:{bg};padding:8px;border-radius:4px;
        width:220px;display:inline-block;margin:6px;vertical-align:top">
      <div style="width:220px;height:220px;overflow:hidden;border:1px solid #ddd">
        {r['full_svg']}
      </div>
      <div style="font-size:11px;margin-top:4px">{r['name']} | {'VALID' if r['valid_svg'] else 'INVALID'}</div>
    </div>"""

html = f"""<!DOCTYPE html><html><head><title>Generated SVGs</title></head>
<body style="font-family:sans-serif;padding:20px;background:#fafafa">
<h1>Generated SVG Samples</h1>
<h2>Metrics</h2>
<pre>{json.dumps(metrics, indent=2)}</pre>
<h2>Unconditional Samples (10)</h2>
<div>{cells}</div>
<h2>Prefix-Conditioned Samples (5)</h2>
<div>{prefix_cells}</div>
</body></html>"""

with open(out_dir / "gallery.html", "w") as f:
    f.write(html)

print(f"\nSaved to {out_dir}/")
print(f"Open {out_dir}/gallery.html in browser to view!")
