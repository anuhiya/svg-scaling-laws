"""
Part 1: Data Collection and Preprocessing Pipeline
SVG Scaling Laws Project — Version 3 (adds svg-stack-simple for 100M tokens)

Run from ~/Desktop/svg_scaling/:
    python part1_data_pipeline3.py
"""

import os
import re
import json
import random
import hashlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from lxml import etree
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers, decoders

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED         = 42
MAX_TOKENS          = 1024
MIN_CHARS           = 50
VOCAB_SIZE          = 4096
TARGET_TRAIN_TOKENS = 100_000_000
STACK_SUBSAMPLE     = 200_000    # how many to take from svg-stack-simple
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.98, 0.01, 0.01

OUT_DIR     = Path("data")
RENDERS_DIR = OUT_DIR / "renders"
STATS_DIR   = OUT_DIR / "stats"
for d in [OUT_DIR, RENDERS_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Step 1: Download datasets ─────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Downloading datasets")
print("="*60)

def load_svgs(ds_name, split="train", subsample=None):
    print(f"\nLoading {ds_name} ...")
    ds = load_dataset(ds_name, split=split)
    svg_field = next((f for f in ["Svg","svg","text","svg_code"] if f in ds.column_names), None)
    if svg_field is None:
        print(f"  WARNING: No SVG field found. Columns: {ds.column_names}")
        return []
    if subsample and len(ds) > subsample:
        indices = random.sample(range(len(ds)), subsample)
        ds = ds.select(indices)
        print(f"  Subsampled to {subsample:,} rows")
    svgs = [row[svg_field] for row in ds if row[svg_field]]
    print(f"  Loaded {len(svgs):,} SVGs  (field='{svg_field}')")
    return svgs

raw_svgs = []
raw_svgs += load_svgs("starvector/svg-icons-simple")
raw_svgs += load_svgs("starvector/svg-emoji-simple")
raw_svgs += load_svgs("starvector/svg-stack-simple", subsample=STACK_SUBSAMPLE)

print(f"\nTotal raw SVGs loaded: {len(raw_svgs):,}")

# ── Step 2: Clean and normalize ───────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Cleaning and normalizing SVGs")
print("="*60)

COMMENT_RE    = re.compile(r'<!--.*?-->', re.DOTALL)
WHITESPACE_RE = re.compile(r'\s+')
FLOAT_RE      = re.compile(r'\b(\d+\.\d{2,})\b')

def round_floats(m):
    return f"{float(m.group(1)):.1f}"

def normalize_svg(svg):
    svg = COMMENT_RE.sub('', svg)
    svg = WHITESPACE_RE.sub(' ', svg).strip()
    svg = FLOAT_RE.sub(round_floats, svg)
    if len(svg) < MIN_CHARS:
        return None
    try:
        etree.fromstring(svg.encode())
    except etree.XMLSyntaxError:
        return None
    return svg

cleaned = []
n_short = n_bad_xml = 0
for svg in tqdm(raw_svgs, desc="Cleaning"):
    result = normalize_svg(svg)
    if result is None:
        if len(svg) < MIN_CHARS:
            n_short += 1
        else:
            n_bad_xml += 1
    else:
        cleaned.append(result)

# Deduplicate
seen, deduped = set(), []
for svg in cleaned:
    h = hashlib.md5(svg.encode()).hexdigest()
    if h not in seen:
        seen.add(h)
        deduped.append(svg)

print(f"  Original:    {len(raw_svgs):,}")
print(f"  Too short:   {n_short:,}")
print(f"  Invalid XML: {n_bad_xml:,}")
print(f"  After dedup: {len(deduped):,}")
cleaned_svgs = deduped

# ── Step 3: Train BPE tokenizer ───────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Training BPE tokenizer")
print("="*60)

corpus_path = OUT_DIR / "corpus_for_tokenizer.txt"
with open(corpus_path, "w") as f:
    for svg in tqdm(cleaned_svgs, desc="Writing corpus"):
        f.write(svg + "\n")

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    show_progress=True,
)
print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE}) ...")
tokenizer.train(files=[str(corpus_path)], trainer=trainer)
tokenizer_path = OUT_DIR / "svg_tokenizer.json"
tokenizer.save(str(tokenizer_path))
print(f"Tokenizer saved to {tokenizer_path}")

# ── Step 4: Tokenize + filter by length ───────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Tokenizing + filtering by max length")
print("="*60)

tokenized = []
n_too_long = 0
for svg in tqdm(cleaned_svgs, desc="Tokenizing"):
    ids = tokenizer.encode(svg).ids
    if len(ids) > MAX_TOKENS:
        n_too_long += 1
    else:
        tokenized.append((svg, ids))

token_lengths = [len(ids) for _, ids in tokenized]
total_tokens  = sum(token_lengths)
print(f"  Too long (>{MAX_TOKENS}): {n_too_long:,}")
print(f"  Final dataset:           {len(tokenized):,} SVGs")
print(f"  Total tokens:            {total_tokens:,}")

if total_tokens < TARGET_TRAIN_TOKENS:
    print(f"  [WARNING]  Still below 100M. Current: {total_tokens/1e6:.1f}M")
    print(f"     Increase STACK_SUBSAMPLE or add svg-fonts-simple.")
else:
    print(f"  [DONE] Token target met! ({total_tokens/1e6:.1f}M tokens)")

# ── Step 5: Train/Val/Test splits ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Creating train/val/test splits")
print("="*60)

indices = list(range(len(tokenized)))
random.shuffle(indices)
n = len(indices)
n_val   = max(1, int(n * VAL_FRAC))
n_test  = max(1, int(n * TEST_FRAC))
n_train = n - n_val - n_test

splits = {
    "train": indices[:n_train],
    "val":   indices[n_train:n_train+n_val],
    "test":  indices[n_train+n_val:],
}

eos_id = tokenizer.token_to_id("<eos>")
split_tokens = {}
for name, idxs in splits.items():
    split_dir = OUT_DIR / name
    split_dir.mkdir(exist_ok=True)
    svg_list   = [tokenized[i][0] for i in idxs]
    token_list = [tokenized[i][1] for i in idxs]
    with open(split_dir / "svgs.txt", "w") as f:
        for svg in svg_list:
            f.write(svg + "\n<SEP>\n")
    flat = []
    for ids in token_list:
        flat.extend(ids)
        flat.append(eos_id)
    np.save(split_dir / "tokens.npy", np.array(flat, dtype=np.uint16))
    n_toks = sum(len(ids) for ids in token_list)
    split_tokens[name] = n_toks
    print(f"  {name:6s}: {len(idxs):7,} SVGs | {n_toks:13,} tokens → {split_dir}")

# ── Step 6: Statistics ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: Computing statistics")
print("="*60)

stats = {
    "vocab_size": VOCAB_SIZE,
    "max_token_length": MAX_TOKENS,
    "n_raw": len(raw_svgs),
    "n_too_short": n_short,
    "n_invalid_xml": n_bad_xml,
    "n_too_long": n_too_long,
    "n_final": len(tokenized),
    "total_tokens": int(total_tokens),
    "train_tokens": int(split_tokens["train"]),
    "val_tokens":   int(split_tokens["val"]),
    "test_tokens":  int(split_tokens["test"]),
    "train_svgs":   len(splits["train"]),
    "val_svgs":     len(splits["val"]),
    "test_svgs":    len(splits["test"]),
    "token_length_mean":   float(np.mean(token_lengths)),
    "token_length_median": float(np.median(token_lengths)),
    "token_length_std":    float(np.std(token_lengths)),
    "token_length_p95":    float(np.percentile(token_lengths, 95)),
}
with open(STATS_DIR / "dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))

# Histogram
plt.figure(figsize=(10, 5))
plt.hist(token_lengths, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
plt.axvline(MAX_TOKENS, color="red", linestyle="--", label=f"Max cutoff ({MAX_TOKENS})")
plt.xlabel("Token length per SVG")
plt.ylabel("Count")
plt.title("SVG Token Length Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(STATS_DIR / "token_length_histogram.png", dpi=150)
plt.close()
print(f"\nHistogram saved.")

# ── Step 7: Render examples ───────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7: Rendering example SVGs")
print("="*60)

try:
    import cairosvg

    sorted_by_len = sorted(tokenized, key=lambda x: len(x[1]))
    n = len(sorted_by_len)
    examples = {
        "short":  sorted_by_len[int(n * 0.05)],
        "medium": sorted_by_len[int(n * 0.50)],
        "long":   sorted_by_len[int(n * 0.95)],
    }
    for label, (svg, ids) in examples.items():
        out_png = RENDERS_DIR / f"example_{label}.png"
        try:
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out_png),
                             output_width=256, output_height=256)
            print(f"  Rendered {label} ({len(ids)} tokens) → {out_png}")
        except Exception as e:
            print(f"  Could not render {label}: {e}")

    # 12-image grid
    sample_idxs = random.sample(range(len(tokenized)), min(12, len(tokenized)))
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    tmp_png = RENDERS_DIR / "_tmp.png"
    for ax, idx in zip(axes.flat, sample_idxs):
        svg, ids = tokenized[idx]
        try:
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(tmp_png),
                             output_width=128, output_height=128)
            ax.imshow(plt.imread(str(tmp_png)))
            ax.set_title(f"{len(ids)} tok", fontsize=8)
        except:
            ax.set_facecolor("#eee")
            ax.text(0.5, 0.5, "error", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        ax.axis("off")
    plt.suptitle("Random SVG Samples from Dataset", fontsize=14)
    plt.tight_layout()
    plt.savefig(RENDERS_DIR / "sample_grid.png", dpi=150)
    plt.close()
    print(f"  Sample grid saved → {RENDERS_DIR}/sample_grid.png")
    if tmp_png.exists():
        tmp_png.unlink()

except Exception as e:
    print(f"  Rendering skipped: {e}")

# ── Done ──────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[DONE]  PART 1 COMPLETE")
print("="*60)
print(f"""
Output files:
  Tokenizer : data/svg_tokenizer.json
  Train     : data/train/tokens.npy  ({split_tokens['train']/1e6:.1f}M tokens)
  Val       : data/val/tokens.npy    ({split_tokens['val']/1e6:.1f}M tokens)
  Test      : data/test/tokens.npy   ({split_tokens['test']/1e6:.1f}M tokens)
  Stats     : data/stats/dataset_stats.json
  Renders   : data/renders/

Next step → python part2_train.py
""")
