"""
Save SVG examples as HTML file for viewing/screenshotting in browser.
No cairo, no system libraries needed.
Run: python render_examples_browser.py
"""
import random
from pathlib import Path
from tokenizers import Tokenizer

random.seed(42)
DATA_DIR    = Path("data")
RENDERS_DIR = DATA_DIR / "renders"
RENDERS_DIR.mkdir(exist_ok=True)

print("Loading SVGs...")
with open(DATA_DIR / "train" / "svgs.txt") as f:
    raw = f.read()
all_svgs = [s.strip() for s in raw.split("<SEP>") if s.strip()]
print(f"Loaded {len(all_svgs):,} SVGs")

tokenizer = Tokenizer.from_file(str(DATA_DIR / "svg_tokenizer.json"))

# Sample 500 for speed
sample = random.sample(all_svgs, min(500, len(all_svgs)))
print("Tokenizing sample...")
paired = [(svg, len(tokenizer.encode(svg).ids)) for svg in sample]
paired.sort(key=lambda x: x[1])
n = len(paired)

short  = paired[int(n * 0.05)]
medium = paired[int(n * 0.50)]
long_  = paired[int(n * 0.95)]
grid   = random.sample(paired, 12)

# Save individual SVG files
for label, (svg, toks) in [("short", short), ("medium", medium), ("long", long_)]:
    path = RENDERS_DIR / f"example_{label}.svg"
    path.write_text(svg)
    print(f"  Saved {label} ({toks} tokens) → {path}")

# Build an HTML gallery page
def make_cell(svg, toks, size=200):
    # Scale SVG to fixed size
    scaled = svg.replace('<svg ', f'<svg width="{size}" height="{size}" ', 1)
    return f"""
    <div style="display:inline-block;margin:8px;text-align:center;vertical-align:top">
      <div style="width:{size}px;height:{size}px;border:1px solid #ddd;overflow:hidden">
        {scaled}
      </div>
      <div style="font-size:11px;color:#666;margin-top:4px">{toks} tokens</div>
    </div>"""

# Grid of 12
grid_html = "".join(make_cell(svg, toks) for svg, toks in grid)

# Examples section
examples_html = ""
for label, (svg, toks) in [("Short", short), ("Medium", medium), ("Long", long_)]:
    examples_html += f"<h3>{label} ({toks} tokens)</h3>" + make_cell(svg, toks, 300)

html = f"""<!DOCTYPE html>
<html>
<head><title>SVG Dataset Examples</title></head>
<body style="font-family:sans-serif;padding:20px;background:#fafafa">
  <h1>SVG Dataset — Part 1 Examples</h1>

  <h2>Complexity Examples (Short / Medium / Long)</h2>
  {examples_html}

  <h2>Random Sample Grid (12 SVGs)</h2>
  <div>{grid_html}</div>

  <h2>Dataset Statistics</h2>
  <table border="1" cellpadding="8" style="border-collapse:collapse">
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Total SVGs (final)</td><td>157,760</td></tr>
    <tr><td>Train SVGs</td><td>154,606</td></tr>
    <tr><td>Val SVGs</td><td>1,577</td></tr>
    <tr><td>Test SVGs</td><td>1,577</td></tr>
    <tr><td>Train tokens</td><td>95.2M</td></tr>
    <tr><td>Vocab size</td><td>4,096</td></tr>
    <tr><td>Mean token length</td><td>616</td></tr>
    <tr><td>Median token length</td><td>619</td></tr>
    <tr><td>P95 token length</td><td>974</td></tr>
    <tr><td>Max token cutoff</td><td>1,024</td></tr>
  </table>
</body>
</html>"""

out_html = RENDERS_DIR / "svg_gallery.html"
out_html.write_text(html)
print(f"\n[DONE] Gallery saved → {out_html}")
print("Open this file in your browser (double-click it) to view and screenshot the SVGs.")
print("\n Part 1 fully complete! Next: python part2_train.py")
