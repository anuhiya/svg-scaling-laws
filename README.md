# Scaling Laws for Transformer Language Models Trained on SVG Code

This project trains decoder-only Transformer language models on SVG (Scalable Vector Graphics) code and studies how model performance scales with model size. It also compares standard parameterization (SP) with Maximal Update Parameterization (µP) for learning rate transfer across model sizes.

## Project Structure

```
svg_scaling/
├── part1_data_pipeline3.py      # Data download, cleaning, tokenization
├── part2_train.py               # Transformer training + LR sweep
├── part2_plot.py                # Scaling law plots
├── part3_mup_v2.py              # muP training + comparison plots
├── part4_generate_v2.py         # SVG sample generation + evaluation
├── render_examples_browser.py   # Renders dataset examples as HTML
├── data/                        # Processed data (not tracked)
├── runs/                        # Training results (not tracked)
├── plots/                       # Generated plots
└── generated/                   # Generated SVG samples
```

## Setup

```bash
conda create -n svg python=3.11
conda activate svg
pip install torch datasets tokenizers sentencepiece lxml numpy matplotlib scipy mup
```

## Usage

### Part 1: Data Pipeline
Downloads and preprocesses SVG datasets from HuggingFace.
```bash
python part1_data_pipeline3.py
```

### Part 2: Training
Run LR sweep on the tiny model first, then train all 5 model sizes.
```bash
python part2_train.py --mode lr_sweep
python part2_train.py --mode train_all --lr 0.01
python part2_plot.py
```

### Part 3: muP
Run muP LR sweep and train all 5 models with muP.
```bash
python part3_mup_v2.py --mode train_all --lr 0.01
python part3_mup_v2.py --mode plot
```

### Part 4: Generation
Generate SVG samples from the best model and evaluate.
```bash
python part4_generate_v2.py
```

## Datasets

Three datasets from the [StarVector project](https://huggingface.co/starvector):
- `starvector/svg-icons-simple` — 80,434 SVG icons
- `starvector/svg-emoji-simple` — 4,114 SVG emoji
- `starvector/svg-stack-simple` — 200,000 SVGs (subsampled)

## Results

| Model | Params | SP Val Loss | µP Val Loss |
|-------|--------|-------------|-------------|
| Tiny  | 1.4M   | 1.0487      | 1.0566      |
| Small | 3.5M   | 0.9047      | 1.0047      |
| Medium| 12.4M  | 2.5268      | 0.9931      |
| Large | 33.8M  | 2.9060      | 1.0257      |
| XL    | 88.5M  | 3.1388      | 1.0415      |

µP resolves training instability in larger models, improving validation loss by 60-67% for medium, large, and XL models.

## Hardware

All training was done on Apple M4 Pro using Metal Performance Shaders (MPS).
