# gradlab

Personal playground for training loop tricks

## Getting started

```bash
pip install -r requirements.txt
```

## Examples

```bash
python train.py --epochs 5 --synthetic
# metrics land in runs/metrics.csv
```

## Features

- Synthetic dataset mode: no download needed to smoke-test
- Metrics logged to CSV for plotting
- Cosine LR schedule with warmup
- Gradient clipping and clean metrics logging
- Single file model definition, easy to hack

## Project structure

```text
├── docs/
│   ├── faq.md
│   ├── roadmap.md
│   └── usage.md
├── examples/
│   └── quickstart.md
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── model.py
├── requirements.txt
└── train.py
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
