# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for CLIP Normalized Cut Clustering - exploring compositional token construction via spectral clustering on CLIP latent representations. Uses Nyström normalized cuts (ncut-pytorch) to cluster image-text affinity matrices and various alignment losses to encourage sparse, generalizable structures.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run linting
uv run ruff check src/

# Run a module directly
uv run python -m engineered_latents.data
uv run python -m engineered_latents.benchmarking
uv run python -m engineered_latents.losses.cluster_loss
```

## Architecture

### Context Management (`context.py`)
Uses Python `ContextVar` for global state management:
- `clip_context(cfg)` - context manager that loads CLIP model/processor and manages device placement
- `aim_context(repo, experiment)` - context manager for Aim experiment tracking
- `get_clip()` / `get_run()` - getters for current context
- `track(value, name, step)` - convenience wrapper for Aim logging

### Configuration (`config.py`)
Uses draccus `ChoiceRegistry` pattern for polymorphic CLI config:
- `ModelConfig` → `ClipConfig` (openai/clip-vit-base-patch32)
- `DatasetConfig` → `WinogroundConfig`, `WhatsupAllConfig`, `COCOCaptionsConfig`, `Flickr30KConfig`
- Each dataset config defines its own `preprocess()` and `collate_fn()` methods
- `MainConfig` aggregates model, dataset, and training configs

### Data Loading (`data.py`)
- `create_loaders(cfg)` - returns streaming train/val DataLoaders using HuggingFace datasets
- Uses `streaming=True` for lazy loading, preprocessing applied via `.map()`

### Loss Functions (`losses/cluster_loss.py`)
Three alignment/clustering losses:
- `cka_alignment_loss(X, Y)` - Centered Kernel Alignment between representations
- `cross_covariance_svd_loss(X, Y, k)` - encourages k strong alignment directions via spectral gap
- `ncut_cluster_loss(eigenvalues, k)` - encourages k-cluster structure in normalized Laplacian eigenvalues
- Each has corresponding `plot_*` functions for loss surface visualization

### Benchmarking (`benchmarking.py`)
Standard CLIP evaluation suite:
- `run_winoground_benchmark()` - fine-grained vision-language understanding
- `run_coco_retrieval()` / `run_flickr30k_retrieval()` - image-text retrieval
- `run_imagenet_zero_shot()` - zero-shot classification
- `run_all_benchmarks()` - aggregates all benchmarks with optional Aim tracking

### Metrics (`metrics.py`)
Pure functions for computing evaluation metrics:
- `RetrievalMetrics` - recall@k, median/mean rank
- `ClassificationMetrics` - top1/top5 accuracy
- `WinogroundMetrics` - text/image/group scores

### Normalized Cuts (`run_cuts.py`)
- `compute_cuts(affinities)` - runs Nyström NCut on affinity matrix, returns eigenvectors
- `compute_word_scores(eigvec_3d)` - aggregates eigenvector scores per word

## Key Patterns

**Running benchmarks with context:**
```python
from engineered_latents.config import ClipConfig
from engineered_latents.context import clip_context

with clip_context(ClipConfig()):
    results = run_all_benchmarks(include_winoground=True)
```

**Tracking experiments:**
```python
from engineered_latents.context import aim_context, track

with aim_context(repo=".", experiment="my_exp"):
    track(loss.item(), "train_loss", step=i)
```

## Output

Generated files go to `out/` directory (see `utils.OUTPUT_DIR`).
