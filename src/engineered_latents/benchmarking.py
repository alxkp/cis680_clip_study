"""
Standard CLIP benchmarking functions.
Supports retrieval (COCO, Flickr30K), zero-shot classification, and Winoground.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import torch
from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm as _tqdm

from .context import get_clip, track
from .metrics import (
    ClassificationMetrics,
    RetrievalMetrics,
    WinogroundMetrics,
    compute_classification_metrics,
    compute_retrieval_metrics,
    compute_winoground_metrics,
)


def tqdm[T](iterable: Iterable[T], desc: str = "") -> Iterable[T]:
    """Typed progress bar wrapper."""
    return cast("Iterable[T]", _tqdm(iterable, desc=desc))


# -----------------------------------------------------------------------------
# Feature extraction (with progress bars - this is the slow part)
# -----------------------------------------------------------------------------


def extract_image_features(
    images: list[Any], batch_size: int = 32, desc: str = "Encoding images"
) -> Tensor:
    """
    Extract CLIP image features from a list of PIL images.

    images: list of PIL images
    batch_size: batch size for processing
    desc: progress bar description

    Returns: [n_images, dim] tensor of normalized image features
    """
    ctx = get_clip()
    model = ctx.model.to(ctx.device)
    model.eval()

    all_features: list[Tensor] = []
    n_batches = (len(images) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc=desc):
        start = i * batch_size
        end = min(start + batch_size, len(images))
        batch_images = images[start:end]

        inputs = ctx.processor(images=batch_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(ctx.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def extract_text_features(
    texts: list[str], batch_size: int = 32, desc: str = "Encoding text"
) -> Tensor:
    """
    Extract CLIP text features from a list of strings.

    texts: list of text strings
    batch_size: batch size for processing
    desc: progress bar description

    Returns: [n_texts, dim] tensor of normalized text features
    """
    ctx = get_clip()
    model = ctx.model.to(ctx.device)
    model.eval()

    all_features: list[Tensor] = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc=desc):
        start = i * batch_size
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        inputs = ctx.processor(text=batch_texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(ctx.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


# -----------------------------------------------------------------------------
# Dataset loading helpers
# -----------------------------------------------------------------------------


def _load_dataset_as_list(
    name: str,
    split: str,
    max_samples: int | None = None,
    trust_remote_code: bool = False,
) -> list[dict[str, Any]]:
    """Load a HuggingFace dataset and return as a typed list of dicts."""
    ds = load_dataset(name, split=split, trust_remote_code=trust_remote_code)
    if hasattr(ds, "select") and max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))  # type: ignore[arg-type]
    return list(ds)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Retrieval benchmarks
# -----------------------------------------------------------------------------


def run_retrieval_benchmark(
    images: list[Any],
    texts: list[str],
    batch_size: int = 32,
) -> tuple[RetrievalMetrics, RetrievalMetrics]:
    """
    Run image-text retrieval benchmark.

    images: list of PIL images
    texts: list of captions (same length as images, 1-to-1 matching)
    batch_size: batch size for feature extraction

    Returns: (image_to_text_metrics, text_to_image_metrics)
    """
    image_features = extract_image_features(images, batch_size)
    text_features = extract_text_features(texts, batch_size)

    return compute_retrieval_metrics(image_features, text_features)


def run_coco_retrieval(
    split: str = "test",
    batch_size: int = 32,
    max_samples: int | None = None,
) -> tuple[RetrievalMetrics, RetrievalMetrics]:
    """
    Run retrieval benchmark on COCO captions.

    split: dataset split to use
    batch_size: batch size for feature extraction
    max_samples: limit number of samples (for quick testing)

    Returns: (image_to_text_metrics, text_to_image_metrics)
    """
    print("Loading COCO dataset...")
    samples = _load_dataset_as_list(
        "HuggingFaceM4/COCO", split, max_samples, trust_remote_code=True
    )

    images: list[Any] = [s["image"].convert("RGB") for s in samples]
    texts: list[str] = [s["sentences"]["raw"][0] for s in samples]

    return run_retrieval_benchmark(images, texts, batch_size)


def run_flickr30k_retrieval(
    split: str = "test",
    batch_size: int = 32,
    max_samples: int | None = None,
) -> tuple[RetrievalMetrics, RetrievalMetrics]:
    """
    Run retrieval benchmark on Flickr30K.

    split: dataset split to use
    batch_size: batch size for feature extraction
    max_samples: limit number of samples (for quick testing)

    Returns: (image_to_text_metrics, text_to_image_metrics)
    """
    print("Loading Flickr30K dataset...")
    samples = _load_dataset_as_list("nlphuji/flickr30k", split, max_samples)

    images: list[Any] = [s["image"].convert("RGB") for s in samples]
    texts: list[str] = [s["caption"][0] for s in samples]

    return run_retrieval_benchmark(images, texts, batch_size)


# -----------------------------------------------------------------------------
# Zero-shot classification
# -----------------------------------------------------------------------------


IMAGENET_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of the large {}.",
    "a photo of the small {}.",
    "a photo of a {} in the wild.",
]


def build_classifier_weights(
    class_names: list[str],
    templates: list[str] | None = None,
    batch_size: int = 32,
) -> Tensor:
    """
    Build zero-shot classifier weights by averaging text embeddings over templates.

    class_names: list of class names
    templates: list of prompt templates with {} placeholder
    batch_size: batch size for text encoding

    Returns: [n_classes, dim] tensor of classifier weights
    """
    if templates is None:
        templates = IMAGENET_TEMPLATES

    ctx = get_clip()
    model = ctx.model.to(ctx.device)
    model.eval()

    all_weights: list[Tensor] = []

    for i in tqdm(range(len(class_names)), desc="Building classifier"):
        class_name = class_names[i]
        prompts = [t.format(class_name) for t in templates]

        inputs = ctx.processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(ctx.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            class_weight = features.mean(dim=0)
            class_weight = class_weight / (class_weight.norm() + 1e-8)
            all_weights.append(class_weight.cpu())

    return torch.stack(all_weights, dim=0)


def run_zero_shot_classification(
    images: list[Any],
    labels: Tensor,
    classifier_weights: Tensor,
    batch_size: int = 32,
) -> ClassificationMetrics:
    """
    Run zero-shot classification.

    images: list of PIL images
    labels: [n_samples] tensor of ground truth class indices
    classifier_weights: [n_classes, dim] tensor of classifier weights
    batch_size: batch size for image encoding

    Returns: ClassificationMetrics
    """
    image_features = extract_image_features(images, batch_size)
    logits = image_features @ classifier_weights.T

    return compute_classification_metrics(logits, labels)


def run_imagenet_zero_shot(
    split: str = "validation",
    batch_size: int = 32,
    max_samples: int | None = None,
    templates: list[str] | None = None,
) -> ClassificationMetrics:
    """
    Run zero-shot classification on ImageNet.

    split: dataset split to use
    batch_size: batch size for processing
    max_samples: limit number of samples
    templates: prompt templates to use

    Returns: ClassificationMetrics
    """
    print("Loading ImageNet dataset...")
    samples = _load_dataset_as_list(
        "imagenet-1k", split, max_samples, trust_remote_code=True
    )

    # get class names from first sample's features
    ds = load_dataset("imagenet-1k", split=split, trust_remote_code=True)
    class_names: list[str] = ds.features["label"].names  # type: ignore[union-attr]

    classifier_weights = build_classifier_weights(class_names, templates, batch_size)

    images: list[Any] = [s["image"].convert("RGB") for s in samples]
    labels = torch.tensor([s["label"] for s in samples])

    return run_zero_shot_classification(images, labels, classifier_weights, batch_size)


# -----------------------------------------------------------------------------
# Winoground benchmark
# -----------------------------------------------------------------------------


def run_winoground_benchmark(batch_size: int = 32) -> WinogroundMetrics:
    """
    Run Winoground benchmark for fine-grained vision-language understanding.

    batch_size: batch size for feature extraction

    Returns: WinogroundMetrics
    """
    print("Loading Winoground dataset...")
    samples = _load_dataset_as_list(
        "facebook/winoground", "test", trust_remote_code=True
    )

    ctx = get_clip()
    model = ctx.model.to(ctx.device)
    model.eval()

    scores_c0_i0: list[float] = []
    scores_c0_i1: list[float] = []
    scores_c1_i0: list[float] = []
    scores_c1_i1: list[float] = []

    winoground_pbar: Iterable[int] = tqdm(range(len(samples)), desc="Evaluating Winoground")
    for i in winoground_pbar:
        sample = samples[i]
        image_0 = sample["image_0"].convert("RGB")
        image_1 = sample["image_1"].convert("RGB")
        caption_0 = sample["caption_0"]
        caption_1 = sample["caption_1"]

        images = [image_0, image_1]
        texts = [caption_0, caption_1]

        img_inputs = ctx.processor(images=images, return_tensors="pt", padding=True)
        txt_inputs = ctx.processor(text=texts, return_tensors="pt", padding=True)

        img_inputs = {k: v.to(ctx.device) for k, v in img_inputs.items()}
        txt_inputs = {k: v.to(ctx.device) for k, v in txt_inputs.items()}

        with torch.no_grad():
            img_feats = model.get_image_features(**img_inputs)
            txt_feats = model.get_text_features(**txt_inputs)

            img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-8)
            txt_feats = txt_feats / (txt_feats.norm(dim=-1, keepdim=True) + 1e-8)

            sim = img_feats @ txt_feats.T

            scores_c0_i0.append(sim[0, 0].item())
            scores_c0_i1.append(sim[1, 0].item())
            scores_c1_i0.append(sim[0, 1].item())
            scores_c1_i1.append(sim[1, 1].item())

    return compute_winoground_metrics(
        torch.tensor(scores_c0_i0),
        torch.tensor(scores_c0_i1),
        torch.tensor(scores_c1_i0),
        torch.tensor(scores_c1_i1),
    )


# -----------------------------------------------------------------------------
# Benchmark runner
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResults:
    """Aggregated results from all benchmarks."""

    winoground: WinogroundMetrics | None = None
    coco_i2t: RetrievalMetrics | None = None
    coco_t2i: RetrievalMetrics | None = None
    flickr_i2t: RetrievalMetrics | None = None
    flickr_t2i: RetrievalMetrics | None = None
    imagenet: ClassificationMetrics | None = None


def run_all_benchmarks(
    include_winoground: bool = True,
    include_coco: bool = True,
    include_flickr: bool = True,
    include_imagenet: bool = False,
    batch_size: int = 32,
    max_samples: int | None = None,
    track_results: bool = False,
) -> BenchmarkResults:
    """
    Run all standard CLIP benchmarks.

    include_winoground: run Winoground benchmark
    include_coco: run COCO retrieval
    include_flickr: run Flickr30K retrieval
    include_imagenet: run ImageNet zero-shot (requires HF auth)
    batch_size: batch size for all benchmarks
    max_samples: limit samples per benchmark (for quick testing)
    track_results: log results to Aim

    Returns: BenchmarkResults
    """
    winoground: WinogroundMetrics | None = None
    coco_i2t: RetrievalMetrics | None = None
    coco_t2i: RetrievalMetrics | None = None
    flickr_i2t: RetrievalMetrics | None = None
    flickr_t2i: RetrievalMetrics | None = None
    imagenet: ClassificationMetrics | None = None

    if include_winoground:
        print("\n=== Winoground Benchmark ===")
        winoground = run_winoground_benchmark(batch_size)
        print(f"Text Score:  {winoground.text_score:.4f}")
        print(f"Image Score: {winoground.image_score:.4f}")
        print(f"Group Score: {winoground.group_score:.4f}")

        if track_results:
            track(winoground.text_score, "benchmark/winoground_text")
            track(winoground.image_score, "benchmark/winoground_image")
            track(winoground.group_score, "benchmark/winoground_group")

    if include_coco:
        print("\n=== COCO Retrieval Benchmark ===")
        coco_i2t, coco_t2i = run_coco_retrieval("test", batch_size, max_samples)
        print(f"I2T R@1: {coco_i2t.recall_at_1:.4f}, R@5: {coco_i2t.recall_at_5:.4f}")
        print(f"T2I R@1: {coco_t2i.recall_at_1:.4f}, R@5: {coco_t2i.recall_at_5:.4f}")

        if track_results:
            track(coco_i2t.recall_at_1, "benchmark/coco_i2t_r1")
            track(coco_t2i.recall_at_1, "benchmark/coco_t2i_r1")

    if include_flickr:
        print("\n=== Flickr30K Retrieval Benchmark ===")
        flickr_i2t, flickr_t2i = run_flickr30k_retrieval(
            "test", batch_size, max_samples
        )
        print(
            f"I2T R@1: {flickr_i2t.recall_at_1:.4f}, R@5: {flickr_i2t.recall_at_5:.4f}"
        )
        print(
            f"T2I R@1: {flickr_t2i.recall_at_1:.4f}, R@5: {flickr_t2i.recall_at_5:.4f}"
        )

        if track_results:
            track(flickr_i2t.recall_at_1, "benchmark/flickr_i2t_r1")
            track(flickr_t2i.recall_at_1, "benchmark/flickr_t2i_r1")

    if include_imagenet:
        print("\n=== ImageNet Zero-Shot Benchmark ===")
        imagenet = run_imagenet_zero_shot("validation", batch_size, max_samples)
        print(f"Top-1 Accuracy: {imagenet.top1_accuracy:.4f}")
        print(f"Top-5 Accuracy: {imagenet.top5_accuracy:.4f}")

        if track_results:
            track(imagenet.top1_accuracy, "benchmark/imagenet_top1")
            track(imagenet.top5_accuracy, "benchmark/imagenet_top5")

    return BenchmarkResults(
        winoground=winoground,
        coco_i2t=coco_i2t,
        coco_t2i=coco_t2i,
        flickr_i2t=flickr_i2t,
        flickr_t2i=flickr_t2i,
        imagenet=imagenet,
    )


def print_benchmark_summary(results: BenchmarkResults) -> None:
    """Print a formatted summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if results.winoground:
        w = results.winoground
        print("\nWinoground:")
        print(
            f"  Text:  {w.text_score:.1%}  Image: {w.image_score:.1%}  Group: {w.group_score:.1%}"
        )

    if results.coco_i2t and results.coco_t2i:
        print("\nCOCO Retrieval:")
        print(
            f"  I2T: R@1={results.coco_i2t.recall_at_1:.1%} R@5={results.coco_i2t.recall_at_5:.1%}"
        )
        print(
            f"  T2I: R@1={results.coco_t2i.recall_at_1:.1%} R@5={results.coco_t2i.recall_at_5:.1%}"
        )

    if results.flickr_i2t and results.flickr_t2i:
        print("\nFlickr30K Retrieval:")
        print(
            f"  I2T: R@1={results.flickr_i2t.recall_at_1:.1%} R@5={results.flickr_i2t.recall_at_5:.1%}"
        )
        print(
            f"  T2I: R@1={results.flickr_t2i.recall_at_1:.1%} R@5={results.flickr_t2i.recall_at_5:.1%}"
        )

    if results.imagenet:
        print("\nImageNet Zero-Shot:")
        print(
            f"  Top-1: {results.imagenet.top1_accuracy:.1%}  Top-5: {results.imagenet.top5_accuracy:.1%}"
        )

    print("=" * 60)


if __name__ == "__main__":
    from engineered_latents.config import ClipConfig
    from engineered_latents.context import clip_context

    cfg = ClipConfig()

    with clip_context(cfg):
        results = run_all_benchmarks(
            include_winoground=True,
            include_coco=False,
            include_flickr=False,
            include_imagenet=False,
            max_samples=100,
        )
        print_benchmark_summary(results)
