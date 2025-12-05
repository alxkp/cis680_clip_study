"""
Metrics for evaluating CLIP performance on standard benchmarks.
Pure functions for computing retrieval and classification metrics.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RetrievalMetrics:
    """Results from retrieval evaluation."""

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    median_rank: float
    mean_rank: float


@dataclass(frozen=True)
class ClassificationMetrics:
    """Results from zero-shot classification."""

    top1_accuracy: float
    top5_accuracy: float


@dataclass(frozen=True)
class WinogroundMetrics:
    """Results from Winoground evaluation."""

    text_score: float  # correct text given image
    image_score: float  # correct image given text
    group_score: float  # both correct


def compute_similarity_matrix(
    image_features: Tensor, text_features: Tensor, normalize: bool = True
) -> Tensor:
    """
    Compute cosine similarity matrix between image and text features.

    image_features: [n_images, dim]
    text_features: [n_texts, dim]
    normalize: whether to L2-normalize features

    Returns: [n_images, n_texts] similarity matrix
    """
    if normalize:
        image_features = image_features / (
            image_features.norm(dim=-1, keepdim=True) + 1e-8
        )
        text_features = text_features / (
            text_features.norm(dim=-1, keepdim=True) + 1e-8
        )

    return image_features @ text_features.T


def recall_at_k(similarity: Tensor, k: int, direction: str = "i2t") -> float:
    """
    Compute recall@k for retrieval.

    similarity: [n_images, n_texts] similarity matrix
    k: number of top results to consider
    direction: 'i2t' (image-to-text) or 't2i' (text-to-image)

    Returns: recall@k as float in [0, 1]

    Assumes diagonal entries are ground truth matches (index i matches index i).
    """
    if direction == "t2i":
        similarity = similarity.T

    n = similarity.shape[0]
    # get indices of top-k predictions for each query
    _, topk_indices = similarity.topk(k, dim=1)

    # ground truth is diagonal (query i should retrieve item i)
    ground_truth = torch.arange(n, device=similarity.device).unsqueeze(1)

    # check if ground truth is in top-k
    correct = (topk_indices == ground_truth).any(dim=1).float()

    return correct.mean().item()


def compute_retrieval_metrics(
    image_features: Tensor, text_features: Tensor
) -> tuple[RetrievalMetrics, RetrievalMetrics]:
    """
    Compute full retrieval metrics for image-text matching.

    image_features: [n, dim] normalized image embeddings
    text_features: [n, dim] normalized text embeddings

    Returns: (image_to_text_metrics, text_to_image_metrics)

    Assumes 1-to-1 matching where index i in images matches index i in texts.
    """
    sim = compute_similarity_matrix(image_features, text_features)

    def metrics_for_direction(direction: str) -> RetrievalMetrics:
        sim_dir = sim if direction == "i2t" else sim.T
        n = sim_dir.shape[0]

        r1 = recall_at_k(sim, 1, direction)
        r5 = recall_at_k(sim, 5, direction)
        r10 = recall_at_k(sim, 10, direction)

        # compute ranks for median/mean
        # rank of ground truth for each query
        sorted_indices = sim_dir.argsort(dim=1, descending=True)
        ground_truth = torch.arange(n, device=sim.device)
        ranks = (sorted_indices == ground_truth.unsqueeze(1)).nonzero()[:, 1] + 1

        return RetrievalMetrics(
            recall_at_1=r1,
            recall_at_5=r5,
            recall_at_10=r10,
            median_rank=ranks.float().median().item(),
            mean_rank=ranks.float().mean().item(),
        )

    return metrics_for_direction("i2t"), metrics_for_direction("t2i")


def compute_topk_accuracy(logits: Tensor, labels: Tensor, k: int) -> float:
    """
    Compute top-k accuracy for classification.

    logits: [n_samples, n_classes] prediction scores
    labels: [n_samples] ground truth class indices
    k: number of top predictions to consider

    Returns: accuracy as float in [0, 1]
    """
    _, topk_preds = logits.topk(k, dim=1)
    correct = (topk_preds == labels.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


def compute_classification_metrics(
    logits: Tensor, labels: Tensor
) -> ClassificationMetrics:
    """
    Compute classification metrics.

    logits: [n_samples, n_classes] prediction scores
    labels: [n_samples] ground truth class indices

    Returns: ClassificationMetrics with top1 and top5 accuracy
    """
    top1 = compute_topk_accuracy(logits, labels, k=1)
    top5 = compute_topk_accuracy(logits, labels, k=min(5, logits.shape[1]))

    return ClassificationMetrics(top1_accuracy=top1, top5_accuracy=top5)


def compute_winoground_metrics(
    scores_c0_i0: Tensor,
    scores_c0_i1: Tensor,
    scores_c1_i0: Tensor,
    scores_c1_i1: Tensor,
) -> WinogroundMetrics:
    """
    Compute Winoground benchmark metrics.

    For each sample, we have 2 images (i0, i1) and 2 captions (c0, c1).
    Ground truth: c0 matches i0, c1 matches i1.

    scores_c0_i0: [n] similarity scores for caption_0 with image_0
    scores_c0_i1: [n] similarity scores for caption_0 with image_1
    scores_c1_i0: [n] similarity scores for caption_1 with image_0
    scores_c1_i1: [n] similarity scores for caption_1 with image_1

    Returns: WinogroundMetrics
    """
    # text score: given image, pick correct caption
    # for i0: should prefer c0 over c1 -> s(c0,i0) > s(c1,i0)
    # for i1: should prefer c1 over c0 -> s(c1,i1) > s(c0,i1)
    text_correct_i0 = scores_c0_i0 > scores_c1_i0
    text_correct_i1 = scores_c1_i1 > scores_c0_i1
    text_score = (text_correct_i0 & text_correct_i1).float().mean().item()

    # image score: given caption, pick correct image
    # for c0: should prefer i0 over i1 -> s(c0,i0) > s(c0,i1)
    # for c1: should prefer i1 over i0 -> s(c1,i1) > s(c1,i0)
    image_correct_c0 = scores_c0_i0 > scores_c0_i1
    image_correct_c1 = scores_c1_i1 > scores_c1_i0
    image_score = (image_correct_c0 & image_correct_c1).float().mean().item()

    # group score: both text and image correct
    group_correct = (
        text_correct_i0 & text_correct_i1 & image_correct_c0 & image_correct_c1
    )
    group_score = group_correct.float().mean().item()

    return WinogroundMetrics(
        text_score=text_score, image_score=image_score, group_score=group_score
    )


def aggregate_metrics(
    metrics_list: Sequence[RetrievalMetrics],
) -> RetrievalMetrics:
    """
    Aggregate multiple RetrievalMetrics (e.g., from batches) into one.
    Uses simple averaging.
    """
    n = len(metrics_list)
    return RetrievalMetrics(
        recall_at_1=sum(m.recall_at_1 for m in metrics_list) / n,
        recall_at_5=sum(m.recall_at_5 for m in metrics_list) / n,
        recall_at_10=sum(m.recall_at_10 for m in metrics_list) / n,
        median_rank=sum(m.median_rank for m in metrics_list) / n,
        mean_rank=sum(m.mean_rank for m in metrics_list) / n,
    )
