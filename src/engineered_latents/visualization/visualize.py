from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from engineered_latents.types import ArrayLike


def compare_correlation(clip_scores: ArrayLike, word_scores: ArrayLike):
    print("=" * 100)
    print("STATISTICAL COMPARISON: CLIP vs Eigenvectors")
    print("=" * 100)

    # Compare CLIP scores with each eigenvector
    for eig_idx in range(5):
        correlations = []

        for sample_idx in range(len(clip_scores)):
            clip_vals = clip_scores[sample_idx]["word_scores"].float().cpu().numpy()
            eig_vals = word_scores[sample_idx, :, eig_idx].float().cpu().numpy()

            # Only compare valid tokens (non-padded)
            min_len = min(len(clip_vals), len(eig_vals))
            if min_len > 2:  # Need at least 3 points for correlation
                corr, _ = spearmanr(clip_vals[:min_len], eig_vals[:min_len])
                assert isinstance(corr, float)  # if its got an error we have a problem
                if not np.isnan(corr):
                    correlations.append(corr)

        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)

        print(f"\nEigenvector {eig_idx}:")
        print(f"  Mean correlation with CLIP: {mean_corr:.3f} ± {std_corr:.3f}")
        print(
            f"  Samples with positive correlation: {sum(c > 0 for c in correlations)} / {len(correlations)}"
        )
        print(
            f"  Samples with strong correlation (>0.5): {sum(c > 0.5 for c in correlations)} / {len(correlations)}"
        )

    print("\n" + "=" * 100)


def visualize_eigenvector_semantics(word_scores: ArrayLike, winoground, clip_processor):
    # Double-check word_scores is a tensor
    print(f"word_scores type: {type(word_scores)}")
    print(f"word_scores shape: {word_scores.shape}")

    n_eigvec_to_show = 5
    n_top_captions = 10

    for eig_idx in range(n_eigvec_to_show):
        print(f"\n{'=' * 100}")
        print(f"EIGENVECTOR {eig_idx}")
        print(f"{'=' * 100}")

        # Get word scores for this eigenvector across all samples
        eig_word_scores = word_scores[:, :, eig_idx]  # (400, 29)

        print(f"eig_word_scores shape: {eig_word_scores.shape}")

        # Find top samples for this eigenvector
        sample_max_scores = eig_word_scores.max(dim=1)[0]  # (400,)
        top_sample_indices = torch.topk(sample_max_scores, k=n_top_captions)[1]

        # Create figure for this eigenvector
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        fig.suptitle(
            f"Eigenvector {eig_idx}: Top Captions and Word Activations",
            fontsize=14,
            fontweight="bold",
        )

        word_counter = Counter()

        for rank, sample_idx in enumerate(top_sample_indices.cpu()):
            if rank >= 10:
                break

            sample_idx = sample_idx.item()
            caption = winoground[sample_idx]["caption_0"]

            # Tokenize caption
            tokens = clip_processor.tokenizer(caption, return_tensors="pt")[
                "input_ids"
            ][0]

            # Get scores for this sample's tokens
            scores = eig_word_scores[sample_idx].cpu()

            # Decode tokens with their scores
            words = []
            word_score_list = []
            for _token_idx, (token_id, score) in enumerate(
                zip(tokens, scores, strict=False)
            ):
                word = clip_processor.tokenizer.decode([token_id]).strip()
                if word and word not in ["<|startoftext|>", "<|endoftext|>", "<pad>"]:
                    words.append(word)
                    word_score_list.append(score.item())
                    word_counter[word] += 1

            # Plot in grid
            if rank < 10:
                row = rank // 4
                col = rank % 4

                axes[row, col].barh(
                    range(len(words)), word_score_list, color="coral", alpha=0.7
                )
                axes[row, col].set_yticks(range(len(words)))
                axes[row, col].set_yticklabels(words, fontsize=7)
                axes[row, col].invert_yaxis()
                axes[row, col].set_xlabel("Score", fontsize=8)
                axes[row, col].set_title(
                    f"#{rank + 1} (S{sample_idx}): {caption[:30]}...", fontsize=9
                )
                axes[row, col].grid(True, alpha=0.3, axis="x")

            # Print details
            if rank < 5:
                print(
                    f"\n#{rank + 1} Sample {sample_idx} (max={sample_max_scores[sample_idx]:.3f}):"
                )
                print(f'  Caption: "{caption}"')
                word_score_pairs = list(zip(words, word_score_list, strict=False))
                word_score_pairs.sort(key=lambda x: x[1], reverse=True)
                print(
                    f"  Top words: {', '.join([f'{w}({s:.3f})' for w, s in word_score_pairs[:5]])}"
                )

        # Hide unused subplots
        for idx in range(10, 12):
            row = idx // 4
            col = idx % 4
            if row < 3 and col < 4:
                axes[row, col].axis("off")

        plt.tight_layout()
        plt.show()

        # Print word frequency
        top_words = word_counter.most_common(10)
        print(f"\nMost common words: {', '.join([f'{w}({c})' for w, c in top_words])}")


def visualize_clip_vs_eigenvector(
    winoground, clip_processor, clip_scores: ArrayLike, word_scores: ArrayLike
):
    # Pick a few samples to compare
    sample_indices = [0, 50, 100, 150, 200]

    _fig, axes = plt.subplots(
        len(sample_indices), 3, figsize=(18, 4 * len(sample_indices))
    )

    for i, sample_idx in enumerate(sample_indices):
        caption = winoground[sample_idx]["caption_0"]
        tokens = clip_processor.tokenizer(caption, return_tensors="pt")["input_ids"][0]

        # Get CLIP scores
        clip_word_scores = clip_scores[sample_idx]["word_scores"].cpu()

        # Get eigenvector scores (use first eigenvector)
        eig_word_scores = word_scores[sample_idx, :, 0].cpu()

        eig_rgb = eig_word_scores.mean(dim=1)

        # Decode tokens
        words = []
        clip_vals = []
        eig_vals = []
        for _token_idx, (token_id, clip_s, eig_s) in enumerate(
            zip(tokens, clip_word_scores, eig_word_scores, strict=False)
        ):
            word = clip_processor.tokenizer.decode([token_id]).strip()
            if word and word not in ["<|startoftext|>", "<|endoftext|>", "<pad>"]:
                words.append(word)
                clip_vals.append(clip_s.item())
                eig_vals.append(eig_s.item())

        # Plot CLIP scores
        axes[i, 0].barh(range(len(words)), clip_vals, color="steelblue", alpha=0.7)
        axes[i, 0].set_yticks(range(len(words)))
        axes[i, 0].set_yticklabels(words, fontsize=9)
        axes[i, 0].invert_yaxis()
        axes[i, 0].set_xlabel("CLIP Score", fontsize=9)
        axes[i, 0].set_title(f"CLIP Scores - Sample {sample_idx}", fontsize=10)
        axes[i, 0].grid(True, alpha=0.3, axis="x")

        # Plot Eigenvector scores
        axes[i, 1].barh(range(len(words)), eig_vals, color="coral", alpha=0.7)
        axes[i, 1].set_yticks(range(len(words)))
        axes[i, 1].set_yticklabels(words, fontsize=9)
        axes[i, 1].invert_yaxis()
        axes[i, 1].set_xlabel("Eigenvector Score", fontsize=9)
        axes[i, 1].set_title(f"Eigenvector 0 - Sample {sample_idx}", fontsize=10)
        axes[i, 1].grid(True, alpha=0.3, axis="x")

        # Show caption and RGB visualization
        axes[i, 2].imshow(eig_rgb[sample_idx].cpu())
        axes[i, 2].axis("off")
        axes[i, 2].set_title(f"{caption}", fontsize=9, wrap=True)

    plt.suptitle(
        "Comparison: CLIP Scores vs Eigenvector Scores", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()
