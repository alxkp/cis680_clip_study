# %% [markdown]
# ## Setup

# %%
import torch
from datasets import load_dataset
from einops import rearrange
from matplotlib import pyplot as plt
from ncut_pytorch import Ncut

# %% [markdown]
#  From docs/docs/parameters.md:
#
#   | Parameter   | Effect                     | Typical Values            |
#   |-------------|----------------------------|---------------------------|
#   | n_eig       | Number of clusters/details | 20-100                    |
#   | d_gamma     | Affinity sharpness         | 0.1-1.0 (lower = sharper) |
#   | n_sample    | Nystrom samples            | 10k-30k                   |
#   | n_neighbors | KNN for propagation        | 10-32                     |
#
#   For Your 3D Stack Use Case
#
#   clip_latents: (b, h, w) affinity scores
#   affinity = clip_latents  # (b, h, w)
#
#   Flatten to (b*h*w,) and add dummy dimension
#   features = affinity.flatten().unsqueeze(-1)  # (b*h*w, 1)
#
#   Or if you have feature dimensions:
#   features = affinity.reshape(-1, feature_dim)  # (b*h*w, d)
#
#   Run AlignedCut
#   ncut = Ncut(n_eig=50, device='cuda')
#   eigvec = ncut.fit_transform(features)
#
#   Reshape to (b, h, w, n_eig)
#   result = eigvec.reshape(b, h, w, -1)
#
#   The magic is .flatten(0, 2) - this concatenates all images/slices into one big graph, enabling aligned colors
#   across your entire stack.

# %% [markdown]
# ## Winoground experiments

# %%
winoground = load_dataset("facebook/winoground")["test"]

# %%
from transformers import AutoModel, AutoProcessor

clip_model = AutoModel.from_pretrained(
    "openai/clip-vit-base-patch32", dtype=torch.bfloat16, attn_implementation="sdpa"
)
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)

# %%
ax1 = plt.subplot(1, 3, 1)
ax1.title.set_text("image_0")
plt.imshow(winoground[155]["image_0"].convert("RGB"))

ax2 = plt.subplot(1, 3, 2)
ax2.title.set_text("image_1")
plt.imshow(winoground[155]["image_1"].convert("RGB"))

plt.show()

print("caption_0:", winoground[155]["caption_0"])
print("caption_1:", winoground[155]["caption_1"])

# Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
# Note that we could run this example through CLIP as a batch, but I want to drive the point home that we get four independent image-caption scores for each example
input_c0_i0 = clip_processor(
    text=[winoground[155]["caption_0"]],
    images=[winoground[155]["image_0"].convert("RGB")],
    return_tensors="pt",
)
input_c1_i0 = clip_processor(
    text=[winoground[155]["caption_1"]],
    images=[winoground[155]["image_0"].convert("RGB")],
    return_tensors="pt",
)
input_c0_i1 = clip_processor(
    text=[winoground[155]["caption_0"]],
    images=[winoground[155]["image_1"].convert("RGB")],
    return_tensors="pt",
)
input_c1_i1 = clip_processor(
    text=[winoground[155]["caption_1"]],
    images=[winoground[155]["image_1"].convert("RGB")],
    return_tensors="pt",
)

# %%
visual_projection = clip_model.visual_projection
text_projection = clip_model.text_projection

# %%
vision_clip = clip_model.vision_model
text_clip = clip_model.text_model

# %%
c0 = input_c0_i0["input_ids"].to(device)
c1 = input_c1_i0["input_ids"].to(device)
i0 = input_c0_i0["pixel_values"].to(device)
i1 = input_c1_i0["pixel_values"].to(device)

# %%
# image projection
im0_logits = vision_clip(i0).last_hidden_state
im0_proj = visual_projection(im0_logits)

# text projection
c0_logits = text_clip(c0).last_hidden_state
c0_proj = text_projection(c0_logits)

# %%
im0_proj.shape, c0_proj.shape

# %%
c0_proj = rearrange(c0_proj, "b l h -> b h l")
print(c0_proj.shape)

# %%
affinity_mat = im0_proj @ c0_proj
print(affinity_mat.shape)

# %%
plt.imshow(affinity_mat[0].float().detach().cpu().numpy())
plt.xlabel("text")
plt.ylabel("image")
plt.show()

# %%
from tqdm import trange

# %%
# collect all the images and then we will cut them via aligned ncut to find meaningful semantics
# only doing this on image and caption 0

all_affinities = []

for i in trange(len(winoground)):
    c = winoground[i]["caption_0"]
    i = winoground[i]["image_0"].convert("RGB")
    input = clip_processor(text=[c], images=[i], return_tensors="pt")
    c = input["input_ids"].to(device)
    i = input["pixel_values"].to(device)

    im_logits = vision_clip(i).last_hidden_state
    im_proj = visual_projection(im_logits)

    c_logits = text_clip(c).last_hidden_state
    c_proj = text_projection(c_logits)

    c_proj = rearrange(c_proj, "b l h -> b h l")

    affinity_mat = im_proj @ c_proj
    all_affinities.append(affinity_mat)


max_size = max([it.shape[2] for it in all_affinities])
print(max_size)

padded_aff = []
for it in all_affinities:
    padded_aff.append(F.pad(it, (0, max_size - it.shape[2])))

# print([it.shape for it in padded_aff])


affinities = pack(padded_aff, "* h w")[0]
print(affinities.shape)

# %%
b, h, w = affinities.shape
pos_b = torch.arange(b).view(-1, 1, 1).expand(b, h, w).to(device)
pos_h = torch.arange(h).view(1, -1, 1).expand(b, h, w).to(device)
pos_w = torch.arange(w).view(1, 1, -1).expand(b, h, w).to(device)

features = rearrange(
    torch.stack([affinities, pos_b, pos_h, pos_w], dim=-1), "b h w d -> (b h w) d"
).to(device)  # (580000, 4) - value + 3D position

# %%
eigvec = Ncut(n_eig=50, device="cuda").fit_transform(features.float())
eigvec_3d = rearrange(eigvec, "(b h w) e -> b h w e", b=400, h=50, w=29)
print(eigvec.shape, eigvec_3d.shape)

# %% [markdown]
# ### Plots

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original values at one spatial location across all slices
axes[0].plot(affinities[:, 25, 14].float().cpu(), "o-", alpha=0.7)
axes[0].set_title("Original Values at (h=25, w=14) across slices")
axes[0].set_xlabel("Slice Index")
axes[0].set_ylabel("Value")
axes[0].grid(True, alpha=0.3)

# Eigenvector values at same location (showing alignment)
for i in range(5):  # show first 5 eigenvectors
    axes[1].plot(eigvec_3d[:, 25, 14, i].cpu(), "o-", alpha=0.7, label=f"Eigenvec {i}")
axes[1].set_title("Eigenvector Values at (h=25, w=14) across slices")
axes[1].set_xlabel("Slice Index")
axes[1].set_ylabel("Eigenvector Value")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("alignment_trace.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from ncut_pytorch import Ncut

# Run AlignedCut
feats = rearrange(affinities, "b h w -> (b h w) 1").float()
eigvec = Ncut(n_eig=20, device="cuda").fit_transform(feats)
eigvec_3d = rearrange(eigvec, "(b h w) e -> b h w e", b=400, h=50, w=29)

# Convert first 3 eigenvectors to RGB
eig_rgb = torch.stack(
    [
        (eigvec_3d[..., i] - eigvec_3d[..., i].min())
        / (eigvec_3d[..., i].max() - eigvec_3d[..., i].min())
        for i in range(3)
    ],
    dim=-1,
)

# Visualize before and after
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i, idx in enumerate([0, 100, 200, 300, 399]):
    axes[0, i].imshow(affinities[idx].float().cpu(), cmap="viridis")
    axes[0, i].axis("off")
    axes[0, i].set_title(f"Before {idx}")

    axes[1, i].imshow(eig_rgb[idx].cpu())
    axes[1, i].axis("off")
    axes[1, i].set_title(f"After {idx}")

plt.tight_layout()
plt.show()

# %%
# CELL 1: Compute regular CLIP scores (baseline)
import torch
from einops import rearrange
from tqdm import trange

clip_scores = []

for i in trange(len(winoground)):
    c = winoground[i]["caption_0"]
    img = winoground[i]["image_0"].convert("RGB")

    # Compute standard CLIP score
    inputs = clip_processor(text=[c], images=[img], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)
        global_clip_score = outputs.logits_per_image.item()

    # Get the affinity matrix for token-level analysis
    affinity = all_affinities[i].squeeze(0)  # (h, w) where h=patches, w=tokens

    # Average over image patches to get per-token scores
    word_scores = affinity.mean(dim=0)  # (w,) - average over image patches

    # Get tokens
    tokens = clip_processor.tokenizer(c, return_tensors="pt")["input_ids"][0]

    clip_scores.append(
        {
            "sample_idx": i,
            "caption": c,
            "tokens": tokens,
            "global_score": global_clip_score,  # Standard CLIP score
            "word_scores": word_scores,  # Fine-grained patch-to-token scores
            "max_word_score": word_scores.max().item(),
            "mean_word_score": word_scores.mean().item(),
        }
    )

print(f"Computed CLIP scores for {len(clip_scores)} samples")
print("\nExample - Sample 0:")
print(f"  Caption: {clip_scores[0]['caption']}")
print(f"  Global CLIP score: {clip_scores[0]['global_score']:.3f}")
print(f"  Max word score: {clip_scores[0]['max_word_score']:.3f}")
print(f"  Mean word score: {clip_scores[0]['mean_word_score']:.3f}")

# Show top samples by global CLIP score
print("\nTop 5 samples by global CLIP score:")
sorted_by_global = sorted(clip_scores, key=lambda x: x["global_score"], reverse=True)
for i, sample in enumerate(sorted_by_global[:5]):
    print(f"{i + 1}. Score={sample['global_score']:.3f}: {sample['caption']}")

# %%
# CELL 2: Run AlignedCut
import torch
from einops import rearrange
from ncut_pytorch import Ncut

print(f"Input affinities shape: {affinities.shape}")

# Run AlignedCut
feats = rearrange(affinities, "b h w -> (b h w) 1").float()
print(f"Flattened features shape: {feats.shape}")

eigvec = Ncut(n_eig=20, device="cuda").fit_transform(feats)
print(f"Eigenvectors shape: {eigvec.shape}")

eigvec_3d = rearrange(eigvec, "(b h w) e -> b h w e", b=400, h=50, w=29)
print(f"Reshaped eigenvectors: {eigvec_3d.shape}")

# Convert first 3 eigenvectors to RGB for visualization
eig_rgb = torch.stack(
    [
        (eigvec_3d[..., i] - eigvec_3d[..., i].min())
        / (eigvec_3d[..., i].max() - eigvec_3d[..., i].min())
        for i in range(3)
    ],
    dim=-1,
)

print(f"RGB visualization shape: {eig_rgb.shape}")


# %%
# CELL 3: Compute eigenvector-based word scores
import torch

# Average across image patches to get word-level scores per eigenvector
word_scores = eigvec_3d.mean(dim=1)  # (400, 29, 20) - average over image patches
print(f"Word scores shape: {word_scores.shape}")
print(f"Word scores type: {type(word_scores)}")

# Verify it's a tensor
if not isinstance(word_scores, torch.Tensor):
    print("ERROR: word_scores is not a tensor!")
    print("Converting to tensor...")
    word_scores = torch.tensor(word_scores)
else:
    print("✓ word_scores is a tensor")

# %%

# CELL 4: Visualize regular CLIP scores
from collections import Counter

import matplotlib.pyplot as plt

n_top_samples = 10

# Sort by max CLIP score
clip_scores_sorted = sorted(clip_scores, key=lambda x: x["max_score"], reverse=True)

fig, axes = plt.subplots(5, 2, figsize=(16, 20))

for i in range(n_top_samples):
    row = i // 2
    col = i % 2

    sample = clip_scores_sorted[i]
    sample_idx = sample["sample_idx"]
    caption = sample["caption"]
    tokens = sample["tokens"]
    scores = sample["word_scores"].cpu()

    # Decode tokens
    words = []
    word_scores = []
    for _token_idx, (token_id, score) in enumerate(zip(tokens, scores, strict=False)):
        word = clip_processor.tokenizer.decode([token_id]).strip()
        if word and word not in ["<|startoftext|>", "<|endoftext|>", "<pad>"]:
            words.append(word)
            word_scores.append(score.item())

    # Bar chart of word scores
    axes[row, col].barh(range(len(words)), word_scores, color="steelblue", alpha=0.7)
    axes[row, col].set_yticks(range(len(words)))
    axes[row, col].set_yticklabels(words, fontsize=8)
    axes[row, col].invert_yaxis()
    axes[row, col].set_xlabel("CLIP Score", fontsize=9)
    axes[row, col].set_title(
        f"Sample {sample_idx}: {caption[:40]}...", fontsize=10, fontweight="bold"
    )
    axes[row, col].grid(True, alpha=0.3, axis="x")

plt.suptitle("Top 10 Samples by Regular CLIP Scores", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("\nTop 10 captions by max CLIP score:")

for i, sample in enumerate(clip_scores_sorted[:10]):
    print(
        f"{i + 1}. (Sample {sample['sample_idx']}, score={sample['max_score']:.3f}) {sample['caption']}"
    )


# %%
# CELL 5: Visualize eigenvector semantics
import matplotlib.pyplot as plt
import torch

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
        tokens = clip_processor.tokenizer(caption, return_tensors="pt")["input_ids"][0]

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

# %%

# CELL 6: Side-by-side comparison - CLIP vs Eigenvector
import matplotlib.pyplot as plt
import numpy as np

# Pick a few samples to compare
sample_indices = [0, 50, 100, 150, 200]

fig, axes = plt.subplots(len(sample_indices), 3, figsize=(18, 4 * len(sample_indices)))

for i, sample_idx in enumerate(sample_indices):
    caption = winoground[sample_idx]["caption_0"]
    tokens = clip_processor.tokenizer(caption, return_tensors="pt")["input_ids"][0]

    # Get CLIP scores
    clip_word_scores = clip_scores[sample_idx]["word_scores"].cpu()

    # Get eigenvector scores (use first eigenvector)
    eig_word_scores = word_scores[sample_idx, :, 0].cpu()

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


# %%

# CELL 7: Statistical comparison
from scipy.stats import spearmanr

print("=" * 100)
print("STATISTICAL COMPARISON: CLIP vs Eigenvectors")
print("=" * 100)

# Compare CLIP scores with each eigenvector
for eig_idx in range(5):
    correlations = []

    for sample_idx in range(len(winoground)):
        clip_vals = clip_scores[sample_idx]["word_scores"].float().cpu().numpy()
        eig_vals = word_scores[sample_idx, :, eig_idx].float().cpu().numpy()

        # Only compare valid tokens (non-padded)
        min_len = min(len(clip_vals), len(eig_vals))
        if min_len > 2:  # Need at least 3 points for correlation
            corr, _ = spearmanr(clip_vals[:min_len], eig_vals[:min_len])
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

# %%
# CELL 8: Compare Global CLIP Scores vs Eigenvector Scores
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

# First, recompute word_scores to make sure it's correct
print("Checking word_scores...")
print(f"eigvec_3d shape: {eigvec_3d.shape}")

# Recompute word_scores properly
word_scores = eigvec_3d.mean(
    dim=1
)  # Average over image patches (dim 1 = height/patches)
print(f"word_scores shape: {word_scores.shape}")  # Should be (400, 29, 20)

# Verify shape
assert word_scores.shape == (400, 29, 20), (
    f"Expected (400, 29, 20), got {word_scores.shape}"
)

# Compute eigenvector-based global scores (aggregate all words)
eigvec_global_scores = []

for eig_idx in range(5):  # For first 5 eigenvectors
    eig_scores = []
    for sample_idx in range(len(winoground)):
        # Get mean eigenvector score across all tokens for this sample
        sample_eig_score = word_scores[sample_idx, :, eig_idx].mean().item()
        eig_scores.append(sample_eig_score)
    eigvec_global_scores.append(eig_scores)

# Get global CLIP scores
clip_global_scores = [s["global_score"] for s in clip_scores]

print(f"clip_global_scores length: {len(clip_global_scores)}")
print(f"eigvec_global_scores[0] length: {len(eigvec_global_scores[0])}")

# Create comprehensive comparison figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)

# Row 1: Scatter plots - CLIP vs each eigenvector
for eig_idx in range(5):
    ax = fig.add_subplot(gs[0, eig_idx])

    eig_scores = eigvec_global_scores[eig_idx]

    # Scatter plot
    ax.scatter(
        clip_global_scores,
        eig_scores,
        alpha=0.5,
        s=20,
        c=clip_global_scores,
        cmap="viridis",
    )

    # Compute correlation
    pearson_r, _ = pearsonr(clip_global_scores, eig_scores)
    spearman_r, _ = spearmanr(clip_global_scores, eig_scores)

    ax.set_xlabel("Global CLIP Score", fontsize=9)
    ax.set_ylabel(f"Eigenvector {eig_idx} Score", fontsize=9)
    ax.set_title(
        f"Eigenvec {eig_idx}\nPearson={pearson_r:.3f}, Spearman={spearman_r:.3f}",
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Add diagonal reference line
    min_val = min(*clip_global_scores, *eig_scores)
    max_val = max(*clip_global_scores, *eig_scores)
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.3, linewidth=1)

# Row 2: Top samples comparison
ax_clip = fig.add_subplot(gs[1, :3])
ax_eig = fig.add_subplot(gs[1, 3:])

# Top 10 by CLIP score
top_clip_indices = np.argsort(clip_global_scores)[-10:][::-1]
top_clip_scores = [clip_global_scores[i] for i in top_clip_indices]
top_clip_captions = [
    f"S{i}: {clip_scores[i]['caption'][:40]}..." for i in top_clip_indices
]

ax_clip.barh(range(10), top_clip_scores, color="steelblue", alpha=0.7)
ax_clip.set_yticks(range(10))
ax_clip.set_yticklabels(top_clip_captions, fontsize=8)
ax_clip.invert_yaxis()
ax_clip.set_xlabel("Global CLIP Score", fontsize=10)
ax_clip.set_title("Top 10 Samples by Global CLIP Score", fontsize=11, fontweight="bold")
ax_clip.grid(True, alpha=0.3, axis="x")

# Top 10 by Eigenvector 0 score
eig_scores_0 = eigvec_global_scores[0]
top_eig_indices = np.argsort(eig_scores_0)[-10:][::-1]
top_eig_scores = [eig_scores_0[i] for i in top_eig_indices]
top_eig_captions = [
    f"S{i}: {clip_scores[i]['caption'][:40]}..." for i in top_eig_indices
]

ax_eig.barh(range(10), top_eig_scores, color="coral", alpha=0.7)
ax_eig.set_yticks(range(10))
ax_eig.set_yticklabels(top_eig_captions, fontsize=8)
ax_eig.invert_yaxis()
ax_eig.set_xlabel("Eigenvector 0 Score", fontsize=10)
ax_eig.set_title(
    "Top 10 Samples by Eigenvector 0 Score", fontsize=11, fontweight="bold"
)
ax_eig.grid(True, alpha=0.3, axis="x")

# Row 3: Agreement analysis
ax_agree = fig.add_subplot(gs[2, :2])
ax_dist = fig.add_subplot(gs[2, 2:4])
ax_rank = fig.add_subplot(gs[2, 4])

# Agreement: samples in both top-k
k_values = [10, 20, 50, 100]
agreement_scores = {f"Eig{i}": [] for i in range(5)}

for k in k_values:
    top_k_clip = set(np.argsort(clip_global_scores)[-k:])

    for eig_idx in range(5):
        top_k_eig = set(np.argsort(eigvec_global_scores[eig_idx])[-k:])
        overlap = len(top_k_clip & top_k_eig)
        agreement_scores[f"Eig{eig_idx}"].append(overlap / k * 100)

for eig_idx in range(5):
    ax_agree.plot(
        k_values,
        agreement_scores[f"Eig{eig_idx}"],
        "o-",
        label=f"Eigenvec {eig_idx}",
        linewidth=2,
        markersize=8,
    )

ax_agree.set_xlabel("Top-K", fontsize=10)
ax_agree.set_ylabel("Agreement (%)", fontsize=10)
ax_agree.set_title(
    "Top-K Agreement: CLIP vs Eigenvectors", fontsize=11, fontweight="bold"
)
ax_agree.legend(fontsize=8)
ax_agree.grid(True, alpha=0.3)
ax_agree.set_ylim([0, 100])

# Distribution comparison
ax_dist.hist(
    clip_global_scores,
    bins=30,
    alpha=0.5,
    label="CLIP",
    color="steelblue",
    density=True,
)
ax_dist.hist(
    eigvec_global_scores[0],
    bins=30,
    alpha=0.5,
    label="Eigenvec 0",
    color="coral",
    density=True,
)
ax_dist.hist(
    eigvec_global_scores[1],
    bins=30,
    alpha=0.5,
    label="Eigenvec 1",
    color="green",
    density=True,
)
ax_dist.set_xlabel("Score", fontsize=10)
ax_dist.set_ylabel("Density", fontsize=10)
ax_dist.set_title("Score Distribution Comparison", fontsize=11, fontweight="bold")
ax_dist.legend(fontsize=9)
ax_dist.grid(True, alpha=0.3, axis="y")

# Rank correlation heatmap
rank_corrs = np.zeros((6, 6))
all_scores = [clip_global_scores, *eigvec_global_scores]

for i in range(6):
    for j in range(6):
        rank_corrs[i, j] = spearmanr(all_scores[i], all_scores[j])[0]

im = ax_rank.imshow(rank_corrs, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax_rank.set_xticks(range(6))
ax_rank.set_yticks(range(6))
ax_rank.set_xticklabels(["CLIP"] + [f"E{i}" for i in range(5)], fontsize=9)
ax_rank.set_yticklabels(["CLIP"] + [f"E{i}" for i in range(5)], fontsize=9)
ax_rank.set_title("Rank Correlation\n(Spearman)", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax_rank, fraction=0.046, pad=0.04)

# Add correlation values as text
for i in range(6):
    for j in range(6):
        text = ax_rank.text(
            j,
            i,
            f"{rank_corrs[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=8,
        )

plt.suptitle(
    "Global CLIP Scores vs Eigenvector Scores: Comprehensive Comparison",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)
plt.savefig("clip_vs_eigenvector_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# Print summary statistics
print("\n" + "=" * 100)
print("SUMMARY: GLOBAL CLIP vs EIGENVECTOR SCORES")
print("=" * 100)

for eig_idx in range(5):
    eig_scores = eigvec_global_scores[eig_idx]
    pearson_r, _ = pearsonr(clip_global_scores, eig_scores)
    spearman_r, _ = spearmanr(clip_global_scores, eig_scores)

    print(f"\nEigenvector {eig_idx}:")
    print(f"  Pearson correlation:  {pearson_r:.4f}")
    print(f"  Spearman correlation: {spearman_r:.4f}")
    print(
        f"  Mean score: {np.mean(eig_scores):.4f} (CLIP: {np.mean(clip_global_scores):.4f})"
    )
    print(
        f"  Std score:  {np.std(eig_scores):.4f} (CLIP: {np.std(clip_global_scores):.4f})"
    )

    # Top-10 overlap
    top_10_clip = set(np.argsort(clip_global_scores)[-10:])
    top_10_eig = set(np.argsort(eig_scores)[-10:])
    overlap = len(top_10_clip & top_10_eig)
    print(f"  Top-10 overlap: {overlap}/10 samples")

print("\n" + "=" * 100)

# Show examples where CLIP and Eigenvector disagree strongly
print("\nSamples where CLIP ranks high but Eigenvector 0 ranks low:")
print("-" * 100)
clip_ranks = np.argsort(np.argsort(clip_global_scores)[::-1])
eig0_ranks = np.argsort(np.argsort(eigvec_global_scores[0])[::-1])
rank_diff = clip_ranks - eig0_ranks

most_disagreed = np.argsort(rank_diff)[:5]
for idx in most_disagreed:
    print(f"Sample {idx} (CLIP rank: {clip_ranks[idx]}, Eig0 rank: {eig0_ranks[idx]}):")
    print(f"  {clip_scores[idx]['caption']}")
    print(
        f"  CLIP score: {clip_global_scores[idx]:.3f}, Eig0 score: {eigvec_global_scores[0][idx]:.3f}"
    )
    print()

print("\nSamples where Eigenvector 0 ranks high but CLIP ranks low:")
print("-" * 100)
most_disagreed_reverse = np.argsort(rank_diff)[-5:][::-1]
for idx in most_disagreed_reverse:
    print(f"Sample {idx} (CLIP rank: {clip_ranks[idx]}, Eig0 rank: {eig0_ranks[idx]}):")
    print(f"  {clip_scores[idx]['caption']}")
    print(
        f"  CLIP score: {clip_global_scores[idx]:.3f}, Eig0 score: {eigvec_global_scores[0][idx]:.3f}"
    )
    print()

# %% [markdown]
# ## Loss Formulations

# %%
# %%
from einops import pack
from torch.nn import functional as F

# %%
# collect all the images and then we will cut them via aligned ncut to find meaningful semantics
# only doing this on image and caption 0

all_affinities = []

for i in trange(len(winoground)):
    c = winoground[i]["caption_0"]
    i = winoground[i]["image_0"].convert("RGB")
    input = clip_processor(text=[c], images=[i], return_tensors="pt")
    c = input["input_ids"].to(device)
    i = input["pixel_values"].to(device)

    im_logits = vision_clip(i).last_hidden_state
    im_proj = visual_projection(im_logits)

    c_logits = text_clip(c).last_hidden_state
    c_proj = text_projection(c_logits)

    c_proj = rearrange(c_proj, "b l h -> b h l")

    affinity_mat = im_proj @ c_proj
    all_affinities.append(affinity_mat)


max_size = max([it.shape[2] for it in all_affinities])
print(max_size)

padded_aff = []
for it in all_affinities:
    padded_aff.append(F.pad(it, (0, max_size - it.shape[2])))

# print([it.shape for it in padded_aff])


affinities = pack(padded_aff, "* h w")[0]
print(affinities.shape)


# %%
b, h, w = affinities.shape
pos_b = torch.arange(b).view(-1, 1, 1).expand(b, h, w).to(device)
pos_h = torch.arange(h).view(1, -1, 1).expand(b, h, w).to(device)
pos_w = torch.arange(w).view(1, 1, -1).expand(b, h, w).to(device)

features = rearrange(
    torch.stack([affinities, pos_b, pos_h, pos_w], dim=-1), "b h w d -> (b h w) d"
).to(device)  # (580000, 4) - value + 3D position

# %%
eigvec = Ncut(n_eig=50, device="cuda").fit_transform(features.float())
eigvec_3d = rearrange(eigvec, "(b h w) e -> b h w e", b=400, h=50, w=29)
print(eigvec.shape, eigvec_3d.shape)
