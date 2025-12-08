"""
NCut visualization for tracking latent structure during evaluation.
"""

from typing import Any, cast

import matplotlib.pyplot as plt
import torch
from aim import Image as AimImage
from datasets import load_dataset
from einops import pack, rearrange
from ncut_pytorch import Ncut
from torch.nn import functional as F
from tqdm import tqdm

from ..context import get_clip, get_run


def compute_affinity_matrices(
    samples: list[dict[str, Any]],
    max_samples: int = 100,
) -> tuple[torch.Tensor, list[str]]:
    """
    Compute image-text affinity matrices for a set of samples.

    Returns:
        affinities: (n_samples, n_patches, max_tokens) padded tensor
        captions: list of caption strings
    """
    ctx = get_clip()
    model = ctx.model.to(ctx.device)
    model.eval()

    vision_model = model.vision_model
    text_model = model.text_model
    visual_projection = model.visual_projection
    text_projection = model.text_projection

    all_affinities = []
    captions = []

    n = min(len(samples), max_samples)

    affinity_pbar = tqdm(range(n), desc="Computing affinities")
    assert isinstance(
        affinity_pbar, tqdm
    )  # HACK: another fix - if this breaks, we have problems anyway

    for i in affinity_pbar:
        sample = samples[i]
        caption = sample["caption_0"]
        image = sample["image_0"].convert("RGB")

        inputs = ctx.processor(text=[caption], images=[image], return_tensors="pt")
        input_ids = inputs["input_ids"].to(ctx.device)
        pixel_values = inputs["pixel_values"].to(ctx.device)

        with torch.no_grad():
            # Get patch-level image features
            im_hidden = vision_model(pixel_values).last_hidden_state
            im_proj = visual_projection(im_hidden)

            # Get token-level text features
            txt_hidden = text_model(input_ids).last_hidden_state
            txt_proj = text_projection(txt_hidden)

            # Compute affinity matrix: (batch, patches, hidden) @ (batch, hidden, tokens)
            txt_proj = rearrange(txt_proj, "b l h -> b h l")
            affinity = im_proj @ txt_proj  # (1, n_patches, n_tokens)

        all_affinities.append(affinity.cpu())
        captions.append(caption)

    # Pad to max token length
    max_tokens = max(a.shape[2] for a in all_affinities)
    padded = [F.pad(a, (0, max_tokens - a.shape[2])) for a in all_affinities]

    # Stack: (n_samples, n_patches, max_tokens)
    affinities = pack(padded, "* h w")[0]

    return affinities, captions


def compute_ncut_eigenvectors(
    affinities: torch.Tensor,
    n_eig: int = 20,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Run NCut on stacked affinity matrices.

    Args:
        affinities: (b, h, w) tensor of affinity matrices
        n_eig: number of eigenvectors to compute
        device: device for NCut computation

    Returns:
        eigvec_3d: (b, h, w, n_eig) eigenvector tensor
    """
    b, h, w = affinities.shape

    # Flatten for NCut
    feats = rearrange(affinities, "b h w -> (b h w) 1").float()

    # Run NCut
    ncut = Ncut(n_eig=n_eig, device=device)
    assert isinstance(ncut, Ncut)  # HACK: type system fixes

    eigvec = ncut.fit_transform(feats)

    # Reshape back
    eigvec_3d = rearrange(eigvec, "(b h w) e -> b h w e", b=b, h=h, w=w)

    return eigvec_3d


def eigvec_to_rgb(eigvec_3d: torch.Tensor) -> torch.Tensor:
    """
    Convert first 3 eigenvectors to RGB for visualization.

    Args:
        eigvec_3d: (b, h, w, n_eig) eigenvector tensor

    Returns:
        rgb: (b, h, w, 3) normalized RGB tensor
    """
    rgb_channels = []
    for i in range(3):
        channel = eigvec_3d[..., i]
        # Normalize to [0, 1]
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        rgb_channels.append(channel)

    return torch.stack(rgb_channels, dim=-1)


def create_ncut_visualization(
    affinities: torch.Tensor,
    eig_rgb: torch.Tensor,
    captions: list[str],
    sample_indices: list[int] | None = None,
) -> plt.Figure:
    """
    Create before/after visualization of NCut clustering.

    Args:
        affinities: (b, h, w) raw affinity matrices
        eig_rgb: (b, h, w, 3) RGB-mapped eigenvectors
        captions: list of captions
        sample_indices: which samples to visualize (default: evenly spaced)

    Returns:
        matplotlib Figure
    """
    n_samples = affinities.shape[0]

    if sample_indices is None:
        # Pick 5 evenly spaced samples
        sample_indices = [int(i * (n_samples - 1) / 4) for i in range(5)]

    n_show = len(sample_indices)
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))

    for col, idx in enumerate(sample_indices):
        # Before: raw affinity
        axes[0, col].imshow(affinities[idx].float().cpu().numpy(), cmap="viridis")
        axes[0, col].axis("off")
        axes[0, col].set_title(f"Raw Affinity [{idx}]", fontsize=9)

        # After: eigenvector RGB
        axes[1, col].imshow(eig_rgb[idx].cpu().numpy())
        axes[1, col].axis("off")
        caption_short = (
            captions[idx][:30] + "..." if len(captions[idx]) > 30 else captions[idx]
        )
        axes[1, col].set_title(f"NCut RGB: {caption_short}", fontsize=8)

    axes[0, 0].set_ylabel("Before (Affinity)", fontsize=10)
    axes[1, 0].set_ylabel("After (NCut RGB)", fontsize=10)

    plt.suptitle(
        "NCut Clustering on Image-Text Affinities", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    return fig


def log_ncut_visualization(
    step: int | None = None,
    max_samples: int = 100,
    n_eig: int = 20,
) -> None:
    """
    Compute and log NCut visualization to Aim.

    Args:
        step: training step for logging
        max_samples: number of Winoground samples to use
        n_eig: number of eigenvectors for NCut
    """
    # Load Winoground
    samples = list(load_dataset("facebook/winoground", split="test"))
    samples = cast("list[dict[str, Any]]", samples)  # HACK: type system fixes

    # Compute affinities
    affinities, captions = compute_affinity_matrices(samples, max_samples)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run NCut
    eigvec_3d = compute_ncut_eigenvectors(affinities, n_eig=n_eig, device=device)

    # Convert to RGB
    eig_rgb = eigvec_to_rgb(eigvec_3d)

    # Create visualization
    fig = create_ncut_visualization(affinities, eig_rgb, captions)

    # Log to Aim
    run = get_run()
    run.track(AimImage(fig), name="ncut_visualization", step=step)

    plt.close(fig)

    print(f"Logged NCut visualization (step={step})")
