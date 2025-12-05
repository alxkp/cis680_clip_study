import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from engineered_latents.type_defs import ArrayLike
from engineered_latents.utils import OUTPUT_DIR


def cka_alignment_loss(
    X: torch.Tensor, Y: torch.Tensor, center: bool = True
) -> torch.Tensor:
    """
    Centered Kernel Alignment (CKA) loss for measuring representation similarity.
    Returns negative CKA (for minimization).

    X: [n_samples, d1] - e.g., image patch features
    Y: [n_samples, d2] - e.g., text token features
    center: whether to center the Gram matrices (standard CKA)

    Returns: -CKA (so minimizing this maximizes alignment)
    """
    # Compute Gram matrices (linear kernel)
    K = X @ X.T  # [n, n]
    L = Y @ Y.T  # [n, n]

    if center:
        # Center the Gram matrices
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        K = H @ K @ H
        L = H @ L @ H

    # HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_kl = (K * L).sum()
    hsic_kk = (K * K).sum()
    hsic_ll = (L * L).sum()

    # CKA = HSIC(K,L) / sqrt(HSIC(K,K) * HSIC(L,L))
    cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-8)

    return -cka  # Negative for minimization


def cross_covariance_svd_loss(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int = 3,
    alpha: float = 1.0,
    beta: float = 0.5,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Loss based on singular values of cross-covariance matrix.
    Encourages a small number of high-magnitude alignment directions.

    X: [n_samples, d1] - e.g., image patch features
    Y: [n_samples, d2] - e.g., text token features
    k: number of alignment directions to encourage
    alpha: weight for spectral gap loss
    beta: weight for magnitude loss
    normalize: whether to L2-normalize features first

    Returns: combined loss (minimize to get k strong alignment directions)
    """
    if normalize:
        X = X / (X.norm(dim=-1, keepdim=True) + 1e-8)
        Y = Y / (Y.norm(dim=-1, keepdim=True) + 1e-8)

    # Cross-covariance matrix
    C = X.T @ Y  # [d1, d2]

    # SVD to get singular values
    S = torch.linalg.svdvals(C)  # sorted descending

    # Maximize spectral gap after k-th singular value
    # S is descending, so gap is S[k-1] - S[k]
    gap_loss = -(S[k - 1] - S[k])

    # Maximize top-k singular values
    magnitude_loss = -S[:k].sum()

    return alpha * gap_loss + beta * magnitude_loss


def ncut_cluster_loss(
    eigenvalues: ArrayLike, k: int = 3, alpha: float = 1.0, beta: float = 0.5
) -> torch.Tensor:
    """
    Computes loss that encourages high magnitude and a small number of clusters
    eigenvalues: sorted eigenvalues of normalized Laplacian (ascending)
    k: number of clusters
    alpha: weight for spectral gap loss
    beta: weight for cluster loss
    """
    # maximize spectral gap
    gap_loss = -(eigenvalues[k] - eigenvalues[k - 1])

    # maximize top-k eigenvalues (for better clusters)
    # use starting at 1 to avoid the first eigenvalue (which is always 0)
    cluster_loss = -1 * eigenvalues[1 : k + 1].sum()

    return alpha * gap_loss + beta * cluster_loss


def plot_loss_surface(
    eigenvalues, k=3, alpha_range=(0.1, 2.0), beta_range=(0.1, 2.0), resolution=50
):
    """
    Plot loss surface over alpha/beta hyperparameters
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], resolution)
    betas = np.linspace(beta_range[0], beta_range[1], resolution)

    Alpha, Beta = np.meshgrid(alphas, betas)
    Loss = np.zeros_like(Alpha)

    for i in range(resolution):
        for j in range(resolution):
            Loss[i, j] = ncut_cluster_loss(eigenvalues, k, Alpha[i, j], Beta[i, j])

    fig = plt.figure(figsize=(15, 5))

    # 3D surface
    ax1 = fig.add_subplot(131, projection="3d")
    surf = ax1.plot_surface(
        Alpha, Beta, Loss, cmap=cm.get_cmap("viridis"), alpha=0.8, edgecolor="none"
    )
    ax1.set_xlabel("Alpha (gap weight)")
    ax1.set_ylabel("Beta (cluster weight)")
    ax1.set_zlabel("Loss")
    ax1.set_title("Loss Surface")
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # Contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(Alpha, Beta, Loss, levels=20, cmap=cm.get_cmap("viridis"))
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Beta")
    ax2.set_title("Loss Contours")
    fig.colorbar(contour, ax=ax2)

    # Eigenvalue spectrum with loss components
    ax3 = fig.add_subplot(133)
    ax3.plot(eigenvalues[: min(20, len(eigenvalues))], "o-", linewidth=2)
    ax3.axvline(k - 1, color="r", linestyle="--", label=f"Gap at k={k}")
    ax3.axvline(k, color="r", linestyle="--")
    ax3.fill_between(
        range(k), 0, eigenvalues[:k], alpha=0.3, color="blue", label=f"Top-{k} sum"
    )
    ax3.set_xlabel("Eigenvalue index")
    ax3.set_ylabel("Eigenvalue")
    ax3.set_title("Eigenvalue Spectrum")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_loss_vs_k(eigenvalues, k_range=None, alpha=1.0, beta=0.5):
    """
    Plot how loss changes as you vary k (number of clusters)
    """
    if k_range is None:
        k_range = range(2, min(len(eigenvalues), 20))

    losses = []
    gaps = []
    cluster_costs = []

    for k in k_range:
        loss = ncut_cluster_loss(eigenvalues, k, alpha, beta)
        gap = eigenvalues[k] - eigenvalues[k - 1]
        cluster_cost = eigenvalues[:k].sum()

        losses.append(loss)
        gaps.append(gap)
        cluster_costs.append(cluster_cost)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Total loss
    axes[0, 0].plot(k_range, losses, "o-", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("k (number of clusters)")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Loss vs k")
    axes[0, 0].grid(True, alpha=0.3)

    # Spectral gap
    axes[0, 1].plot(k_range, gaps, "o-", linewidth=2, markersize=8, color="orange")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("Spectral Gap")
    axes[0, 1].set_title("Eigengap vs k")
    axes[0, 1].grid(True, alpha=0.3)

    # Cluster cost
    axes[1, 0].plot(
        k_range, cluster_costs, "o-", linewidth=2, markersize=8, color="green"
    )
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("Sum of top-k eigenvalues")
    axes[1, 0].set_title("Cluster Cost vs k")
    axes[1, 0].grid(True, alpha=0.3)

    # Eigenvalue spectrum with markers
    axes[1, 1].plot(eigenvalues[: min(30, len(eigenvalues))], "o-", alpha=0.6)
    for k in k_range[::3]:  # Mark every 3rd k value
        axes[1, 1].axvline(k, alpha=0.3, linestyle="--")
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("Eigenvalue")
    axes[1, 1].set_title("Eigenvalue Spectrum")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cka_loss_surface(
    X: torch.Tensor,
    Y: torch.Tensor,
    noise_range: tuple = (0.0, 2.0),
    resolution: int = 50,
):
    """
    Plot CKA loss as we add noise to perturb alignment.
    Shows how CKA degrades with misalignment.
    """
    noise_levels = np.linspace(noise_range[0], noise_range[1], resolution)

    cka_values = []
    for noise in noise_levels:
        Y_noisy = Y + noise * torch.randn_like(Y)
        cka_val = -cka_alignment_loss(X, Y_noisy).item()  # Negate to get actual CKA
        cka_values.append(cka_val)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CKA vs noise
    axes[0].plot(noise_levels, cka_values, "o-", linewidth=2, markersize=4)
    axes[0].set_xlabel("Noise Level")
    axes[0].set_ylabel("CKA")
    axes[0].set_title("CKA vs Perturbation")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.1)

    # Gram matrices visualization
    K = X @ X.T
    L = Y @ Y.T
    im1 = axes[1].imshow(K.detach().cpu().numpy(), cmap="viridis", aspect="auto")
    axes[1].set_title("Gram Matrix K (X)")
    fig.colorbar(im1, ax=axes[1], shrink=0.7)

    im2 = axes[2].imshow(L.detach().cpu().numpy(), cmap="viridis", aspect="auto")
    axes[2].set_title("Gram Matrix L (Y)")
    fig.colorbar(im2, ax=axes[2], shrink=0.7)

    plt.tight_layout()
    return fig


def plot_svd_loss_surface(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int = 3,
    alpha_range: tuple = (0.1, 2.0),
    beta_range: tuple = (0.1, 2.0),
    resolution: int = 50,
):
    """
    Plot cross-covariance SVD loss surface over alpha/beta hyperparameters.
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], resolution)
    betas = np.linspace(beta_range[0], beta_range[1], resolution)

    Alpha, Beta = np.meshgrid(alphas, betas)
    Loss = np.zeros_like(Alpha)

    for i in range(resolution):
        for j in range(resolution):
            Loss[i, j] = cross_covariance_svd_loss(
                X, Y, k, Alpha[i, j], Beta[i, j]
            ).item()

    fig = plt.figure(figsize=(15, 5))

    # 3D surface
    ax1 = fig.add_subplot(131, projection="3d")
    surf = ax1.plot_surface(
        Alpha, Beta, Loss, cmap=cm.get_cmap("viridis"), alpha=0.8, edgecolor="none"
    )
    ax1.set_xlabel("Alpha (gap weight)")
    ax1.set_ylabel("Beta (magnitude weight)")
    ax1.set_zlabel("Loss")
    ax1.set_title("SVD Loss Surface")
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # Contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(Alpha, Beta, Loss, levels=20, cmap=cm.get_cmap("viridis"))
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Beta")
    ax2.set_title("SVD Loss Contours")
    fig.colorbar(contour, ax=ax2)

    # Singular value spectrum
    X_norm = X / (X.norm(dim=-1, keepdim=True) + 1e-8)
    Y_norm = Y / (Y.norm(dim=-1, keepdim=True) + 1e-8)
    C = X_norm.T @ Y_norm
    S = torch.linalg.svdvals(C).detach().cpu().numpy()

    ax3 = fig.add_subplot(133)
    ax3.plot(S[: min(30, len(S))], "o-", linewidth=2)
    ax3.axvline(k - 1, color="r", linestyle="--", label=f"Gap at k={k}")
    ax3.axvline(k, color="r", linestyle="--")
    ax3.fill_between(range(k), 0, S[:k], alpha=0.3, color="blue", label=f"Top-{k} sum")
    ax3.set_xlabel("Singular value index")
    ax3.set_ylabel("Singular value")
    ax3.set_title("Cross-Covariance Singular Values")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_svd_loss_vs_k(
    X: torch.Tensor,
    Y: torch.Tensor,
    k_range: range | None = None,
    alpha: float = 1.0,
    beta: float = 0.5,
):
    """
    Plot how SVD loss changes as you vary k (number of alignment directions).
    """
    X_norm = X / (X.norm(dim=-1, keepdim=True) + 1e-8)
    Y_norm = Y / (Y.norm(dim=-1, keepdim=True) + 1e-8)
    C = X_norm.T @ Y_norm
    S = torch.linalg.svdvals(C)

    if k_range is None:
        k_range = range(1, min(len(S) - 1, 20))

    losses = []
    gaps = []
    magnitude_costs = []

    for k in k_range:
        loss = cross_covariance_svd_loss(X, Y, k, alpha, beta).item()
        gap = (S[k - 1] - S[k]).item()
        magnitude = S[:k].sum().item()

        losses.append(loss)
        gaps.append(gap)
        magnitude_costs.append(magnitude)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Total loss
    axes[0, 0].plot(list(k_range), losses, "o-", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("k (alignment directions)")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("SVD Loss vs k")
    axes[0, 0].grid(True, alpha=0.3)

    # Spectral gap
    axes[0, 1].plot(
        list(k_range), gaps, "o-", linewidth=2, markersize=8, color="orange"
    )
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("Spectral Gap (S[k-1] - S[k])")
    axes[0, 1].set_title("Singular Value Gap vs k")
    axes[0, 1].grid(True, alpha=0.3)

    # Magnitude (top-k sum)
    axes[1, 0].plot(
        list(k_range), magnitude_costs, "o-", linewidth=2, markersize=8, color="green"
    )
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("Sum of top-k singular values")
    axes[1, 0].set_title("Alignment Magnitude vs k")
    axes[1, 0].grid(True, alpha=0.3)

    # Singular value spectrum
    S_np = S.detach().cpu().numpy()
    axes[1, 1].plot(S_np[: min(30, len(S_np))], "o-", alpha=0.6)
    for k in list(k_range)[::3]:
        axes[1, 1].axvline(k, alpha=0.3, linestyle="--")
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("Singular Value")
    axes[1, 1].set_title("Singular Value Spectrum")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_alignment_comparison(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int = 3,
    alpha: float = 1.0,
    beta: float = 0.5,
):
    """
    Side-by-side comparison of CKA and SVD-based alignment metrics.
    Useful for ablation studies.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: CKA analysis
    # CKA value
    cka_val = -cka_alignment_loss(X, Y).item()
    axes[0, 0].bar(["CKA"], [cka_val], color="steelblue")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title(f"CKA Score: {cka_val:.4f}")
    axes[0, 0].set_ylabel("CKA")

    # Gram matrices
    n = X.shape[0]
    H = torch.eye(n, device=X.device) - torch.ones(n, n, device=X.device) / n
    K = H @ (X @ X.T) @ H
    L = H @ (Y @ Y.T) @ H

    im1 = axes[0, 1].imshow(K.detach().cpu().numpy(), cmap="coolwarm", aspect="auto")
    axes[0, 1].set_title("Centered Gram K (X)")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.7)

    im2 = axes[0, 2].imshow(L.detach().cpu().numpy(), cmap="coolwarm", aspect="auto")
    axes[0, 2].set_title("Centered Gram L (Y)")
    fig.colorbar(im2, ax=axes[0, 2], shrink=0.7)

    # Row 2: SVD analysis
    X_norm = X / (X.norm(dim=-1, keepdim=True) + 1e-8)
    Y_norm = Y / (Y.norm(dim=-1, keepdim=True) + 1e-8)
    C = X_norm.T @ Y_norm
    S = torch.linalg.svdvals(C).detach().cpu().numpy()

    # Singular value spectrum
    axes[1, 0].plot(S[: min(20, len(S))], "o-", linewidth=2)
    axes[1, 0].axvline(k - 0.5, color="r", linestyle="--", alpha=0.7)
    axes[1, 0].fill_between(range(k), 0, S[:k], alpha=0.3, color="blue")
    axes[1, 0].set_xlabel("Index")
    axes[1, 0].set_ylabel("Singular Value")
    axes[1, 0].set_title("Cross-Cov Singular Values")
    axes[1, 0].grid(True, alpha=0.3)

    # Cross-covariance matrix
    im3 = axes[1, 1].imshow(C.detach().cpu().numpy(), cmap="coolwarm", aspect="auto")
    axes[1, 1].set_title("Cross-Covariance Matrix")
    axes[1, 1].set_xlabel("Y dimensions")
    axes[1, 1].set_ylabel("X dimensions")
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.7)

    # Loss components breakdown
    svd_loss = cross_covariance_svd_loss(X, Y, k, alpha, beta).item()
    gap = S[k - 1] - S[k]
    top_k_sum = S[:k].sum()

    axes[1, 2].bar(
        ["Total Loss", "Gap (neg)", "Top-k Sum (neg)"],
        [svd_loss, -alpha * gap, -beta * top_k_sum],
        color=["purple", "orange", "green"],
    )
    axes[1, 2].axhline(0, color="black", linewidth=0.5)
    axes[1, 2].set_title(f"SVD Loss Components (k={k})")
    axes[1, 2].set_ylabel("Loss Value")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage with eigenvalues (original):
    print("Testing ncut_cluster_loss...")
    eigenvalues = torch.randn(100).abs().sort().values
    fig = plot_loss_surface(eigenvalues, k=3)
    plt.savefig("ncut_loss_surface.png", dpi=150)
    plt.close()

    fig = plot_loss_vs_k(eigenvalues, k_range=range(2, 15), alpha=1.0, beta=0.5)
    plt.savefig(OUTPUT_DIR / "ncut_loss_vs_k.png", dpi=150)
    plt.close()

    # Example usage with feature matrices (new):
    print("Testing CKA alignment loss...")
    n_samples, d1, d2 = 64, 128, 256
    X = torch.randn(n_samples, d1)
    Y = torch.randn(n_samples, d2)
    # Add some correlation to make it interesting
    Y[:, :d1] += 0.5 * X

    fig = plot_cka_loss_surface(X, Y)
    plt.savefig(OUTPUT_DIR / "cka_loss_surface.png", dpi=150)
    plt.close()

    print("Testing cross-covariance SVD loss...")
    fig = plot_svd_loss_surface(X, Y, k=3)
    plt.savefig(OUTPUT_DIR / "svd_loss_surface.png", dpi=150)
    plt.close()

    fig = plot_svd_loss_vs_k(X, Y, k_range=range(1, 15))
    plt.savefig(OUTPUT_DIR / "svd_loss_vs_k.png", dpi=150)
    plt.close()

    print("Testing alignment comparison...")
    fig = plot_alignment_comparison(X, Y, k=3)
    plt.savefig(OUTPUT_DIR / "alignment_comparison.png", dpi=150)
    plt.close()

    print("All plots saved!")
    plt.show()
