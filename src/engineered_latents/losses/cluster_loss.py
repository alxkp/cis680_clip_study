import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from engineered_latents.types import ArrayLike


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
    alphas = np.linspace(*alpha_range, resolution)
    betas = np.linspace(*beta_range, resolution)

    Alpha, Beta = np.meshgrid(alphas, betas)
    Loss = np.zeros_like(Alpha)

    for i in range(resolution):
        for j in range(resolution):
            Loss[i, j] = ncut_cluster_loss(eigenvalues, k, Alpha[i, j], Beta[i, j])

    fig = plt.figure(figsize=(15, 5))

    # 3D surface
    ax1 = fig.add_subplot(131, projection="3d")
    surf = ax1.plot_surface(
        Alpha, Beta, Loss, cmap=cm.viridis, alpha=0.8, edgecolor="none"
    )
    ax1.set_xlabel("Alpha (gap weight)")
    ax1.set_ylabel("Beta (cluster weight)")
    ax1.set_zlabel("Loss")
    ax1.set_title("Loss Surface")
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # Contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contourf(Alpha, Beta, Loss, levels=20, cmap=cm.viridis)
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


if __name__ == "__main__":
    # Example usage with your eigenvalues:
    eigenvalues = torch.randn(100)
    fig = plot_loss_surface(eigenvalues, k=3)

    eigenvalues = torch.randn(100)
    fig = plot_loss_vs_k(eigenvalues, k_range=range(2, 15), alpha=1.0, beta=0.5)
    plt.show()
