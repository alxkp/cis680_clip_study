import torch
from einops import rearrange
from ncut_pytorch import Ncut


def compute_cuts(affinities: torch.Tensor):
    feats = rearrange(affinities, "b h w -> (b h w) 1").float()
    # TODO: look into the typing for this
    eigvec = Ncut(n_eig=20, device="cuda").fit_transform(feats)  # type: ignore - this is a valid method; maybe type hint problem?

    eigvec_3d = rearrange(eigvec, "(b h w) e -> b h w e", b=400, h=50, w=29)

    #     # Convert first 3 eigenvectors to RGB for visualization
    #     eig_rgb = torch.stack(
    #         [
    #             (eigvec_3d[..., i] - eigvec_3d[..., i].min())
    #             / (eigvec_3d[..., i].max() - eigvec_3d[..., i].min())
    #             for i in range(3)
    #         ],
    #         dim=-1,
    #     )
    #     plt.imshow(eig_rgb.cpu().numpy())
    #     plt.show()

    return eigvec_3d


def compute_word_scores(eigvec_3d: torch.Tensor):
    return eigvec_3d.mean(dim=1)
