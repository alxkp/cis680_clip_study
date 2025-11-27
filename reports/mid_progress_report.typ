#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: "Mid Progress Report",
  authors: (
    (
    name: "Alexander Kyimpopkin",
    email: "alxkp@seas.upenn.edu",
    organization: "University of Pennsylvania",
    ),
  ),
)

= Mid Progress Report vs initial work

Our primary work in this phase has been to refactor the initial work from a notebook to a codebase.  The notebook format increasingly became a bottleneck to running ablations or other experiments, so we spent our time refactoring the code.  we give some summary statistics below.

#image("../out/cloc.png")

= Loss formulation

We also did a brief study on loss formulation.  We wanted to produce a loss which had the characteristics of producing a small number of high magnitude clusters, towards finding a simple and separable basis for clip losses.

We implemented:
+ Centered Kernel Alignment (CKA) loss
+ Cross-Covariance SVD loss
+ Spectral Gap + Top-k Eigenvalues loss


We give their plots below:

#figure(
  image("../out/cka_loss_surface.png"),
  caption: "Loss surface plots for CKA",
)

Our motivation for the CKA is to ensure that across many different clusters, the different representations are aligned, and is the case across the entire matrix, rather than simply in the `<CLS>` `<EOS>` tokens, which compress to an artifically tight bottleneck.

We can see that as we raise the noise level in synthetic perturbation, the centered kernel alignment decreases, indicating that we are becoming less and less aligned, and that this scales well at a matrix level.

#figure(
  image("../out/svd_loss_surface.png"),
  caption: "Loss surface plots for SVD",
)

Here, our core insight is that we want to use the singular values of the cross covariance matrix to compute the alignment.  Focusing on the rightmost plot, we can see the nonlinear dropoff moving from the first 3 eigenvalues to the further ones, suggesting that we can meaningfully cluster the signal into a small number of high dimensional directions - this gives us confidence that we can capture nonlinear information content in the latents, given that we are able to see this in the synthetic case here. 

#figure(
  image("../out/alignment_comparison.png"),
  caption: "Loss surface comparision for Alignment",
)

We use this comparsion to see how we can get banding structure on an off diagonal in the cross covariance matrix - ie that we can capture asymetric information relationships.


= Future directions
We will train CLIP models over all three different losses and then benchmark them on Winoground VQA for spatial reasonsing tasks.  We believe that switching to the full matrix of interplays will let us separate the spatial relationships in the patches, rather than compressing all the different quadratically large number of patch interactions into a single vector.  We are optimstic that this will show improved performance on Winoground VQA, and we will benchmark this vs existing methods.  If we have extra time, we will also test this over clip variants such as SIGLIP, to ensure that this recipe is not speciic to only CLIP, and generalizes to all CLIP type models.

We are excited to report our future results!