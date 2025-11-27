import torch
from transformers import CLIPModel, CLIPProcessor

from .context import get_clip


def compute_clip_scores(
    images: torch.Tensor,
    captions: torch.Tensor,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
):
    ctx = get_clip()
    processor = ctx.processor
    device = ctx.device

    assert isinstance(processor, CLIPProcessor)
    inputs = processor(text=captions, images=images, return_tensors="pt", padding=True)  # type: ignore
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        global_clip_score = outputs.logits_per_image.item()

    return global_clip_score
