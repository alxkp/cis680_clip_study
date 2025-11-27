from dataclasses import asdict

import draccus
import lovely_tensors as lt
from tqdm import trange

from engineered_latents.config import ClipConfig, MainConfig
from engineered_latents.context import aim_context, clip_context, track
from engineered_latents.data import create_loaders
from engineered_latents.train import train_step, val_step

lt.monkey_patch()  # fixing tensors to print nicely


@draccus.wrap()
def main(cfg: MainConfig):
    with aim_context(repo="engineered_latents", experiment=cfg.experiment) as run:
        run["config"] = asdict(cfg)

    train_loader, val_loader = create_loaders(cfg.dataset)

    assert isinstance(cfg.model, ClipConfig)  # HACK: assuming clip config for this

    with clip_context(cfg.model):
        for ep in trange(cfg.train.n_epochs, desc="epochs"):
            for batch in train_loader:
                loss, _grads = train_step(batch)
                track(loss.item(), name="train_loss", step=ep)
                if ep % cfg.print_every_n == 0:
                    print(f"Epoch {ep} loss: {loss.item()}")

        for batch in val_loader:
            loss, _grads = val_step(batch)
            track(loss.item(), name="val_loss", step=ep)

            if ep % cfg.print_every_n == 0:
                print(f"Epoch {ep} validation loss: {loss.item()}")

    print("Hello from project!")
