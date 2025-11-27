import draccus 
from engineered_latents.config import MainConfig

@draccus.wrap()
def main(cfg: MainConfig):
    print("Hello from project!")


