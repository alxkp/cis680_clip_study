from dataclasses import asdict

import draccus

from engineered_latents.benchmarking import print_benchmark_summary, run_all_benchmarks
from engineered_latents.config import ClipConfig, MainConfig
from engineered_latents.context import aim_context, clip_context


@draccus.wrap()
def main(cfg: MainConfig):
    assert isinstance(cfg.model, ClipConfig)

    with aim_context(repo=".", experiment=cfg.experiment) as run:
        run["config"] = asdict(cfg)

        with clip_context(cfg.model):
            results = run_all_benchmarks(
                include_winoground=True,
                include_coco=True,
                include_flickr=False,
                include_imagenet=False,
                batch_size=32,
                max_samples=None,
                track_results=True,
            )
            print_benchmark_summary(results)

if __name__ == "__main__":
    main()