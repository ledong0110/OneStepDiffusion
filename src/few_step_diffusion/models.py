from pathlib import Path

from diffusers import DDPMScheduler, UNet2DModel
from .config import image_size


def load_model(config, source=None):
    source = source or config["model"]["source"]
    if source in ("tiny", "scratch"):
        channels = (16, 32) if source == "tiny" else (64, 128, 128)
        model = UNet2DModel(
            sample_size=image_size(config), in_channels=3, out_channels=3,
            layers_per_block=1 if source == "tiny" else 2,
            block_out_channels=channels,
            down_block_types=("DownBlock2D",) * len(channels),
            up_block_types=("UpBlock2D",) * len(channels),
            norm_num_groups=8,
        )
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                                  prediction_type="epsilon", clip_sample=False)
    else:
        kwargs = {}
        if source == config["model"]["source"] and not Path(source).exists():
            kwargs["revision"] = config["model"]["revision"]
        safe_weights = True if Path(source).exists() else config["model"].get("use_safetensors", True)
        model = UNet2DModel.from_pretrained(source, use_safetensors=safe_weights, **kwargs)
        scheduler = DDPMScheduler.from_pretrained(source, **kwargs)
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("Only epsilon-prediction checkpoints are supported")
    if model.config.sample_size != image_size(config) or model.config.in_channels != 3:
        raise ValueError(f"Model must match RGB {image_size(config)}x{image_size(config)} data")
    return model, scheduler
