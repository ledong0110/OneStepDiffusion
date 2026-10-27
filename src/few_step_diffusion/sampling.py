import torch

from .diffusion import ddim_step
from .config import image_size
from .runtime import autocast


@torch.inference_mode()
def generate(model, alphas, grid, count, seed, config, device, noise=None):
    model.eval()
    if noise is None:
        # CPU RNG makes identical initial noise independent of the selected GPU.
        g = torch.Generator().manual_seed(seed)
        size = image_size(config)
        noise = torch.randn((count, 3, size, size), generator=g).to(device)
    x = noise.float()
    endpoints = torch.cat((grid, grid.new_tensor([-1])))
    for i, t in enumerate(grid):
        with autocast(config, device):
            epsilon = model(x, t).sample
        x = ddim_step(x, epsilon, t, endpoints[i+1], alphas)
    return x
