"""DMD2 and f-distill components, following pinned NVlabs/FastGen SD1.5 recipes.

Reference: third_party/fastgen_reference/source.json (Apache-2.0).
The local adaptation retains the project's SD1.5 VP grid and rank-32 student.
Unlike original DMD, these objectives include adversarial training and no LPIPS.
"""

import torch
from torch import nn
from torch.nn import functional as F

SOURCE_COMMIT = "3943fd5cfb2a8266ec53e7e541e4b8d2e5b1e94d"
SOURCE_URL = "https://github.com/NVlabs/FastGen"
METHODS = ("dmd2", "f_distill")


def distribution_surrogate(generated, teacher_x0, fake_x0, weights=None):
    """Detached-target VSD; outer weights give h*g, never h squared times g."""
    if generated.shape != teacher_x0.shape or generated.shape != fake_x0.shape:
        raise ValueError("Distribution matching requires equal latent shapes")
    generated = generated.float()
    dims = tuple(range(1, generated.ndim))
    with torch.no_grad():
        normalizer = (generated-teacher_x0.float()).abs().mean(dim=dims, keepdim=True)+1e-6
        direction = (fake_x0.float()-teacher_x0.float())/normalizer
        target = generated-direction
    per_sample = .5*(generated-target).square().mean(dim=dims)
    if weights is not None:
        if weights.shape != per_sample.shape:
            raise ValueError("One detached f-divergence weight is required per sample")
        per_sample = per_sample*weights.detach()
    return per_sample.mean()


def discriminator_loss(real_logits, fake_logits):
    return F.softplus(-real_logits.float()).mean()+F.softplus(fake_logits.float()).mean()


def generator_adversarial_loss(fake_logits):
    return F.softplus(-fake_logits.float()).mean()


class DensityRatioWeights(nn.Module):
    """JS f-distill ratios normalized over the entire effective batch.

    Call once per generator update, after collecting all microbatch logits.
    Normalizing independently in singleton microbatches would erase f-distill.
    Histogram state is serialized with the checkpoint, including on resume.
    """
    def __init__(self, bins=10, lower=.1, upper=20., ema_rate=0.):
        super().__init__()
        self.register_buffer("bins", torch.ones(bins))
        self.lower, self.upper, self.ema_rate = lower, upper, ema_rate

    @torch.no_grad()
    def forward(self, logits, timesteps, *, update=True):
        if logits.ndim != 2 or timesteps.shape != (len(logits),) or len(logits) < 2:
            raise ValueError("f-distill requires a full effective batch of at least two samples")
        ratio = logits.float().mean(1).clamp(-10, 10).exp().clamp(self.lower, self.upper)
        # Local VP grid t=0..999; bins cover the full [0,1] schedule as upstream.
        index = (timesteps.float()/1000*len(self.bins)).floor().long().clamp(0, len(self.bins)-1)
        count = torch.bincount(index, minlength=len(self.bins)).float()
        sums = torch.bincount(index, weights=ratio, minlength=len(self.bins))
        valid = count > 0
        if update:
            self.bins[valid] = self.ema_rate*self.bins[valid]+(1-self.ema_rate)*sums[valid]/(count[valid]+1e-6)
        ratio = ratio/(self.bins[index]+1e-6)
        weight = 1-1/(1+ratio)
        return weight/(weight.mean()+1e-6)


class SD15Discriminator(nn.Module):
    """FastGen's bottleneck head: 8x8 -> 4x4 -> 1x1 on frozen teacher features."""
    def __init__(self, channels=1280):
        super().__init__()
        groups = max(1, channels//4) if channels <= 32 else 32
        while channels % groups:
            groups -= 1
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, 4, stride=2, padding=1),
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 4, stride=4),
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, 1, 1))

    def forward(self, features):
        if features.shape[-2:] != (8, 8):
            raise ValueError("SD1.5 discriminator expects an 8x8 teacher bottleneck")
        return self.head(features).flatten(1)


def phase_for_update(config, generator_update):
    """The same two COCO pools as the completed comparison; no validation data."""
    if generator_update < 0 or generator_update >= config["train"]["updates_per_stage"]:
        raise ValueError("Generator update is outside the configured training budget")
    switch = config["recent_baseline"]["phase1_updates"]
    return (0, generator_update) if generator_update < switch else (1, generator_update-switch)


def validate_recent(config):
    tr = config["train"]
    if tr["mode"] not in METHODS:
        return
    if config["model"].get("family") != "stable-diffusion" or config["model"]["lora_rank"] != 32:
        raise ValueError("Recent baselines require the shared SD1.5 rank-32 student")
    if "continuation" in config or "dmd_refinement" in config:
        raise ValueError("Recent baselines start from pretrained SD1.5, not an existing distilled student")
    recipe = config.get("recent_baseline", {})
    if recipe.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("Recent baselines require the pinned audited FastGen recipe")
    if (tr["updates_per_stage"] != 36000 or recipe.get("student_updates") != 36000 or
            recipe.get("auxiliary_updates") != 324000 or recipe.get("phase1_updates") != 18000):
        raise ValueError("Recent baseline comparison requires exactly 36,000 student and 324,000 auxiliary updates")
    if (config["model"]["guidance_scale"] != 1.75 or
            any(tr.get(k) != 1e-5 for k in ("learning_rate", "fake_learning_rate", "discriminator_learning_rate")) or
            tr["warmup_steps"] != 0 or tr["weight_decay"] != .01):
        raise ValueError("Recent baseline optimizer or guidance differs from the audited SD1.5 recipe")
    if (not recipe.get("full_fake_scorer") or not recipe.get("real_training_images_for_discriminator") or
            recipe.get("student_initialization") != "pretrained-sd15"):
        raise ValueError("Recent baselines require real images, full fake scorer and pretrained initialization")
    if tr["stages"] != [1] or tr["edge_weight"] != 0 or tr.get("dmd_regression_weight", 0) != 0:
        raise ValueError("DMD2/f-distill use one step, no edge term and no paired regression")
    if tr.get("student_update_frequency") != 10 or tr.get("gan_weight") != .001:
        raise ValueError("Use the audited SD1.5 9:1 auxiliary/student schedule and GAN weight")
    if recipe.get("phase1_updates", 0) < 1 or recipe["phase1_updates"] >= tr["updates_per_stage"]:
        raise ValueError("Both disjoint COCO training phases need student updates")
    if recipe.get("phase1_limit") != 50000 or config["data"].get("train_offset", 0) != 0 or config["data"]["train_limit"] != 118287:
        raise ValueError("Recent baselines must use the existing 50,000 + 68,287 training pools")
    if config["data"]["batch_size"]*tr["grad_accumulation"] != 16:
        raise ValueError("Recent baselines match effective generator batch 16")
    if tr.get("checkpoint_eval_samples", 0) < 8:
        raise ValueError("Saved checkpoints need a native evaluation preview")
