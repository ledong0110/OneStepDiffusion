"""Frozen SD 1.5 components and small, portable trainable LoRA adapters."""

from contextlib import contextmanager
import json
from pathlib import Path

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
import torch
from transformers import CLIPTextModel, CLIPTokenizer

from .diffusion import ddim_step
from .runtime import atomic_json, autocast

PREVIEW_PROMPTS = [
    "A photograph of a mountain lake reflecting snowy peaks, clear blue sky.",
    "A photograph of a cozy living room with plants beside a sunny window.",
    "A photograph of a red vintage car parked on a quiet city street.",
    "A photograph of a lighthouse on a rocky coast at sunset.",
    "A photograph of a bowl of fresh strawberries on a wooden table.",
    "A photograph of a sailboat on turquoise water beside a tropical island.",
    "A photograph of an old stone bridge over a river in a green forest.",
    "A photograph of a golden retriever sitting in a field of flowers.",
]
PREVIEW_LABELS = ["Mountain lake", "Living room", "Vintage car", "Lighthouse",
                  "Strawberries", "Sailboat", "Stone bridge", "Dog in flowers"]


def new_unet(config, device):
    # The official SwiftBrush trainer keeps full G and scorer master weights in
    # FP32, including when its forward passes use autocast.
    source_recipe = config["train"].get("vsd_recipe") == "swiftbrush-source-v1"
    model = UNet2DConditionModel.from_pretrained(
        config["model"]["source"], subfolder="unet", revision=config["model"]["revision"],
        use_safetensors=True, torch_dtype=torch.bfloat16 if config["precision"] == "bf16"
        and device.type == "cuda" and not source_recipe else torch.float32)
    return model.to(device).eval().requires_grad_(False)


def add_lora(model, config, trainable=True, state=None):
    rank = config["model"]["lora_rank"]
    options = {"r": rank, "lora_alpha": config["model"].get("lora_alpha", rank),
               "target_modules": config["model"].get("lora_targets",
                   ["to_q", "to_k", "to_v", "to_out.0"])}
    if "lora_init" in config["model"]:
        options["init_lora_weights"] = config["model"]["lora_init"]
    elif config["train"].get("vsd_recipe") != "swiftbrush-source-v1":
        options["init_lora_weights"] = "gaussian"
    model.add_adapter(LoraConfig(**options))
    for name, parameter in model.named_parameters():
        is_adapter = "lora_" in name
        parameter.requires_grad_(trainable and is_adapter)
        if is_adapter:
            parameter.data = parameter.data.float()
    if state is not None:
        result = set_peft_model_state_dict(model, state, adapter_name="default")
        if result.unexpected_keys:
            raise ValueError(f"Unexpected adapter parameters: {result.unexpected_keys}")
        missing = [key for key in result.missing_keys if "lora_" in key]
        if missing:
            raise ValueError(f"Missing adapter parameters: {missing}")
    if trainable:
        model.train()
        if config["model"].get("gradient_checkpointing", True):
            model.enable_gradient_checkpointing()
    return model


def adapter_state(model):
    return {name: tensor.detach().float().clone() for name, tensor in
            get_peft_model_state_dict(model).items()}


def load_adapter(model, state):
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    if result.unexpected_keys or any("lora_" in key for key in result.missing_keys):
        raise ValueError("Adapter is incompatible with the SD backbone")


@torch.no_grad()
def update_adapter_ema(ema, model, decay):
    for name, value in get_peft_model_state_dict(model).items():
        ema[name].lerp_(value.detach().float(), 1-decay)


@contextmanager
def using_adapter(model, state):
    previous = adapter_state(model)
    training = model.training
    load_adapter(model, state)
    model.eval()
    try:
        yield
    finally:
        load_adapter(model, previous)
        model.train(training)


class SDBundle:
    def __init__(self, config, device):
        self.config, self.device = config, device
        source, revision = config["model"]["source"], config["model"]["revision"]
        self.tokenizer = CLIPTokenizer.from_pretrained(source, subfolder="tokenizer", revision=revision)
        self.text_encoder = CLIPTextModel.from_pretrained(
            source, subfolder="text_encoder", revision=revision, use_safetensors=True,
            torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
        # FP32 VAE prevents reduced-precision decoder artifacts; its weights stay frozen.
        self.vae = AutoencoderKL.from_pretrained(source, subfolder="vae", revision=revision,
            use_safetensors=True, torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
        self.scheduler = DDPMScheduler.from_pretrained(source, subfolder="scheduler", revision=revision)
        if self.scheduler.config.prediction_type != "epsilon":
            raise ValueError("This SD distillation backend implements epsilon prediction")
        self.unet = new_unet(config, device)
        self.alphas = self.scheduler.alphas_cumprod.to(device)

    @torch.no_grad()
    def embeddings(self, prompts):
        tokens = self.tokenizer(prompts, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt")
        return self.text_encoder(tokens.input_ids.to(self.device))[0]

    @torch.no_grad()
    def encode(self, images):
        return self.vae.encode(images.float()).latent_dist.sample() * self.vae.config.scaling_factor

    def decode(self, latents):
        # Must permit differentiation through frozen VAE for RGB edge supervision.
        return self.vae.decode(latents.float()/self.vae.config.scaling_factor).sample


def epsilon(model, x, t, conditional, unconditional, guidance, config, device):
    t = torch.as_tensor(t, device=device, dtype=torch.long)
    if t.ndim == 0:
        t = t.expand(len(x))
    if guidance > 1:
        inputs = torch.cat((x, x))
        timesteps = torch.cat((t, t))
        embeddings = torch.cat((unconditional, conditional))
    else:
        inputs, timesteps, embeddings = x, t, conditional
    with autocast(config, device):
        prediction = model(inputs, timesteps, encoder_hidden_states=embeddings).sample.float()
    if guidance > 1:
        negative, positive = prediction.chunk(2)
        prediction = negative + guidance*(positive-negative)
    return prediction


def consistency_prediction(bundle, model, x, t, conditional, sigma_data=.5):
    from .distillation_objectives import consistency_output
    if torch.all(torch.as_tensor(t, device=bundle.device) < 0):
        return x.float()
    prediction = epsilon(model, x, torch.as_tensor(t, device=bundle.device).clamp_min(0),
                         conditional, None, 1, bundle.config, bundle.device)
    return consistency_output(x, prediction, t, bundle.alphas, sigma_data)


def set_consistency_prediction(model, sigma_data):
    model._fsd_prediction_type = "consistency-x0"
    model._fsd_consistency_sigma_data = sigma_data


def flow_prediction(bundle, model, x, progress, conditional):
    from .distillation_objectives import flow_vp_inputs, epsilon_to_flow_velocity
    if getattr(model, "_fsd_prediction_type", "epsilon") == "epsilon-converted-velocity":
        inputs, t = flow_vp_inputs(x, progress, bundle.alphas)
        prediction = epsilon(model, inputs, t, conditional, None, 1, bundle.config, bundle.device)
        return epsilon_to_flow_velocity(inputs, prediction, t, bundle.alphas)
    t = ((1-progress)*999).round().long()
    return epsilon(model, x, t, conditional, None, 1, bundle.config, bundle.device)


@torch.no_grad()
def generate_sd_latents(bundle, model, grid, prompts, seed, guidance=1.0, noise=None, embeddings=None):
    config, device = bundle.config, bundle.device
    if noise is None:
        noise = torch.randn((len(prompts), 4, 64, 64), generator=torch.Generator().manual_seed(seed)).to(device)
    conditional = bundle.embeddings(prompts) if embeddings is None else embeddings[0]
    unconditional = (bundle.embeddings([""]*len(prompts)) if embeddings is None
                     else embeddings[1]) if guidance > 1 else None
    x = noise.float()
    if getattr(model, "_fsd_prediction_type", "epsilon") == "swiftbrush-source-x0":
        if len(grid) != 1 or guidance != 1:
            raise ValueError("The SwiftBrush source generator requires one step and CFG 1")
        from .sd_swiftbrush import source_original
        prediction = epsilon(model, x, grid[0], conditional, None, 1, config, device)
        return source_original(x, prediction)
    if getattr(model, "_fsd_prediction_type", "epsilon") in ("rectified-flow-velocity", "epsilon-converted-velocity"):
        if guidance != 1:
            raise ValueError("Rectified-flow students have baked guidance and require CFG 1")
        for index in range(len(grid)):
            progress = torch.full((len(x),), index/len(grid), device=device)
            x = x+flow_prediction(bundle, model, x, progress, conditional)/len(grid)
        return x
    if getattr(model, "_fsd_prediction_type", "epsilon") == "consistency-x0":
        if len(grid) != 1 or guidance != 1:
            raise ValueError("This consistency adapter uses one boundary-conditioned prediction, CFG 1")
        return consistency_prediction(bundle, model, x, grid[0], conditional,
                                      model._fsd_consistency_sigma_data)
    endpoints = torch.cat((grid, grid.new_tensor([-1])))
    for t, s in zip(endpoints[:-1], endpoints[1:], strict=True):
        prediction = epsilon(model, x, t, conditional, unconditional, guidance, config, device)
        x = ddim_step(x, prediction, t, s, bundle.alphas)
    return x


@torch.no_grad()
def generate_sd(bundle, model, grid, prompts, seed, guidance=1.0, noise=None, embeddings=None):
    return bundle.decode(generate_sd_latents(bundle, model, grid, prompts, seed, guidance,
                                            noise, embeddings))


def export_adapter(state, path, grid, config, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.cpu().contiguous() for name, tensor in state.items()},
              str(path / "adapter.safetensors"))
    # Also export a standard Diffusers LoRA, usable with pipe.load_lora_weights(path).
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    StableDiffusionPipeline.save_lora_weights(str(path),
        unet_lora_layers=convert_state_dict_to_diffusers({k: v.cpu() for k, v in state.items()}),
        safe_serialization=True)
    atomic_json(path / "metadata.json", {**metadata, "timesteps": grid.cpu().tolist(),
                "model_family": "stable-diffusion", "model_source": config["model"]["source"],
                "model_revision": config["model"]["revision"], "lora_rank": config["model"]["lora_rank"],
                "teacher_guidance_scale": config["model"]["guidance_scale"],
                "student_guidance_scale": 1.0, "image_size": 512})


def restore_export(bundle, checkpoint):
    path = Path(checkpoint)
    metadata = json.loads((path / "metadata.json").read_text())
    full = metadata.get("student_parameterization") == "full-unet"
    source_recipe = bundle.config["train"].get("vsd_recipe") == "swiftbrush-source-v1"
    expected_parameterization = (bundle.config["train"].get("student_parameterization", "full-unet")
                                 if source_recipe else "lora")
    if metadata.get("student_parameterization", "lora") != expected_parameterization:
        raise ValueError("Checkpoint student parameterization differs from the evaluation configuration")
    if (metadata.get("student_prediction_type") == "swiftbrush-source-x0") != source_recipe:
        raise ValueError("Checkpoint source recipe differs from the evaluation configuration")
    for key, expected in (("model_source", bundle.config["model"]["source"]),
                          ("model_revision", bundle.config["model"]["revision"]),
                          ("initial_steps", bundle.config["train"]["initial_steps"]),
                          ("lora_rank", 0 if full else bundle.config["model"]["lora_rank"]),
                          ("teacher_guidance_scale", bundle.config["model"]["guidance_scale"])):
        if metadata.get(key) != expected:
            raise ValueError(f"Checkpoint {key} differs from the evaluation configuration")
    from .diffusion import make_grid, nested_grid
    grid = torch.tensor(metadata["timesteps"], dtype=torch.long)
    expected_grid = nested_grid(make_grid(len(bundle.alphas), metadata["initial_steps"]), len(grid))
    if not torch.equal(grid, expected_grid):
        raise ValueError("Checkpoint inference grid differs from its nested training grid")
    if source_recipe:
        from .sd_swiftbrush import SOURCE_COMMIT, TERMINAL_ALPHA_CUMPROD
        if (metadata.get("source_commit") != SOURCE_COMMIT or
                metadata.get("terminal_alpha_cumprod") != TERMINAL_ALPHA_CUMPROD or
                metadata.get("student_prediction_type") != "swiftbrush-source-x0"):
            raise ValueError("Invalid SwiftBrush source generator metadata")
    if full:
        bundle.unet.load_state_dict(load_file(str(path/"diffusion_pytorch_model.safetensors")))
    else:
        add_lora(bundle.unet, bundle.config, trainable=False, state=load_file(str(path/"adapter.safetensors")))
    if source_recipe:
        bundle.unet._fsd_prediction_type = "swiftbrush-source-x0"
    if metadata.get("student_prediction_type") == "consistency-x0":
        set_consistency_prediction(bundle.unet, metadata["consistency_sigma_data"])
    elif metadata.get("student_prediction_type") in ("rectified-flow-velocity", "epsilon-converted-velocity"):
        bundle.unet._fsd_prediction_type = metadata["student_prediction_type"]
    return metadata


def sample_sd(config, checkpoint="teacher", steps=None, prompts=None, output="artifacts/sd15_samples.png"):
    from .config import is_stable_diffusion
    from .diffusion import make_grid, nested_grid
    from .runtime import device_for, seed_everything
    from torchvision.utils import save_image
    from tqdm.auto import tqdm
    if not is_stable_diffusion(config):
        raise ValueError("fsd sample currently uses the SD15 profile; other profiles use fsd evaluate")
    seed_everything(config["seed"])
    bundle = SDBundle(config, device_for(config))
    if checkpoint == "teacher":
        grid = nested_grid(make_grid(len(bundle.alphas), config["train"]["initial_steps"], bundle.device),
                           steps or config["train"]["initial_steps"])
        guidance = config["model"]["guidance_scale"]
    else:
        metadata = restore_export(bundle, checkpoint)
        grid = torch.tensor(metadata["timesteps"], device=bundle.device)
        if steps is not None and steps != len(grid):
            raise ValueError("Use the checkpoint's saved inference grid")
        guidance = 1.0
    prompts = prompts or PREVIEW_PROMPTS
    generator = torch.Generator().manual_seed(config["eval"]["seed"])
    images = []
    batch = config["eval"]["batch_size"]
    for start in tqdm(range(0, len(prompts), batch), desc="Text-to-image samples",
                      disable=not config.get("progress_bar", True)):
        current = prompts[start:start+batch]
        noise = torch.randn((len(current), 4, 64, 64), generator=generator).to(bundle.device)
        images.append(generate_sd(bundle, bundle.unet, grid, current, config["eval"]["seed"],
                                  guidance, noise=noise).cpu())
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image((torch.cat(images).clamp(-1, 1)+1)/2, path, nrow=min(4, len(prompts)))
    atomic_json(path.with_suffix(".prompts.json"), {"prompts": prompts, "checkpoint": str(checkpoint),
        "timesteps": grid.cpu().tolist(), "guidance_scale": guidance, "eval_seed": config["eval"]["seed"],
        "model_source": config["model"]["source"], "model_revision": config["model"]["revision"]})
    print(f"Samples: {path}", flush=True)
