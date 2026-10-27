"""Fixed qualitative prompts; authentic model outputs in portable JSON files."""

import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import torch
from tqdm.auto import tqdm

from .config import load_config
from .diffusion import make_grid, nested_grid
from .runtime import atomic_json, device_for, seed_everything, sha256
from .sd_models import SDBundle, generate_sd, load_adapter, restore_export

COMPARISON_PROMPTS = [
    "A photograph of a scoop of vanilla ice cream in a gold bowl, soft golden bokeh in the background.",
    "A detailed photograph of a fox astronaut wearing a spacesuit, planet Earth in the background.",
    "A photograph of a poodle wearing a red baseball cap and reading an open dictionary.",
    "A sharp photograph of a tabby cat jumping over a wooden garden fence in daylight.",
    "A photograph of a blue vintage sports car parked in front of a red brick wall.",
    "A photograph of an astronaut riding a brown horse on a rocky moon landscape.",
]
ABLATION_PROMPTS = ["An oil painting of a steam train crossing a stone bridge in a green forest.",
                    "A detailed photograph of a tiger walking down a city street."]
ANIMALS = ("owl", "raccoon", "cat", "panda")
ACTIONS = ("reading an open book", "drinking a latte", "eating pizza")
CORE_METHODS = ("baseline", "consistency", "swiftbrush_vsd", "dmd", "instaflow", "proposed")
EDGE_METHODS = ("baseline", "edge_constant", "edge_weight_low", "proposed", "edge_weight_high")


def panel_specs(seed=12345):
    return {
        "comparison": [{"prompt": p, "noise_seed": seed+i} for i, p in enumerate(COMPARISON_PROMPTS)],
        "ablation": [{"prompt": p, "noise_seed": seed+100+j}
                     for p in ABLATION_PROMPTS for j in range(2)],
        "control": [{"prompt": f"A DSLR photograph of a {animal} {action}, sitting at an outdoor cafe table.",
                     "noise_seed": seed+200, "animal": animal, "action": action}
                    for action in ACTIONS for animal in ANIMALS],
    }


def protocol_hash(seed):
    return hashlib.sha256(json.dumps(panel_specs(seed), sort_keys=True).encode()).hexdigest()


def generate_qualitative(run_root, output_dir, methods=CORE_METHODS+EDGE_METHODS, seed=42,
                         eval_seed=12345, resume=False, skip_teacher=False):
    run_root, output_dir = Path(run_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(run_root/"plans"/f"baseline_seed{seed}.yaml")
    config["seed"] = seed
    seed_everything(seed)
    device = device_for(config)
    bundle = SDBundle(config, device)
    root = make_grid(len(bundle.alphas), config["train"]["initial_steps"], device)
    specs = panel_specs(eval_seed)
    jobs = [(m, 1) for m in dict.fromkeys(methods)] + ([] if skip_teacher else [("teacher", 64), ("teacher", 1)])
    has_adapter = False
    for method, steps in jobs:
        checkpoint = run_root/f"{method}_seed{seed}"/f"stage_{steps}"
        model_hash = (config["model"]["source"]+"@"+config["model"]["revision"] if method == "teacher"
                      else sha256(checkpoint/"adapter.safetensors"))
        path = output_dir/f"{method}_{steps}steps.qualitative.json"
        identity = {"method": method, "sampling_steps": steps, "model_identity": model_hash,
                    "seed": 0 if method == "teacher" else seed, "eval_seed": eval_seed,
                    "protocol_hash": protocol_hash(eval_seed), "precision": config["precision"],
                    "teacher_identity": config["model"]["source"]+"@"+config["model"]["revision"],
                    "lora_rank": config["model"]["lora_rank"]}
        if path.exists() and resume:
            previous = json.loads(path.read_text())
            if any(previous.get(k) != v for k, v in identity.items()):
                raise ValueError(f"Existing qualitative samples have different settings: {path}")
            print(f"Reuse qualitative samples: {path}", flush=True)
            continue
        if path.exists():
            raise FileExistsError(f"Qualitative samples already exist: {path}; use --resume")
        if method == "teacher":
            if has_adapter:
                bundle.unet.disable_adapters()
            bundle.unet._fsd_prediction_type = "epsilon"
            metadata, guidance = {}, config["model"]["guidance_scale"]
            grid = nested_grid(root, steps)
        else:
            if not has_adapter:
                metadata = restore_export(bundle, checkpoint)
                has_adapter = True
            else:
                from safetensors.torch import load_file
                metadata = json.loads((checkpoint/"metadata.json").read_text())
                for key, expected in (("model_source", config["model"]["source"]),
                                      ("model_revision", config["model"]["revision"]),
                                      ("lora_rank", config["model"]["lora_rank"]),
                                      ("seed", seed), ("stage_steps", 1)):
                    if metadata.get(key) != expected:
                        raise ValueError(f"Qualitative checkpoint {key} mismatch: {checkpoint}")
                load_adapter(bundle.unet, load_file(str(checkpoint/"adapter.safetensors"), device=str(device)))
            bundle.unet._fsd_prediction_type = metadata.get("student_prediction_type", "epsilon")
            if metadata.get("student_prediction_type") == "consistency-x0":
                bundle.unet._fsd_consistency_sigma_data = metadata["consistency_sigma_data"]
            guidance, grid = 1, torch.tensor(metadata["timesteps"], device=device)
        bundle.unet.eval()
        panels = {}
        kinds = ["comparison"] + (["ablation"] if method in EDGE_METHODS else [])
        if method == "proposed":
            kinds.append("control")
        for kind in kinds:
            panels[kind] = []
            for item in tqdm(specs[kind], desc=f"{method} {steps}-step {kind}"):
                noise = torch.randn((1, 4, 64, 64), generator=torch.Generator().manual_seed(item["noise_seed"]))
                pixels = generate_sd(bundle, bundle.unet, grid, [item["prompt"]], eval_seed,
                                     guidance, noise=noise.to(device))
                if not torch.isfinite(pixels).all():
                    raise FloatingPointError("Nonfinite qualitative image")
                array = ((pixels[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).cpu().numpy()
                buffer = io.BytesIO()
                Image.fromarray(array).save(buffer, format="PNG")
                panels[kind].append({**item, "png_base64": base64.b64encode(buffer.getvalue()).decode("ascii")})
        atomic_json(path, {"schema_version": 1, "status": "qualitative", "image_size": 512,
                    **identity, "nfe": steps*(2 if method == "teacher" else 1),
                    "training_updates": metadata.get("global_step", 0), "guidance_scale": guidance,
                    "panels": panels, "adaptation": metadata.get("adaptation", "Pretrained SD1.5 teacher")})
        print(f"Qualitative samples: {path}", flush=True)


def load_qualitative(folder, records=()):
    """Reject mixed protocols, changed prompts/noise, and mismatched checkpoints."""
    imported, seen = [], set()
    metrics = {(r["method"], r["seed"], r["sampling_steps"]): r for r in records}
    shared = set()
    for path in sorted(Path(folder).rglob("*.json")):
        r = json.loads(path.read_text())
        if r.get("status") != "qualitative":
            continue
        if r.get("image_size") != 512:
            raise ValueError(f"Invalid qualitative record: {path}")
        if r["protocol_hash"] != protocol_hash(r["eval_seed"]):
            raise ValueError("Qualitative prompt protocol changed")
        key = (r["method"], r["seed"], r["sampling_steps"])
        if key in seen:
            raise ValueError("Duplicate qualitative method/seed/steps")
        seen.add(key)
        if r["method"] != "teacher" and (r["sampling_steps"] != 1 or r["nfe"] != 1):
            raise ValueError("Qualitative students must have the matched 1-step / 1-NFE budget")
        if r["method"] == "teacher" and r["nfe"] != 2*r["sampling_steps"]:
            raise ValueError("Qualitative teacher must count both CFG branches")
        reference = metrics.get(key)
        if reference and (r["model_identity"] != reference["model_identity"] or
                          r["training_updates"] != reference["training_updates"]):
            raise ValueError("Qualitative checkpoint differs from the quantitative evaluation")
        shared.add((r["protocol_hash"], r["teacher_identity"], r["precision"], r["lora_rank"]))
        if not r["panels"].get("comparison"):
            raise ValueError("Qualitative records require the complete comparison panel")
        specs = panel_specs(r["eval_seed"])
        for kind, items in r["panels"].items():
            if kind not in specs or [{k: v for k, v in i.items() if k != "png_base64"} for i in items] != specs[kind]:
                raise ValueError("Qualitative prompts or initial noise differ from the fixed protocol")
            for item in items:
                image = Image.open(io.BytesIO(base64.b64decode(item["png_base64"], validate=True)))
                if image.format != "PNG" or image.size != (512, 512):
                    raise ValueError("Qualitative panels require native 512x512 PNGs")
        imported.append({**r, "source_file": str(path)})
    if len(shared) > 1:
        raise ValueError("Incompatible qualitative protocols or teachers")
    return imported
