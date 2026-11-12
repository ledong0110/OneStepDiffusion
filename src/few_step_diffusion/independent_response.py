"""Independent 36k-update response study, initialized from pretrained SD1.5.

No distilled student, scorer, optimizer, EMA or RNG is inherited. The response
coefficient is fixed before training from the earlier pilot; it is not claimed
to be recalibrated at the new initialization. A matched zero-response control
uses the same training procedure, teacher banks and initialization seed.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from .config import config_fingerprint
from .runtime import atomic_json, experiment_lock, sha256
from .sd_data import CaptionStream, caption_records, manifest_hash
from .teacher_pairs import TeacherPairs

POLICY = "independent-sd15-response-36k-v1"
METHODS = ("response_scratch", "dmd_lpips_scratch")
WEIGHT = 0.2984940099461313


def validate_independent(config):
    study = config.get("independent_training")
    if study is None:
        if config.get("name") in METHODS:
            raise ValueError("Independent method needs explicit fresh-initialization provenance")
        return
    tr, data = config["train"], config["data"]
    if (study.get("policy") != POLICY or config["name"] not in METHODS or
            study.get("initialization") != "pretrained-sd15-fresh-lora" or
            any(key in config for key in ("continuation", "dmd_refinement", "dmd_study"))):
        raise ValueError("Independent training must not inherit a distilled model or optimizer")
    if (tr["mode"] != "dmd" or tr["stages"] != [1] or tr["updates_per_stage"] != 36000 or
            data.get("train_offset", 0) != 0 or data["train_limit"] != 118287 or
            data["batch_size"] != 2 or tr["grad_accumulation"] != 8 or
            study.get("phase1_updates") != 18000 or study.get("phase1_limit") != 50000 or
            tr.get("fake_updates_per_step") != 1 or
            config["model"].get("lora_rank") != 32 or config["model"].get("guidance_scale") != 7.5):
        raise ValueError("Independent study requires two 18k phases, the same COCO pools and rank-32 students")
    if (tr.get("dmd_pair_seeds") != 2 or tr.get("dmd_pair_count") != 5000 or
            tr.get("dmd_regression_type") != "lpips_vgg" or tr.get("dmd_regression_weight") != .25 or
            tr.get("dmd_diversity_weight", 0) != 0 or tr["edge_weight"] != 0 or
            tr.get("dmd_gradient_mode") or tr.get("dmd_noise_mode")):
        raise ValueError("Independent response study keeps paired LPIPS-DMD and excludes other experimental losses")
    expected = WEIGHT if config["name"] == "response_scratch" else 0.
    if tr.get("dmd_response_weight", 0) != expected:
        raise ValueError("Independent response coefficient differs from the frozen plan")
    if expected and (tr.get("dmd_response_mode") != "angular" or tr.get("dmd_response_variant") != "angular_50"):
        raise ValueError("Independent proposal uses the selected angular objective from update zero")
    if not expected and tr.get("dmd_response_mode") is not None:
        raise ValueError("The control must omit the response objective")


def phase_configs(config):
    first, second = deepcopy(config), deepcopy(config)
    first["data"].update(train_offset=0, train_limit=50000, sampling="replacement")
    second["data"].update(train_offset=50000, train_limit=68287, sampling="shuffled_epochs")
    return [first, second]


class PhasedCaptions:
    def __init__(self, config):
        self.config = config
        self.streams = [CaptionStream(c, images=False) for c in phase_configs(config)]
        self.records = self.streams[0].records+self.streams[1].records
        self.phase, self.step_offset = None, 0
        if manifest_hash(self.records) != manifest_hash(caption_records(config)):
            raise ValueError("Independent data phases differ from the declared training pool")

    def select(self, step):
        switch = self.config["independent_training"]["phase1_updates"]
        self.phase = int(step >= switch)
        self.step_offset = switch*self.config["train"]["grad_accumulation"] if self.phase else 0

    def batch(self, microstep, device):
        if self.phase is None or microstep < self.step_offset:
            raise ValueError("Training phase must be selected before drawing a batch")
        return self.streams[self.phase].batch(microstep-self.step_offset, device)


class PhasedPairs:
    def __init__(self, config, bundle, grid, captions):
        self.config, self.bundle, self.grid, self.captions = config, bundle, grid, captions
        self.current, self.phase = None, None
        self.bank_metadata = {}

    def select(self, step):
        self.captions.select(step)
        phase = self.captions.phase
        if phase == self.phase:
            return
        cfg = self.captions.streams[phase].config
        folder = self.config["independent_training"]["pair_cache_dirs"][phase]
        with experiment_lock(folder):
            self.current = TeacherPairs(cfg, self.bundle, self.grid, folder, self.captions.streams[phase].records)
        self.bank_metadata[phase] = self.current.metadata
        self.phase = phase

    @property
    def metadata(self):
        return self.current.metadata

    @property
    def rows(self):
        return self.current.rows

    def batch(self, microstep, device):
        return self.current.batch(microstep-self.captions.step_offset, device)


def tensor_identity(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def bind_initialization(config, output, student_state, fake_state, optimizer, fake_optimizer, saved):
    """Bind checkpoints to fresh initialization; refuse foreign warm starts."""
    path = output/"initialization.json"
    if saved is not None:
        if not path.exists() or saved.get("independent_initialization_sha256") != sha256(path):
            raise ValueError("Independent resume is not bound to this run's fresh initialization")
        record = json.loads(path.read_text())
        if record["config_hash"] != config_fingerprint(config) or record.get("initial_updates") != 0:
            raise ValueError("Independent initialization identity mismatch")
        return
    if optimizer.state or fake_optimizer.state:
        raise ValueError("Independent training requires empty optimizer states at initialization")
    record = {"policy": POLICY, "initialization": "pretrained-sd15-fresh-lora", "initial_updates": 0,
        "teacher_identity": config["model"]["source"]+"@"+config["model"]["revision"],
        "student_adapter_identity": tensor_identity(student_state), "fake_adapter_identity": tensor_identity(fake_state),
        "inherited_distillation_checkpoint": None, "optimizer_state_entries": [0, 0],
        "config_hash": config_fingerprint(config), "seed": config["seed"], "lora_rank": 32,
        "response_enabled_from_update": 0 if config["name"] == "response_scratch" else None}
    reference = output.parent/"response_scratch_seed42/initialization.json"
    if config["name"] == "dmd_lpips_scratch" and reference.exists():
        other = json.loads(reference.read_text())
        for key in ("student_adapter_identity", "fake_adapter_identity", "teacher_identity", "seed"):
            if record[key] != other[key]:
                raise ValueError(f"Control initialization differs from proposal: {key}")
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("Refusing to overwrite a different initialization record")
    atomic_json(path, record)


def independent_metadata(config, output, data, pairs, step, auxiliary):
    phases = []
    bank_meta = []
    for i, stream in enumerate(data.streams):
        phases.append({"count": len(stream.records), "manifest_hash": manifest_hash(stream.records),
            "updates": 18000, "sampling": stream.config["data"]["sampling"]})
        folder = Path(config["independent_training"]["pair_cache_dirs"][i])
        path = folder/"teacher_pairs.json"
        if path.exists():
            bank_meta.append(json.loads(path.read_text()))
    from .dmd_diversity import lpips_identity
    return {"independent_training": config["independent_training"], "training_data_phases": phases,
        "initialization": json.loads((output/"initialization.json").read_text()),
        "independent_initialization_sha256": sha256(output/"initialization.json"),
        "cumulative_training_updates": step, "cumulative_auxiliary_updates": auxiliary,
        "teacher_pair_phases": bank_meta, "cumulative_teacher_pair_count": sum(b["teacher_pair_count"] for b in bank_meta),
        "cumulative_teacher_pair_nfe": sum(b["teacher_pair_nfe"] for b in bank_meta),
        "dmd_regression_type": "lpips_vgg", "dmd_diversity_weight": 0., "dmd_pair_seeds": 2,
        "teacher_pair_unique_prompts": len(pairs.rows)//2,
        "teacher_pair_protocol_hash": pairs.metadata["protocol_hash"],
        "teacher_pair_weights_sha256": pairs.metadata["weights_sha256"],
        "perceptual_metric": lpips_identity("vgg"),
        **{k: config["train"][k] for k in ("dmd_response_mode", "dmd_response_variant", "dmd_response_weight",
                                           "dmd_response_gradient_fraction") if k in config["train"]},
        "adaptation": "Independent SD1.5 rank-32 DMD+LPIPS training; angular response loss from update zero for the proposal; no distilled weights or optimizer inherited"}


def checkpoint_preview(bundle, student, ema, grid, config, folder, step):
    """Eight fixed held-out captions, using the current EMA; no new GPU model."""
    from .sd_models import generate_sd, using_adapter
    rows = caption_records(config, "validation", 8)
    folder.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(config["eval"]["seed"])
    with using_adapter(student, ema), torch.no_grad():
        for i, row in enumerate(rows):
            noise = torch.randn((1, 4, 64, 64), generator=generator).to(bundle.device)
            pixels = generate_sd(bundle, student, grid, [row["caption"]], config["eval"]["seed"], noise=noise)
            if not torch.isfinite(pixels).all():
                raise FloatingPointError("Nonfinite independent checkpoint preview")
            save_image((pixels.clamp(-1, 1)+1)/2, folder/f"{i:02d}.png")
    atomic_json(folder/"samples.json", {"status": "qualitative", "method": config["name"],
        "global_step": step, "eval_seed": config["eval"]["seed"], "rows": rows,
        "model_identity": sha256(folder/"adapter.safetensors"), "sampling_steps": 1, "nfe": 1})


def validate_result(record):
    config = record["resolved_config"]
    validate_independent(config)
    init = record.get("initialization", {})
    if (record.get("independent_training") != config.get("independent_training") or
            init.get("initial_updates") != 0 or init.get("inherited_distillation_checkpoint") is not None or
            init.get("optimizer_state_entries") != [0, 0] or init.get("config_hash") != record["config_hash"] or
            record.get("training_updates") != 36000 or record.get("cumulative_training_updates") != 36000 or
            record.get("auxiliary_updates") != 36000 or record.get("sampling_steps") != 1 or record.get("nfe") != 1 or
            record.get("lora_rank") != 32 or record.get("effective_batch_size") != 16 or
            record.get("dmd_regression_type") != "lpips_vgg" or record.get("dmd_regression_weight") != .25 or
            record.get("dmd_response_weight", 0) != config["train"].get("dmd_response_weight", 0) or
            not record.get("independent_initialization_sha256") or len(record.get("teacher_pair_phases", [])) != 2):
        raise ValueError("Invalid independent response training result")
    phases = record.get("training_data_phases", [])
    if (len(phases) != 2 or [p.get("count") for p in phases] != [50000, 68287] or
            any(p.get("updates") != 18000 for p in phases) or
            [p.get("sampling") for p in phases] != ["replacement", "shuffled_epochs"] or
            record.get("cumulative_teacher_pair_count") != 10000 or
            record.get("cumulative_teacher_pair_nfe") != 1280000):
        raise ValueError("Independent result training pools or teacher-pair budget mismatch")


def write_independent_results(root, out):
    from .reporting import collect_results
    paths = [root/f"results/sd15/independent36k/{method}_seed42_1steps.result.json" for method in METHODS]
    found = [p for p in paths if p.exists()]
    target = out/"independent_results.tex"
    if not found:
        target.write_text("% Awaiting completed independent-training evaluations.\n")
        return
    records, _ = collect_results(root, root/"schemas/result.schema.json", result_files=found)
    rows = [r"\section{Independent initialization experiment}",
        r"This experiment trains the direction-alignment objective from the first student update, "
        r"starting with pretrained SD1.5 and newly initialized rank-32 student and scorer adapters. "
        r"No distilled weights or optimizer states are inherited. Each run receives 36,000 student "
        r"and 36,000 scorer updates: 18,000 on the first 50,000 COCO training captions and 18,000 "
        r"on the remaining 68,287. The response coefficient is fixed at 0.298494 from the earlier "
        r"pilot, rather than recalibrated at initialization. The zero-response control uses "
        r"the same initialization, schedule, data and LPIPS regression. These independently "
        r"trained models are distinct from the 36,500-update model in the main comparison.",
        r"\begin{table}[t]\centering\small\begin{tabular}{lrrr}\toprule",
        r"Independent run & FID $\downarrow$ & IS $\uparrow$ & CLIP $\uparrow$ \\", r"\midrule"]
    for r in records:
        if r["num_samples"] != 5000 or r["num_real"] != 5000:
            raise ValueError("Independent report needs the full held-out evaluation")
        label = "Direction alignment" if r["method"] == "response_scratch" else "Without response term"
        rows.append(label+" & "+" & ".join(f"{r[k]:.2f}" for k in ("fid", "inception_score_mean", "clip_score_mean"))+r" \\")
    rows.extend([r"\bottomrule\end{tabular}",
        r"\caption{Independent 36,000-update training, one step / one NFE, with 5,000 validation samples.}",
        r"\end{table}"])
    target.write_text("\n".join(rows)+"\n")
