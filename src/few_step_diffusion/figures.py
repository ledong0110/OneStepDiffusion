"""CVPR-style qualitative figures from actual, fixed-noise evaluation samples."""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

METHOD_LABELS = {
    "teacher": "Teacher (DDIM)", "baseline": "Progressive distillation",
    "edge_constant": "PD + constant edge", "proposed": "Ours: noise-aware edge",
    "swiftbrush_vsd": "SwiftBrush-inspired VSD",
    "swiftbrush_source": "SwiftBrush recipe (full G)",
    "swiftbrush_lora": "SwiftBrush-inspired (LoRA G, source recipe)",
    "consistency": "Consistency distillation", "dmd": "DMD-inspired",
    "dmd_grouped": "DMD control",
    "dmd_perceptual": "DMD + LPIPS",
    "dmd_diversity": "DMD + diversity",
    "dmd_perceptual_diversity": "Ours: DMD + LPIPS + diversity",
    "instaflow": "InstaFlow-inspired",
    "edge_weight_low": "Noise-aware edge, 0.03", "edge_weight_high": "Noise-aware edge, 0.30",
}


def read_samples(record):
    size = record.get("image_size", 32)
    encoded = record.get("sample_images_png_base64", [])
    if encoded:
        samples = []
        for value in encoded[:8]:
            if not isinstance(value, str) or len(value) > 2097152:
                raise ValueError("Invalid portable sample size")
            image = Image.open(io.BytesIO(base64.b64decode(value, validate=True)))
            if image.size != (size, size) or image.format != "PNG":
                raise ValueError(f"Qualitative samples must be {size}x{size} PNGs matching metadata")
            samples.append(np.asarray(image.convert("RGB")))
        return samples
    # Compatibility with earlier results: recover the first row of the local
    # torchvision grid. New result JSONs need no external image files.
    path = Path(record.get("sample_grid", ""))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.is_file():
        return []
    with Image.open(path) as image:
        stride = size+2
        if image.size[0] != 8*stride+2 or image.size[1] < size+4:
            return []
        image = image.convert("RGB")
        return [np.asarray(image.crop((2+stride*i, 2, size+2+stride*i, size+2))) for i in range(8)]


def make_sample_figure(records, output_dir, steps=1, seed=42, teacher_preview=()):
    import matplotlib
    matplotlib.use("Agg")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    eligible = [r for r in records if r["status"] == "complete"]
    selected = []
    teachers = [r for r in eligible if r["method"] == "teacher"]
    if teachers:
        reference = max(teachers, key=lambda r: r["sampling_steps"])
        selected.append(reference)
        selected.extend(r for r in teachers if r["sampling_steps"] == steps and r is not reference)
    for method in ("baseline", "edge_constant", "proposed", "swiftbrush_vsd", "consistency", "dmd", "instaflow",
                   "dmd_grouped", "dmd_perceptual", "dmd_diversity", "dmd_perceptual_diversity"):
        selected.extend(r for r in eligible if r["method"] == method and
                        r["sampling_steps"] == steps and r["seed"] == seed)
    rows = []
    for record in selected:
        images = read_samples(record)
        if images:
            rows.append((record, images))
    preview = False
    if not rows and teacher_preview:
        for record in teacher_preview:
            if record.get("status") == "preview" and record.get("method") == "teacher":
                images = read_samples(record)
                if images:
                    rows.append((record, images))
        preview = bool(rows)
    if not rows:
        (out / "qualitative_figure.tex").write_text(
            "% No completed evaluation sample images available; no smoke images are inserted.\n")
        stale = out / "samples_comparison.pdf"
        stale.unlink(missing_ok=True)
        return []
    sd = rows[0][0].get("dataset") == "COCO-2017"
    if sd and any(r.get("sample_prompts") != rows[0][0].get("sample_prompts") for r, _ in rows):
        raise ValueError("Text-to-image sample rows must share the same prompts in the same order")
    columns = min(4 if sd else 8, min(len(images) for _, images in rows))
    if sd:
        _draw_prompt_matrix(rows, columns, out/"samples_comparison.pdf")
    else:
        _draw_pixel_samples(rows, columns, out)
    caption = (
        f"Qualitative {rows[0][0].get('dataset', 'CIFAR-10')} generation. " +
        ("Rows share text prompts and initial noise; columns are methods. " if sd else
         "Columns share initial noise in the evaluation sequence. ") +
        "Method labels state actual denoiser evaluations (NFE). The longest teacher trajectory is a "
        "reference and has a different inference budget. Samples are the first consecutive "
        "evaluation outputs, without quality-based selection."
    )
    if any(r["method"] != "teacher" for r, _ in rows):
        caption += f" Student rows use training seed {seed}."
    if any(r["method"] == "swiftbrush_vsd" for r, _ in rows):
        caption += (" SwiftBrush-inspired VSD uses LoRA student and auxiliary adapters." if sd else
                    " SwiftBrush-inspired VSD is an unconditional adaptation with a full auxiliary U-Net.")
    if preview:
        caption = (
            f"{'Stable Diffusion 1.5' if sd else rows[0][0].get('dataset', 'CIFAR-10')} teacher preview at "
            f"{rows[0][0].get('image_size', 32)} by {rows[0][0].get('image_size', 32)} pixels. "
            + ("Fixed noise (seed 12345) and predefined prompts are used. "
               "The guided teacher uses two denoiser branches per step. " if sd else
               "Columns share noise (seed 12345); rows show DDIM evaluations. ") +
            "Consecutive teacher samples illustrate the setup; completed student comparisons "
            "will replace this preview."
        )
    from .reporting import tex_escape
    (out / "qualitative_figure.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n"
        "\\includegraphics[width=\\textwidth,height=2.6in,keepaspectratio]{generated/samples_comparison.pdf}\n"
        f"\\caption{{{tex_escape(caption)}}}\n"
        "\\label{fig:samples}\n\\end{figure*}\n")
    return [{"method": r["method"], "steps": r["sampling_steps"], "seed": r["seed"],
             "source_file": r.get("source_file"), "sample_indices": list(range(columns))}
            for r, _ in rows]


SHORT_LABELS = {"baseline": "PD", "edge_constant": "Constant edge", "proposed": "Ours",
                "swiftbrush_vsd": "VSD*", "consistency": "CD*", "dmd": "DMD*",
                "instaflow": "InstaFlow*", "edge_weight_low": r"$\lambda=0.03$",
                "edge_weight_high": r"$\lambda=0.30$", "teacher": "SD teacher"}


def _draw_prompt_matrix(rows, count, path, width=6.875):
    import textwrap
    import matplotlib.pyplot as plt

    n = len(rows)
    label_width = 1.2 if n > 2 else 1.3
    side = min(.9, (width-label_width-.04*n)/n)
    width = label_width+n*(side+.04)
    height = .37+count*(side+.035)
    with plt.rc_context({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(width, height))
        grid = fig.add_gridspec(count+1, n+1, width_ratios=[label_width]+[side]*n,
                               height_ratios=[.32]+[side]*count, left=.002, right=.998,
                               top=.997, bottom=.002, wspace=.055, hspace=.055)
        for col, (r, images) in enumerate(rows):
            header = fig.add_subplot(grid[0, col+1])
            header.axis("off")
            name = SHORT_LABELS.get(r["method"], r["method"])
            header.text(.5, .7, name, ha="center", va="center", fontsize=7,
                        fontweight="bold" if r["method"] == "proposed" else "normal")
            header.text(.5, .2, f"{r['sampling_steps']} step / {r['nfe']} NFE", ha="center",
                        va="center", fontsize=5.5, color="#444444")
            for row in range(count):
                if col == 0:
                    label = fig.add_subplot(grid[row+1, 0])
                    label.axis("off")
                    prompt = (r.get("sample_prompts") or r.get("sample_labels") or
                              [f"Prompt {i+1}" for i in range(count)])[row]
                    label.text(0, .5, textwrap.fill(prompt, 24), ha="left", va="center",
                               fontsize=6.5, clip_on=True)
                ax = fig.add_subplot(grid[row+1, col+1])
                ax.imshow(images[row], interpolation="nearest")
                ax.axis("off")
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.01)
        plt.close(fig)


def make_qualitative_panels(records, out, seed=42):
    """Create a paper comparison and two controlled grids like the supplied examples."""
    from .qualitative import CORE_METHODS, EDGE_METHODS, ANIMALS, ACTIONS
    from .reporting import tex_escape
    import matplotlib.pyplot as plt

    out = Path(out)
    extra = out/"extra_qualitative.tex"
    extra.write_text("% Controlled qualitative panels are pending.\n")
    for name in ("ablation_visual.pdf", "prompt_control.pdf", "samples_comparison_full.pdf"):
        (out/name).unlink(missing_ok=True)
    eligible = [r for r in records if r["seed"] == (0 if r["method"] == "teacher" else seed)]
    if not eligible:
        return []

    def pixels(item):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(item["png_base64"]))).convert("RGB"))

    comparison = []
    for method, steps in [(m, 1) for m in CORE_METHODS]+[("teacher", 64), ("teacher", 1)]:
        match = next((r for r in eligible if (r["method"], r["sampling_steps"]) == (method, steps)), None)
        if match:
            items = match["panels"]["comparison"]
            comparison.append(({**match, "sample_prompts": [i["prompt"] for i in items]}, [pixels(i) for i in items]))
    if comparison:
        _draw_prompt_matrix(comparison, 3, out/"samples_comparison.pdf")
        _draw_prompt_matrix(comparison, 6, out/"samples_comparison_full.pdf")
        caption = ("Fixed-prompt qualitative comparison. Rows share the full prompt and initial noise; "
                   "columns are models. Every trained student uses one step / one NFE. SD teacher "
                   "references use the labelled budgets and CFG 7.5. Asterisks denote our LoRA "
                   "adaptations, rather than released checkpoints. The first three of six predefined "
                   "prompts appear here; all six are exported in the qualitative supplement. "
                   "No quality-based sample selection is used.")
        (out/"qualitative_figure.tex").write_text(
            "\\begin{figure*}[t]\n\\centering\n"
            "\\includegraphics[width=\\textwidth,height=2.6in,keepaspectratio]{generated/samples_comparison.pdf}\n"
            f"\\caption{{{tex_escape(caption)}}}\n\\label{{fig:samples}}\n\\end{{figure*}}\n")
    edges = {r["method"]: r for r in eligible if r["method"] in EDGE_METHODS}
    if all(m in edges and "ablation" in edges[m]["panels"] for m in EDGE_METHODS):
        with plt.rc_context({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42}):
            fig = plt.figure(figsize=(3.3, .94))
            grid = fig.add_gridspec(3, 5, height_ratios=[.25, 1, 1], left=.005, right=.995,
                                   top=.995, bottom=.005, wspace=.045, hspace=.05)
            titles = ("No edge", "Constant", r"$\lambda=.03$", "Ours (.10)", r"$\lambda=.30$")
            for col, method in enumerate(EDGE_METHODS):
                header = fig.add_subplot(grid[0, col])
                header.axis("off")
                header.text(.5, .5, titles[col], ha="center", va="center", fontsize=6.5)
                items = edges[method]["panels"]["ablation"]
                for row in range(2):
                    ax = fig.add_subplot(grid[row+1, col])
                    ax.axis("off")
                    ax.imshow(np.concatenate([pixels(items[2*row]), pixels(items[2*row+1])], axis=1))
            fig.savefig(out/"ablation_visual.pdf", dpi=300, bbox_inches="tight", pad_inches=.01)
            plt.close(fig)
        extra.write_text("\\begin{figure}[t]\n\\centering\n"
                         "\\includegraphics[width=\\linewidth]{generated/ablation_visual.pdf}\n"
                         "\\caption{Visual edge-loss ablation. Rows use fixed train-painting and tiger-city "
                         "prompts, respectively; each cell contains two fixed noise seeds shared across "
                         "all five variants. All use one step / one NFE and the same training budget. "
                         "Only edge weight and gate vary.}\\label{fig:visual-ablation}\n\\end{figure}\n")
    proposed = next((r for r in eligible if r["method"] == "proposed"), None)
    if proposed and "control" in proposed["panels"]:
        with plt.rc_context({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42}):
            fig = plt.figure(figsize=(3.3, 2.4))
            grid = fig.add_gridspec(4, 5, width_ratios=[.38, 1, 1, 1, 1], height_ratios=[.18, 1, 1, 1],
                                   left=.002, right=.998, top=.995, bottom=.005, wspace=.025, hspace=.025)
            items = proposed["panels"]["control"]
            for col, animal in enumerate(ANIMALS):
                ax = fig.add_subplot(grid[0, col+1])
                ax.axis("off")
                ax.text(.5, .5, animal.capitalize(), ha="center", va="center", fontsize=7)
                for row, action in enumerate(ACTIONS):
                    if col == 0:
                        ax = fig.add_subplot(grid[row+1, 0])
                        ax.axis("off")
                        ax.text(.5, .5, ("Book", "Latte", "Pizza")[row], rotation=90, ha="center", va="center")
                    ax = fig.add_subplot(grid[row+1, col+1])
                    ax.axis("off")
                    ax.imshow(pixels(items[row*len(ANIMALS)+col]))
            fig.savefig(out/"prompt_control.pdf", dpi=300, bbox_inches="tight", pad_inches=.01)
            plt.close(fig)
        with extra.open("a") as handle:
            handle.write("\\begin{figure}[t]\n\\centering\n"
                         "\\includegraphics[width=\\linewidth]{generated/prompt_control.pdf}\n"
                         "\\caption{Compositional prompt control with our one-step student. "
                         "Template: `A DSLR photograph of a [animal] [action], sitting at an outdoor "
                         "cafe table.' Animal varies across columns; action varies down rows. "
                         "All 12 cells use identical initial noise. These are separate prompt edits, "
                         "not continuous embedding interpolation.}\\label{fig:prompt-control}\n\\end{figure}\n")
    return [{"method": r["method"], "steps": r["sampling_steps"], "seed": r["seed"],
             "source_file": r["source_file"], "panels": list(r["panels"])} for r in eligible]


def _draw_scene_samples(rows, columns, out):
    """Put labels above image strips, with no reserved label column on the left."""
    import matplotlib.pyplot as plt

    panels = 2 if len(rows) > 2 else 1
    panel_rows = (len(rows)+panels-1)//panels
    width = 6.875
    image_side = (width-.12*(panels-1))/panels/columns
    height = panel_rows*(image_side+.16+.01)+.16
    with plt.rc_context({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(width, height))
        outer = fig.add_gridspec(1, panels, left=.003, right=.997, top=.995, bottom=.003,
                                wspace=.035)
        for panel in range(panels):
            heights = [value for row in range(panel_rows)
                       for value in (.16, .17 if row == 0 else .01, image_side)]
            grid = outer[0, panel].subgridspec(3*panel_rows, columns,
                                              height_ratios=heights, wspace=.025, hspace=.015)
            for row in range(panel_rows):
                index = panel*panel_rows+row
                if index >= len(rows):
                    continue
                record, images = rows[index]
                label = fig.add_subplot(grid[3*row, :])
                label.axis("off")
                name = METHOD_LABELS.get(record["method"], record["method"])
                label.text(0, .5, name, ha="left", va="center", fontsize=8,
                           fontweight="bold" if record["method"] == "proposed" else "normal")
                badge = f"{record['sampling_steps']} steps / {record.get('nfe', record['sampling_steps'])} NFE"
                label.text(1, .5, badge, ha="right", va="center", fontsize=7, color="#333333",
                           bbox=dict(boxstyle="round,pad=.2", facecolor="#edf0f3", edgecolor="none"))
                for column in range(columns):
                    if row == 0:
                        header = fig.add_subplot(grid[3*row+1, column])
                        header.axis("off")
                        title = (record["sample_labels"][column] if record.get("sample_labels")
                                 else f"Prompt {column+1}")
                        header.text(.5, .5, title, ha="center", va="center",
                                    fontsize=8 if panels == 1 else 7)
                    ax = fig.add_subplot(grid[3*row+2, column])
                    ax.imshow(images[column], interpolation="nearest")
                    ax.axis("off")
        fig.savefig(out / "samples_comparison.pdf", dpi=300, bbox_inches="tight", pad_inches=.015)
        plt.close(fig)


def _draw_pixel_samples(rows, columns, out):
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(6.875, .54*len(rows)+.28))
        grid = fig.add_gridspec(len(rows), columns+1, width_ratios=[2.5]+[1]*columns,
                               left=0.01, right=0.995, top=1-0.24/fig.get_figheight(), bottom=0.01,
                               wspace=0.03, hspace=0.06)
        for row, (record, images) in enumerate(rows):
            label = fig.add_subplot(grid[row, 0])
            label.axis("off")
            name = METHOD_LABELS.get(record["method"], record["method"])
            label.text(0.94, 0.66, name, ha="right", va="center",
                       fontsize=7.5, fontweight="bold" if record["method"] == "proposed" else "normal")
            badge = f"{record['sampling_steps']} NFE"
            label.text(0.94, 0.26, badge, ha="right", va="center",
                       fontsize=7, color="#333333", bbox=dict(boxstyle="round,pad=0.25",
                       facecolor="#edf0f3", edgecolor="none"))
            for column in range(columns):
                ax = fig.add_subplot(grid[row, column+1])
                ax.imshow(images[column], interpolation="nearest")
                ax.axis("off")
                if row == 0:
                    ax.set_title(f"$z_{{{column+1}}}$", fontsize=8, pad=3)
        fig.savefig(out / "samples_comparison.pdf", dpi=300)
        plt.close(fig)
