"""Render selected native paired outputs; never synthesize or retouch their content."""
import json
import textwrap

import matplotlib.pyplot as plt
from PIL import Image

from few_step_diffusion.runtime import sha256
from few_step_diffusion.sd_data import manifest_hash


def write_paired_comparison(root, out):
    spec_path = root/"configs/assignment2_paired_samples.json"
    tex = out/"paired_detail.tex"
    if not spec_path.exists():
        tex.write_text("% No selected paired detail.\n")
        return
    spec = json.loads(spec_path.read_text())
    candidate_path = root/spec["candidate_file"]
    if sha256(candidate_path) != spec["candidate_sha256"]:
        raise ValueError("Paired candidate pool changed after selection")
    pool = json.loads(candidate_path.read_text())
    selected = json.loads((root/"configs/assignment2_selected_method.json").read_text())
    if (pool["models"][1]["model_identity"] != selected["model_identity"] or
            pool["models"][1]["result_sha256"] != selected["result_sha256"] or
            pool["prompt_hash"] != manifest_hash(pool["rows"]) or
            pool["sampling_steps"] != 1 or pool["nfe"] != 1):
        raise ValueError("Paired figure does not use the report's selected model/protocol")
    indices = spec["indices"]
    if len(indices) != 2 or len(set(indices)) != 2:
        raise ValueError("Paired figure expects two distinct selected validation prompts")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(7.1, 2.35))
        for pair, index in enumerate(indices):
            for col, (model, label) in enumerate(zip(pool["models"], ("DMD + LPIPS", "Ours"), strict=True)):
                path = root/model["paths"][index]
                if sha256(path) != model["image_sha256"][index]:
                    raise ValueError("Selected native output has changed")
                if sha256(root/model["result_file"]) != model["result_sha256"]:
                    raise ValueError("Paired figure no longer corresponds to its evaluated models")
                ax = fig.add_axes([.005+pair*.51+col*.24, .21, .232, .70])
                ax.imshow(Image.open(path).convert("RGB"))
                ax.set_axis_off()
                ax.set_title(label, fontsize=9, fontweight="bold" if col else "normal", pad=5)
            caption = pool["rows"][index]["caption"]
            fig.text(.241+pair*.51, .145, "\n".join(textwrap.wrap(caption, width=49)),
                     ha="center", va="top", fontsize=8)
        for ext in ("pdf", "png"):
            fig.savefig(out/f"paired_detail.{ext}", dpi=180, bbox_inches="tight", pad_inches=.02)
        plt.close(fig)
    tex.write_text(
        "\\begin{figure*}[t]\n\\centering\n"
        "\\includegraphics[width=.96\\textwidth]{generated/assignment2/paired_detail.pdf}\n"
        "\\caption{\\textbf{Selected local changes relative to DMD + LPIPS.} "
        "Each pair uses the same validation caption and initial noise, with one-step inference. "
        "The left pair illustrates a change in foreground subject count; the right pair shows "
        "a more consolidated board silhouette. These two examples were selected from 128 matched "
        "caption pairs to illustrate favorable local differences; they do not establish overall "
        "visual superiority. DMD has 36,000 updates and Ours has \\ASelectedUpdates{}.}\n"
        "\\label{fig:paired-detail}\n\\end{figure*}\n")
