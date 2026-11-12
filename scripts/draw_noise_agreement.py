#!/usr/bin/env python3
"""Vector method schematic; colored grids are explanatory, not measured samples."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


def draw(target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    # Physical size matters: this is reduced to a seven-inch conference column span.
    fig, ax = plt.subplots(figsize=(11, 3.3))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 15), ylim=(0, 4.5))
    ax.axis("off")
    ink, muted, line = "#172C43", "#5C7083", "#AAB8C5"
    teal, coral, violet = "#117D86", "#BF6747", "#6957A4"

    def text(x, y, value, size=11, color=ink, weight="normal", ha="center", **kw):
        ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                ha=ha, va="center", **kw)

    def card(x, y, w, h, face, edge, radius=.10):
        patch = FancyBboxPatch((x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=face, edgecolor=edge, linewidth=1.1)
        ax.add_patch(patch)

    def arrow(a, b, color=muted, dashed=False, curved=0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
            color=color, linewidth=1.2, linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={curved}", shrinkA=1, shrinkB=1))

    def lock(x, y):
        ax.add_patch(Arc((x, y+.04), .11, .13, theta1=0, theta2=180, linewidth=1.1, color=muted))
        ax.add_patch(Rectangle((x-.075, y-.07), .15, .12, facecolor=muted, edgecolor="none"))

    for x, w, title, face in [(0.05, 6.0, "A   Paired score estimates", "#F5F8FB"),
                               (6.20, 4.7, "B   Proposed spatial filter", "#EEF8F6"),
                               (11.05, 3.9, "C   Matched ablations", "#F7F5FA")]:
        card(x, 1.12, w, 3.28, face, "none", .15)
        text(x+.22, 4.08, title, 13, weight="bold", ha="left")

    # One generated latent feeds both noised probes. The scorer pair is shared.
    text(.66, 3.30, "$z, c$", 13)
    arrow((.66, 3.09), (.66, 2.88))
    card(.18, 2.20, .97, .65, "#FAE8DE", coral)
    text(.665, 2.60, "Student", 10, weight="bold")
    text(.665, 2.35, "$G_\\theta$", 12, color=coral)
    arrow((1.15, 2.52), (1.58, 2.52), coral)
    card(1.60, 2.24, .60, .56, "white", line)
    text(1.90, 2.52, "$x$", 15)
    ax.plot([2.20, 2.42, 2.42], [2.52, 2.52, 3.20], color=muted, lw=1.1)
    ax.plot([2.42, 2.42], [2.52, 1.90], color=muted, lw=1.1)
    for y, sign in [(3.20, "+"), (1.90, "-")]:
        arrow((2.42, y), (2.67, y))
        card(2.69, y-.27, 1.34, .54, "white", line)
        text(3.36, y, rf"$\alpha_t x {sign} \sigma_t\epsilon$", 12)
        arrow((4.03, y), (4.35, y))
    text(3.35, 2.53, "same $t$, caption", 9, color=muted)
    card(4.37, 1.49, 1.35, 2.09, "white", line)
    lock(4.55, 3.25)
    text(5.12, 3.25, "Teacher", 9, weight="bold")
    ax.plot([4.51, 5.57], [2.95, 2.95], color="#D6DEE6", lw=1)
    text(5.04, 2.67, "Fake scorer", 9.5, weight="bold")
    text(5.04, 2.35, "shared pair", 9, color=muted)
    text(5.04, 1.91, "stop gradient", 8.5, color=muted)
    arrow((5.72, 2.54), (6.48, 2.54), teal)
    text(6.06, 2.83, "$u,v$", 12, color=teal)
    text(.25, 1.40, "Fake scorer trained separately on detached samples", 8.7, color=muted, ha="left")

    # Schematic agreement map, visibly distinct from any generated-image panel.
    card(6.52, 2.24, 1.49, .60, "white", teal)
    text(7.265, 2.54, "$m=(u+v)/2$", 10.5, color=teal)
    ax.plot([6.32, 6.32], [2.54, 3.19], color=teal, lw=1.2)
    ax.plot(6.32, 2.54, "o", color=teal, ms=3)
    arrow((6.32, 3.19), (6.70, 3.19), teal)
    text(7.27, 3.47, "Patch agreement", 10, color=teal, weight="bold")
    text(7.27, 3.19, "$r_p\\in[0,1]$", 10, color=teal)
    arrow((8.03, 3.30), (8.43, 3.30), teal)
    values = np.array([[.15, .2, .65, .75], [.1, .4, .85, .7], [.15, .3, .6, .5], [.4, .2, .1, .3]])
    for row in range(4):
        for col in range(4):
            ax.add_patch(Rectangle((8.48+.16*col, 2.98+.16*row), .145, .145,
                facecolor=plt.cm.BuGn(.15+.70*values[row, col]), edgecolor="none"))
    text(9.69, 3.30, "$w=1+r$", 11, color=teal)
    text(9.16, 3.70, "schematic weights", 8.5, color=muted)
    arrow((8.79, 2.98), (8.79, 2.79), teal)
    arrow((8.01, 2.54), (8.40, 2.54), teal)
    card(8.42, 2.22, 2.13, .62, "#D7EEE8", teal)
    text(9.485, 2.53, "$h=w\\odot m$", 13, color=teal)
    arrow((9.49, 2.21), (9.49, 1.96), teal)
    text(8.62, 1.69, "$\\widetilde m = h\\,\\|m\\|_2/\\|h\\|_2$", 13, color=teal)
    text(8.62, 1.34, "Preserve latent update norm", 8.8, color=muted)

    for y, title, note, accent in [
            (3.37, "Spatial agreement", "use the local weight map", teal),
            (2.57, "Paired mean", "remove spatial weighting", muted),
            (1.77, "Shifted map", "move weights to other patches", violet)]:
        card(11.26, y-.30, 3.47, .62, "white", "#D9D4E4")
        ax.add_patch(Rectangle((11.26, y-.20), .045, .40, facecolor=accent, edgecolor="none"))
        text(11.43, y+.09, title, 10.5, color=accent, weight="bold", ha="left")
        text(11.43, y-.14, note, 9, color=muted, ha="left")
    text(13.0, 1.30, "Same probes, initialization, budget", 8.7, color=muted)

    # Explicitly separate the inherited stabilizer from the candidate mechanism.
    card(.05, .48, 14.90, .48, "#F3F4F6", "none", .10)
    text(.27, .72, "GENERATOR OBJECTIVE", 9, weight="bold", color=muted, ha="left")
    text(8.60, .72, "Filtered score surrogate + 0.25 × LPIPS regression (LPIPS inherited from DMD)", 10)
    text(.27, .19, "INFERENCE", 9, weight="bold", color=coral, ha="left")
    text(7.80, .19, "Noise + caption  →  one student denoiser call  →  frozen VAE decoder  →  image", 10, color=ink)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(target.with_suffix("."+ext), dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw(Path(__file__).resolve().parents[1]/"report/generated/assignment2/noise_agreement_method")
