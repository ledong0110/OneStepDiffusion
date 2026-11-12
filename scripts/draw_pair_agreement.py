#!/usr/bin/env python3
"""Vector schematic of independent-pair agreement; grids are illustrative."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def draw(target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(11, 3.7))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 15.6), ylim=(0, 5.25))
    ax.axis("off")
    ink, muted, teal, violet, coral = "#172C43", "#5C7083", "#117D86", "#6957A4", "#BF6747"

    def text(x, y, value, size=11, color=ink, weight="normal", ha="center"):
        ax.text(x, y, value, fontsize=size, color=color, weight=weight, ha=ha, va="center")

    def card(x, y, w, h, face="white", edge="#CBD6DF", shadow=False):
        if shadow:
            ax.add_patch(FancyBboxPatch((x+.035, y-.045), w, h,
                boxstyle="round,pad=0,rounding_size=.09", facecolor="#172C43",
                edgecolor="none", alpha=.07))
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0,rounding_size=.09", facecolor=face,
            edgecolor=edge, linewidth=1.05))

    def arrow(a, b, color=muted):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
            color=color, linewidth=1.15, shrinkA=1, shrinkB=1))

    for x, w, title, face in [(.05, 6.25, "1   Four score probes", "#F3F7FB"),
                             (6.45, 3.10, "2   Average pairs", "#F5F2FA"),
                             (9.70, 5.85, "3   Spatial weighting", "#EEF8F6")]:
        card(x, 1.04, w, 4.10, face, "none")
        text(x+.20, 4.82, title, 13, weight="bold", ha="left")

    card(.23, 2.60, 1.10, .83, "#FAE8DE", coral, shadow=True)
    text(.78, 3.17, "Student", 10, color=coral, weight="bold")
    text(.78, 2.86, "$x=G_\\theta(z,c)$", 9.3)
    ax.plot([1.33, 1.48, 1.48], [3.02, 3.02, 3.72], color=muted, lw=1.1)
    ax.plot([1.48, 1.48], [3.02, 2.31], color=muted, lw=1.1)
    for y, index in [(3.72, 1), (2.31, 2)]:
        arrow((1.48, y), (1.71, y))
        card(1.73, y-.31, 1.55, .62, shadow=True)
        text(2.505, y, rf"$\alpha_t x\pm\sigma_t\epsilon_{index}$", 11)
        arrow((3.28, y), (3.56, y))
    text(2.50, 3.11, "Independent", 9.3, color=violet)
    text(2.50, 2.89, "noise draws", 9.3, color=violet)
    text(2.50, 1.78, "Same latent, time, caption", 10, color=muted)

    card(3.58, 1.99, 1.59, 2.04, shadow=True)
    text(4.375, 3.72, "Shared models", 9.3, color=muted)
    text(4.375, 3.30, "Teacher +", 10, weight="bold")
    text(4.375, 2.99, "fake scorer", 10, weight="bold")
    text(4.375, 2.31, "stop gradient", 9.3, color=muted)
    for y, label in [(3.72, "$u^+,u^-$"), (2.31, "$v^+,v^-$")]:
        arrow((5.17, y), (6.69, y), violet)
        text(5.85, y+.27, label, 11, color=violet)
    text(.26, 1.32, "Directions use DMD's per-probe normalization", 10, color=muted, ha="left")

    for y, formula in [(3.72, "$p=(u^++u^-)/2$"), (2.31, "$q=(v^++v^-)/2$")]:
        card(6.71, y-.35, 2.57, .70, "white", "#C8BDDE", shadow=True)
        text(7.995, y, formula, 12, color=violet)
    text(8.00, 3.01, "Compare after averaging", 10, color=muted)
    text(8.00, 1.32, "Two independent pair means", 10, color=muted)

    ax.plot([9.28, 9.49, 9.49, 9.28], [3.72, 3.72, 2.31, 2.31], color=violet, lw=1.1)
    arrow((9.49, 3.72), (10.02, 3.72), teal)
    arrow((9.49, 2.31), (10.02, 2.31), teal)
    card(10.04, 3.20, 3.06, .97, "white", "#9DCAC2", shadow=True)
    text(11.57, 3.94, "Patch agreement", 9.5, color=teal, weight="bold")
    text(11.57, 3.55, "$r=\\mathrm{clip}\\!\\left(\\frac{2\\langle p,q\\rangle}{\\|p\\|^2+\\|q\\|^2},0,1\\right)$", 12, color=teal)
    arrow((13.10, 3.72), (13.47, 3.72), teal)
    values = [[.2, .3, .6, .8], [.1, .4, .8, .6], [.3, .3, .7, .5], [.4, .2, .1, .4]]
    for row in range(4):
        for col in range(4):
            ax.add_patch(Rectangle((13.50+.17*col, 3.38+.17*row), .154, .154,
                facecolor=plt.cm.BuGn(.15+.7*values[row][col]), edgecolor="none"))
    text(14.84, 3.72, "$w=1+r$", 11, color=teal)
    text(14.28, 4.34, "schematic weights", 10, color=muted)
    arrow((13.83, 3.36), (13.83, 2.95), teal)
    card(10.04, 1.98, 2.04, .66, shadow=True)
    text(11.06, 2.31, "$m=(p+q)/2$", 11, color=teal)
    arrow((12.08, 2.31), (12.42, 2.31), teal)
    card(12.44, 1.70, 2.84, 1.23, "#D7EEE8", "#9DCAC2", shadow=True)
    text(13.86, 2.62, "$h=w\\odot m$", 12, color=teal)
    text(13.86, 2.13, "$\\widetilde m=h\\,\\|m\\|_2/\\|h\\|_2$", 12, color=teal)
    text(12.63, 1.32, "Weight locally; preserve the total update norm", 10, color=muted)

    card(.05, .48, 15.50, .42, "#F3F4F6", "none")
    text(.26, .69, "MATCHED CONTROLS", 10, color=muted, weight="bold", ha="left")
    text(9.00, .69, "Four-probe mean (no weights)   |   Shifted weight map   |   Same initialization and updates", 10)
    text(.26, .18, "UNCHANGED", 10, color=coral, weight="bold", ha="left")
    text(8.50, .18, "DMD + 0.25 × LPIPS regression   •   Rank-32 student   •   One denoiser call at inference", 10)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(target.with_suffix("."+ext), dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw(Path(__file__).resolve().parents[1]/"report/generated/assignment2/pair_agreement_method")
