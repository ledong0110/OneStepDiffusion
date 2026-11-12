#!/usr/bin/env python3
"""Vector schematic of cached, direction-only finite-response alignment."""

from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch


def draw(target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    colors = dict(ink="#172C43", muted="#5C7083", teal="#117D86", coral="#BF6747")
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "mathtext.fontset": "dejavusans"}):
        fig, ax = plt.subplots(figsize=(11, 4.7))
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set(xlim=(0, 16), ylim=(0, 6.8))
        ax.axis("off")

        def text(x, y, value, size=12, color="ink", weight="normal", ha="center"):
            ax.text(x, y, value, fontsize=size, color=colors.get(color, color),
                    weight=weight, ha=ha, va="center")

        def card(x, y, w, h, face="white", edge="#CBD6DF", shadow=False):
            if shadow:
                ax.add_patch(FancyBboxPatch((x+.035, y-.045), w, h,
                    boxstyle="round,pad=0,rounding_size=.12", facecolor=colors["ink"],
                    edgecolor="none", alpha=.07))
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                boxstyle="round,pad=0,rounding_size=.12", facecolor=face,
                edgecolor=edge, linewidth=1.1))

        def arrow(a, b, color="muted", **kwargs):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                color=colors.get(color, color), linewidth=1.45, shrinkA=1, shrinkB=1, **kwargs))

        def line(points, color="muted", **kwargs):
            ax.plot(*zip(*points), color=colors.get(color, color), lw=1.3, **kwargs)

        panels = [(.06, 4.48, "Cached teacher pairs", "#EFF7F6", "teal"),
                  (4.76, 5.27, "One-step student", "#FCF3EC", "coral"),
                  (10.25, 5.69, "Our direction alignment", "#F3F6FA", "ink")]
        for i, (x, w, title, face, color) in enumerate(panels, 1):
            card(x, 2.47, w, 4.17, face, "none")
            ax.add_patch(Circle((x+.36, 6.23), .19, facecolor=colors[color], edgecolor="none"))
            text(x+.36, 6.23, str(i), 10, "white", "bold")
            text(x+.70, 6.23, title, 14, color, "bold", "left")

        # The bank stores both initial noises and frozen teacher endpoints.
        # Teacher endpoints are reused; this is not an online teacher forward.
        card(.42, 5.12, 3.38, .56, "white", "#B8D7D2")
        text(2.11, 5.40, "Same caption  $c$", 12, "teal")
        for y, index in ((4.54, 1), (3.79, 2)):
            card(.44, y-.26, .94, .52, "#E5EBF0", "none")
            text(.91, y, rf"$z_{index}$", 15)
            arrow((1.40, y), (2.55, y), "teal")
            text(1.98, y+.22, "cached", 9, "muted")
            card(2.58, y-.26, 1.42, .52, "white", "#A5CAC4", shadow=True)
            text(3.29, y, rf"$T(z_{index},c)$", 13, "teal")
        # A small lock marks fixed targets; it stays clear of the response arrow.
        ax.add_patch(Arc((4.08, 5.77), .18, .22, theta1=0, theta2=180,
                         edgecolor=colors["teal"], lw=1.3))
        card(3.96, 5.58, .24, .19, colors["teal"], "none")
        text(2.28, 2.69, "Independent noises; fixed targets", 10, "muted")

        # Frozen displacement bypasses the generator.
        line([(4.00, 3.79), (4.26, 3.79), (4.26, 5.40)], "teal")
        line([(4.00, 4.54), (4.26, 4.54)], "teal")
        arrow((4.26, 5.40), (10.52, 5.40), "teal")
        text(7.38, 5.71, r"$d_T=T(z_2,c)-T(z_1,c)$", 13, "teal")

        # The same two cached noises go through the same rank-32 student.
        line([(.44, 4.54), (.25, 4.54), (.25, 3.03), (6.20, 3.03)], "coral")
        line([(.44, 3.79), (.25, 3.79)], "coral")
        arrow((6.20, 3.03), (6.20, 3.58), "coral")
        text(3.62, 3.27, r"$z_1,z_2$ and caption $c$", 11, "coral")
        card(5.12, 3.61, 2.15, 1.39, "white", "#D9A88F", shadow=True)
        text(6.195, 4.64, r"$G_\theta$", 20, "coral", "bold")
        text(6.195, 4.17, "Trainable LoRA", 11, "coral")
        text(6.195, 3.85, "rank 32", 10, "muted")
        line([(7.27, 4.30), (7.50, 4.30), (7.50, 4.57)], "coral")
        line([(7.50, 4.30), (7.50, 3.89)], "coral")
        for y, index in ((4.57, 1), (3.89, 2)):
            arrow((7.50, y), (7.79, y), "coral")
            card(7.82, y-.25, 1.20, .50, "white", "#D9A88F")
            text(8.42, y, rf"$x_S^{{({index})}}$", 14, "coral")
            line([(9.02, y), (9.29, y)], "coral")
        line([(9.29, 3.89), (9.29, 4.57)], "coral")
        arrow((9.29, 4.23), (10.52, 4.23), "coral")
        text(9.83, 4.49, r"$d_S$", 13, "coral")
        text(8.35, 3.22, r"$d_S=x_S^{(2)}-x_S^{(1)}$", 12, "coral")
        text(7.40, 2.69, "Reuse paired-regression outputs", 10, "muted")

        # Schematic geometry: only the angle is penalized, not vector length.
        origin = (11.22, 3.88)
        teacher_end, student_end = (14.56, 4.61), (13.27, 5.16)
        ax.add_patch(Circle(origin, .045, facecolor=colors["ink"], edgecolor="none"))
        arrow(origin, teacher_end, "teal")
        arrow(origin, student_end, "coral")
        angles = [math.degrees(math.atan2(end[1]-origin[1], end[0]-origin[0]))
                  for end in (teacher_end, student_end)]
        ax.add_patch(Arc(origin, 1.75, 1.75, theta1=angles[0], theta2=angles[1],
                         edgecolor=colors["ink"], lw=1.25))
        text(12.36, 4.37, r"$\phi$", 13)
        text(14.62, 4.83, r"$d_T$", 14, "teal")
        text(13.29, 5.38, r"$d_S$", 14, "coral")
        text(13.08, 3.45, "Align direction, leave length free", 11, "muted")
        text(13.08, 2.88, r"$\mathcal{L}_{\mathrm{angle}}=1-\cos(d_S,d_T)$", 14)

        card(.06, 1.28, 15.88, .99, "#F3F4F6", "none")
        text(.34, 1.79, "TRAINING", 11, "muted", "bold", "left")
        text(6.32, 1.84, r"$\mathcal{L}_{\mathrm{DMD}}+0.25\,\mathcal{L}_{\mathrm{LPIPS}}$", 17)
        text(6.32, 1.47, "Existing DMD objective", 10, "muted")
        text(11.96, 1.84, r"$+\;\lambda\,\mathcal{L}_{\mathrm{angle}}$", 17, "coral")
        text(11.96, 1.47, "Added response term", 10, "coral")

        card(.06, .17, 15.88, .85, "white", "#D4DEE7")
        text(.34, .60, "INFERENCE", 11, "muted", "bold", "left")
        text(7.78, .60, r"$(z,c)\quad\longrightarrow\quad G_\theta\quad\longrightarrow\quad\mathrm{image}$", 16)
        card(13.30, .35, 2.22, .49, "#E2F1ED", "none")
        text(14.41, .595, "1 denoiser call", 11, "teal", "bold")

        for extension in ("pdf", "svg", "png"):
            fig.savefig(target.with_suffix("."+extension), dpi=240, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    draw(Path(__file__).resolve().parents[1]/"report/generated/assignment2/response_alignment_method")
