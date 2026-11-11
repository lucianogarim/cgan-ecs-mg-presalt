"""Diagram of the implemented cGAN (1D U-Net generator, 1D PatchGAN discriminator)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

DARK, LIGHT, ORANGE, GREEN, INK, PANEL = "#2c5f8a", "#dce8f3", "#f7c9a1", "#cfe5c6", "#1a1a1a", "#efefef"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})


def box(ax, x, y, w, h, title, sub="", fc=DARK, tc="white", fs=7.2, r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec="none", zorder=2))
    ax.text(x + w / 2, y + h / 2 + (0.018 if sub else 0), title, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.028, sub, ha="center", va="center", color=tc,
                fontsize=fs - 1.2, zorder=3, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, label="", lx=0, ly=0.015):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkA=0, shrinkB=0), zorder=1)
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, ha="center", va="bottom", fontsize=6.3)


def column(ax, x, y, w, h, n, fc):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=INK, lw=0.5, zorder=3))
    for k in range(1, n):
        ax.plot([x, x + w], [y + k * h / n] * 2, color=INK, lw=0.35, zorder=4)


def grid(ax, x, y, w, h, nr, nc, fc):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=INK, lw=0.5, zorder=3))
    for k in range(1, nr):
        ax.plot([x, x + w], [y + k * h / nr] * 2, color=INK, lw=0.35, zorder=4)
    for k in range(1, nc):
        ax.plot([x + k * w / nc] * 2, [y, y + h], color=INK, lw=0.35, zorder=4)


def panel(ax, label, title):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.005, 0.01), 0.99, 0.97, boxstyle="round,pad=0,rounding_size=0.02",
                                fc=PANEL, ec="#9a9a9a", lw=0.8, zorder=0))
    ax.text(0.015, 0.95, f"({label}) {title}", fontsize=8, fontweight="bold", va="top")


fig = plt.figure(figsize=(7.4, 9.0))
axA = fig.add_axes([0.0, 0.70, 1, 0.30])
axG = fig.add_axes([0.0, 0.33, 1, 0.37])
axD = fig.add_axes([0.0, 0.0, 1, 0.33])

# ---------------- (a) overview ----------------
ax = axA; panel(ax, "a", "Conditional GAN for ECS log reconstruction")
box(ax, 0.03, 0.46, 0.17, 0.42, "", "", fc=LIGHT, tc=INK)
ax.text(0.115, 0.83, "Conditioning window", ha="center", fontsize=7, fontweight="bold")
ax.text(0.115, 0.76, r"$\mathbf{X}\in\mathbb{R}^{C\times L}$", ha="center", fontsize=7)
grid(ax, 0.075, 0.50, 0.08, 0.20, 5, 4, "white")
box(ax, 0.03, 0.08, 0.17, 0.30, "", "", fc=LIGHT, tc=INK)
ax.text(0.115, 0.32, "Latent vector", ha="center", fontsize=7, fontweight="bold")
ax.text(0.115, 0.245, r"$\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),\ \mathbf{z}\in\mathbb{R}^{16}$", ha="center", fontsize=6.5)
for k in range(8):
    ax.add_patch(Rectangle((0.05 + k * 0.016, 0.12), 0.016, 0.05, fc="white", ec=INK, lw=0.4, zorder=3))
box(ax, 0.32, 0.38, 0.15, 0.22, "Generator", r"$G_\theta(\mathbf{X},\mathbf{z})$")
box(ax, 0.53, 0.72, 0.09, 0.21, "", "", fc=ORANGE, tc=INK)
ax.text(0.575, 0.895, "Real window", ha="center", fontsize=6.3, fontweight="bold", va="center")
column(ax, 0.565, 0.74, 0.02, 0.12, 6, "white")
box(ax, 0.53, 0.06, 0.09, 0.24, "", "", fc=GREEN, tc=INK)
ax.text(0.575, 0.265, "Generated", ha="center", fontsize=6.3, fontweight="bold", va="center")
column(ax, 0.565, 0.08, 0.02, 0.13, 6, "white")
box(ax, 0.68, 0.38, 0.16, 0.22, "Discriminator", r"$D_\phi(\mathbf{X},\mathbf{y})$")
box(ax, 0.875, 0.40, 0.11, 0.18, "Decision", "real or\ngenerated", fc=LIGHT, tc=INK, fs=6.5)
arrow(ax, 0.20, 0.67, 0.29, 0.67); arrow(ax, 0.29, 0.67, 0.29, 0.52); arrow(ax, 0.29, 0.52, 0.32, 0.52)
ax.plot([0.29, 0.29], [0.67, 0.68], color=INK, lw=0.8); ax.plot([0.29, 0.65], [0.68, 0.68], color=INK, lw=0.8, zorder=1); arrow(ax, 0.65, 0.68, 0.68, 0.55)
ax.text(0.41, 0.695, "condition", ha="center", fontsize=6.3)
arrow(ax, 0.20, 0.23, 0.29, 0.23); ax.plot([0.29, 0.29], [0.23, 0.46], color=INK, lw=0.8); arrow(ax, 0.29, 0.46, 0.32, 0.46)
arrow(ax, 0.47, 0.44, 0.53, 0.20)
arrow(ax, 0.62, 0.82, 0.69, 0.60); arrow(ax, 0.62, 0.18, 0.69, 0.38)
arrow(ax, 0.84, 0.49, 0.875, 0.49)
ax.text(0.40, 0.25, "inference: $\\mathbf{z}=\\mathbf{0}$,\noverlapping windows averaged", fontsize=6, ha="center", style="italic")

# ---------------- (b) U-Net generator ----------------
ax = axG; panel(ax, "b", "Generator: one-dimensional U-Net")
box(ax, 0.02, 0.60, 0.13, 0.26, "Conditioning", "window\n$C\\times L$", fc=LIGHT, tc=INK, fs=6.5)
box(ax, 0.02, 0.24, 0.13, 0.22, "Latent z", "16, broadcast\nalong depth", fc=LIGHT, tc=INK, fs=6.5)
ax.add_patch(Rectangle((0.19, 0.49), 0.035, 0.07, fc="white", ec=INK, lw=0.8, zorder=3))
ax.text(0.2075, 0.525, "+", ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
arrow(ax, 0.15, 0.73, 0.2075, 0.56); arrow(ax, 0.15, 0.35, 0.2075, 0.49)
ax.text(0.2075, 0.43, "$(C{+}16)\\times41$", ha="center", fontsize=6)
# encoder
enc = [(0.27, 0.62, "Conv1D", "64, k5, s1\nBN + LReLU", "64 x 41"),
       (0.40, 0.44, "Conv1D", "128, k5, s2\nBN + LReLU", "128 x 21"),
       (0.53, 0.26, "Conv1D", "128, k5, s2\nBN + LReLU", "128 x 11")]
for x, y, t, s, d in enc:
    box(ax, x, y, 0.105, 0.17, t, s, fs=6.5)
    ax.text(x + 0.0525, y - 0.03, d, ha="center", fontsize=6.2, fontweight="bold")
arrow(ax, 0.225, 0.525, 0.27, 0.66)
arrow(ax, 0.375, 0.66, 0.40, 0.56); arrow(ax, 0.505, 0.50, 0.53, 0.38)
# decoder
dec = [(0.67, 0.44, "Up x2 + Conv1D", "128, k5\nconcat skip\nConv1D 128", "128 x 21"),
       (0.79, 0.62, "Up x2 + Conv1D", "64, k5\nconcat skip\nConv1D 64", "64 x 41")]
for x, y, t, s, d in dec:
    box(ax, x, y, 0.105, 0.20, t, s, fs=6.1)
    ax.text(x - 0.006, y + 0.10, d, ha="right", fontsize=6.2, fontweight="bold")
arrow(ax, 0.635, 0.34, 0.67, 0.48); arrow(ax, 0.775, 0.60, 0.79, 0.68)
# skips
ax.annotate("", xy=(0.67, 0.58), xytext=(0.505, 0.58), arrowprops=dict(arrowstyle="-|>", color="#c0504d", lw=0.9, ls="--"))
ax.annotate("", xy=(0.79, 0.85), xytext=(0.375, 0.85), arrowprops=dict(arrowstyle="-|>", color="#c0504d", lw=0.9, ls="--"))
ax.plot([0.3225, 0.3225], [0.79, 0.85], color="#c0504d", lw=0.9, ls="--"); ax.plot([0.3225, 0.375], [0.85, 0.85], color="#c0504d", lw=0.9, ls="--")
ax.plot([0.4525, 0.4525], [0.61, 0.58], color="#c0504d", lw=0.9, ls="--"); ax.plot([0.4525, 0.505], [0.58, 0.58], color="#c0504d", lw=0.9, ls="--")
ax.text(0.58, 0.865, "skip connections", color="#c0504d", fontsize=6.2, ha="center")
box(ax, 0.79, 0.20, 0.105, 0.20, "Conv1D", "1, k3\nsigmoid", fc=ORANGE, tc=INK, fs=6.5)
arrow(ax, 0.8425, 0.62, 0.8425, 0.40)
box(ax, 0.915, 0.18, 0.07, 0.26, "", "", fc=GREEN, tc=INK)
ax.text(0.95, 0.405, "Output", ha="center", fontsize=6.5, fontweight="bold", va="center")
column(ax, 0.94, 0.21, 0.02, 0.15, 6, "white")
arrow(ax, 0.895, 0.30, 0.915, 0.30)
ax.text(0.95, 0.12, "$1\\times41$\nscaled DWMG", ha="center", fontsize=6, va="top")

# ---------------- (c) PatchGAN discriminator ----------------
ax = axD; panel(ax, "c", "Discriminator: one-dimensional PatchGAN")
box(ax, 0.02, 0.56, 0.13, 0.26, "Conditioning", "window\n$C\\times L$", fc=LIGHT, tc=INK, fs=6.5)
box(ax, 0.02, 0.14, 0.13, 0.30, "", "", fc=ORANGE, tc=INK)
ax.text(0.085, 0.40, "Target window", ha="center", fontsize=6.5, fontweight="bold", va="center")
column(ax, 0.075, 0.17, 0.02, 0.17, 6, "white")
ax.text(0.085, 0.10, "real $\\mathbf{y}$ or generated $\\hat{\\mathbf{y}}$", ha="center", fontsize=5.8, va="top")
ax.add_patch(Rectangle((0.19, 0.45), 0.035, 0.08, fc="white", ec=INK, lw=0.8, zorder=3))
ax.text(0.2075, 0.49, "+", ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
arrow(ax, 0.15, 0.69, 0.2075, 0.53); arrow(ax, 0.15, 0.29, 0.2075, 0.45)
ax.text(0.2075, 0.38, "$(C{+}1)\\times41$", ha="center", fontsize=6)
blocks = [(0.28, "Conv1D", "64, k5, s2\nLReLU", "64 x 21"),
          (0.45, "Conv1D", "128, k5, s2\nBN + LReLU", "128 x 11"),
          (0.62, "Conv1D", "1, k3, s1", "1 x 11")]
for x, t, s, d in blocks:
    box(ax, x, 0.38, 0.12, 0.22, t, s, fs=6.6)
    ax.text(x + 0.06, 0.33, d, ha="center", fontsize=6.2, fontweight="bold")
arrow(ax, 0.225, 0.49, 0.28, 0.49); arrow(ax, 0.40, 0.49, 0.45, 0.49); arrow(ax, 0.57, 0.49, 0.62, 0.49)
box(ax, 0.80, 0.30, 0.18, 0.38, "", "", fc=GREEN, tc=INK)
ax.text(0.89, 0.60, "Output", ha="center", fontsize=6.8, fontweight="bold", va="center")
ax.text(0.89, 0.44, "one logit per\ndepth patch\n(real or generated)", ha="center", fontsize=6.2, va="center")
arrow(ax, 0.74, 0.49, 0.80, 0.49)

import sys
out = sys.argv[1] if len(sys.argv) > 1 else "figures"
fig.savefig(f"{out}/fig_cgan_arch.pdf")
