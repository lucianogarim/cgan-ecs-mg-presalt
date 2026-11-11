"""Figures and tables of the article from the experiment CSV files."""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"RF-II": "#2a78d6", "cGAN-B": "#eb6834", "RF-F": "#1baf7a", "cGAN-F": "#eda100"}
MARK = {"RF-II": "o", "RF-F": "s", "cGAN-F": "^", "cGAN-B": "D"}
RENAME = {"cGAN-base4-wide": "cGAN-B"}
ZONES = ["A", "B1", "B2"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "savefig.dpi": 300,
})


def fig_r2(res: pd.DataFrame, out: str):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    models = ["RF-II", "RF-F", "cGAN-F", "cGAN-B"]
    for ax, z in zip(axes, ZONES):
        r = res[res.zone == z]
        for i, m in enumerate(models):
            v = r[r.model == m].R2.clip(lower=-1.0).to_numpy()
            x = i + np.linspace(-0.18, 0.18, len(v)) if len(v) > 1 else np.array([i])
            ax.scatter(x, v, s=18, marker=MARK[m], facecolor=SERIES[m], edgecolor="white",
                       linewidth=0.6, zorder=3, label=m)
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=1.4, zorder=4)
        ax.set_xticks(range(len(models)), models, rotation=0, fontsize=7.5)
        ax.set_title(f"Zone {z} ({r.well.nunique()} wells)", color=INK, fontsize=8.5)
        ax.axhline(0, color=INK2, lw=0.7)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("$R^2$ in held-out well (clipped at $-1$)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03), markerscale=1.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out); plt.close(fig)


def fig_depth(pred: pd.DataFrame, cases: list[tuple[str, str]], out: str):
    fig, axes = plt.subplots(1, len(cases), figsize=(7.2, 4.2), sharey=False)
    for ax, (z, w) in zip(np.atleast_1d(axes), cases):
        p = pred[(pred.ZONE == z) & (pred.WELL_NAME == w)].sort_values("DEPT")
        ax.plot(p.DWMG, p.DEPT, color=INK, lw=1.0, label="Measured")
        for m in ["RF-II", "cGAN-B"]:
            ax.plot(p[m], p.DEPT, color=SERIES[m], lw=1.0, label=m)
        ax.invert_yaxis(); ax.set_xlabel("DWMG (dry-weight fraction)")
        ax.set_title(f"{w}, zone {z}", fontsize=8.5, color=INK)
    np.atleast_1d(axes)[0].set_ylabel("Measured depth (m)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out); plt.close(fig)


def fig_wells(df: pd.DataFrame, out: str):
    z = df[df.ZONE.isin(ZONES) & df.DWMG.notna()]
    g = z.groupby("WELL_NAME").agg(mean=("DWMG", "mean"), zero=("DWMG", lambda s: (s == 0).mean()))
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.scatter(g.zero * 100, g["mean"], s=24, color=SERIES["RF-II"], edgecolor="white", zorder=3)
    for w, r in g.iterrows():
        ax.annotate(w.replace("-SPS", ""), (r.zero * 100, r["mean"]), fontsize=6, color=INK2,
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("Samples with DWMG = 0 (%)"); ax.set_ylabel("Mean DWMG, zones A to B2")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def tables(res: pd.DataFrame, models: list[str]) -> str:
    lines = []
    for z in ZONES:
        r = res[res.zone == z]
        for m in models:
            v = r[r.model == m]
            if v.empty:
                continue
            lines.append(f"{z} & {m} & {v.R2.mean():.3f} $\\pm$ {v.R2.std():.3f} & {v.R2.median():.3f} & "
                         f"{v.RMSE.mean()*1000:.2f} & {v.MAE.mean()*1000:.2f} \\\\")
        lines.append("\\midrule")
    return "\n".join(lines[:-1])


def paired(res: pd.DataFrame, a: str, b: str):
    out = []
    for z in ZONES + ["all"]:
        r = res if z == "all" else res[res.zone == z]
        p = r.pivot_table(index=["zone", "well"], columns="model", values="R2")[[a, b]].dropna()
        d = p[a] - p[b]
        pv = wilcoxon(p[a], p[b]).pvalue if len(p) >= 5 else np.nan
        out.append((z, len(p), d.median(), (d > 0).sum(), pv))
    return out


if __name__ == "__main__":
    res = pd.read_csv(sys.argv[1]); pred = pd.read_csv(sys.argv[2]); df = pd.read_pickle(sys.argv[3])
    outdir = sys.argv[4]
    if len(sys.argv) > 5:
        res = pd.concat([res, pd.read_csv(sys.argv[5])])
    if len(sys.argv) > 6:
        extra = pd.read_csv(sys.argv[6])
        pred = pred.merge(extra, on=["WELL_NAME", "DEPT", "ZONE"], how="left")
    res["model"] = res["model"].replace(RENAME)
    pred = pred.rename(columns=RENAME)
    fig_r2(res, f"{outdir}/fig_r2_lowo.pdf")
    fig_depth(pred, [("A", "7-SPH-16D-SPS"), ("B1", "7-SPH-20D-SPS"), ("B1", "7-SPH-2D-SPS")],
              f"{outdir}/fig_depth.pdf")
    fig_wells(df, f"{outdir}/fig_wells.pdf")

