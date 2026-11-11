"""Rerun cGAN-B on three held-out wells to draw the depth figure.

Usage: python -m chemo_exp.depth_cases --las-dir data/las --zones data/zone_tops.xlsx --out results
"""
import argparse
import os

import pandas as pd

from . import data as D, models as M
from .run import AUX, TARGET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--las-dir", required=True); ap.add_argument("--zones", required=True)
    ap.add_argument("--out", default="results"); ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    df = D.load_wells(a.las_dir, a.zones)
    full = D.FEATURES_FULL + AUX; base = D.FEATURES_BASE4 + AUX
    out = []
    for z, w in [("A", "7-SPH-16D-SPS"), ("B1", "7-SPH-20D-SPS"), ("B1", "7-SPH-2D-SPS")]:
        zdf = df[df.ZONE == z]
        te = D.complete_rows(zdf, full + [TARGET]); te = te[te.WELL_NAME == w]
        tr = D.complete_rows(zdf, base + [TARGET]); tr = tr[tr.WELL_NAME != w]
        m = M.CGANImputer(M.CGANConfig(epochs=a.epochs, threads=2)).fit(D.make_windows(tr, base, TARGET, AUX, stride=4))
        p = m.predict_rows(D.make_windows(te, base, TARGET, AUX, stride=1))
        o = te.loc[p.index, ["WELL_NAME", "DEPT"]].copy(); o["ZONE"] = z; o["cGAN-B"] = p.values
        out.append(o); print(z, w, flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.concat(out).to_csv(os.path.join(a.out, "depth_cgan_b.csv"), index=False)


if __name__ == "__main__":
    main()
