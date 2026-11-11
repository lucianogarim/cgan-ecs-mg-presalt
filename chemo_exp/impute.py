"""Reconstruct DWMG in wells where it was not measured.

For each zone, a cGAN is trained on every well that has DWMG and the predictor
set, and then applied to the wells that lack DWMG. The output is a CSV file
with one row per reconstructed depth.

Usage:
  python -m chemo_exp.impute --las-dir data/las --zones data/zone_tops.xlsx --out runs/imputed_dwmg.csv
  python -m chemo_exp.impute --synthetic --epochs 5 --out runs/imputed_synthetic.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from . import data as D
from . import models as M
from .run import AUX, TARGET


def impute_zone(df: pd.DataFrame, zone: str, feats: list[str], epochs: int, seed: int,
                threads: int, stride: int) -> pd.DataFrame:
    zdf = df[df["ZONE"] == zone]
    cols = feats + AUX
    train = D.complete_rows(zdf, cols + [TARGET])
    has_mg = set(zdf.loc[zdf[TARGET].notna(), "WELL_NAME"])
    target_rows = D.complete_rows(zdf[~zdf["WELL_NAME"].isin(has_mg)], cols)
    print(f"Zone {zone}: training on {train['WELL_NAME'].nunique()} wells, "
          f"reconstructing {target_rows['WELL_NAME'].nunique()} wells")
    if train["WELL_NAME"].nunique() < 1 or target_rows.empty:
        return pd.DataFrame()
    w_train = D.make_windows(train, cols, TARGET, AUX, stride=stride)
    w_pred = D.make_windows(target_rows.assign(**{TARGET: 0.0}), cols, TARGET, AUX, stride=1)
    if len(w_train.X) == 0 or len(w_pred.X) == 0:
        return pd.DataFrame()
    model = M.CGANImputer(M.CGANConfig(epochs=epochs, seed=seed, threads=threads)).fit(w_train)
    pred = model.predict_rows(w_pred)
    out = target_rows.loc[pred.index, ["WELL_NAME", "DEPT"]].copy()
    out["ZONE"] = zone
    out[f"{TARGET}_cGAN"] = pred.values
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--las-dir", help="folder with the LAS files")
    ap.add_argument("--zones", help="spreadsheet with columns Well, Zone, Top, Bottom")
    ap.add_argument("--synthetic", action="store_true", help="use the synthetic test dataset")
    ap.add_argument("--featset", choices=["base4", "full"], default="base4",
                    help="predictor set: base4 (set B, default, best in the article) or full (set F)")
    ap.add_argument("--zone-list", default="A,B1,B2,C")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--stride", type=int, default=4, help="window stride for training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="runs/imputed_dwmg.csv")
    a = ap.parse_args(argv)

    df = D.synthetic_dataset() if a.synthetic else D.load_wells(a.las_dir, a.zones)
    feats = D.usable_features(df, D.FEATURES_BASE4 if a.featset == "base4" else D.FEATURES_FULL)
    parts = [impute_zone(df, z, feats, a.epochs, a.seed, a.threads, a.stride) for z in a.zone_list.split(",")]
    res = pd.concat([p for p in parts if not p.empty]) if any(not p.empty for p in parts) else pd.DataFrame()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res.to_csv(a.out, index=False)
    print(f"Wrote {len(res)} reconstructed samples to {a.out}")


if __name__ == "__main__":
    main()
