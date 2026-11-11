"""cGAN versus random forest for DWMG reconstruction, per stratigraphic zone.

Protocol: leave-one-well-out (LOWO) inside each zone. All scaling statistics and
models are fitted on the training wells only; all models are scored on the same
rows of the held-out well.

Usage:
  python -m chemo_exp.run --las-dir data/las --zones data/zone_tops.xlsx --out runs/main
  python -m chemo_exp.run --synthetic --epochs 5 --out results_synth
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import data as D
from . import models as M

TARGET = "DWMG"
AUX = ["DWCA", "DWSI", "DWAL"]       # co-measured ECS logs, used as predictors and in the penalties


def metrics(y, p):
    return dict(R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=mean_absolute_error(y, p), n=len(y))


def cgan_variants(epochs: int, seed: int, threads: int = 2):
    base = dict(epochs=epochs, seed=seed, threads=threads)
    return {
        "cGAN-F": M.CGANConfig(**base),
        "cGAN-F-noPhys": M.CGANConfig(**base, physics=False),
        "CNN-F-L1": M.CGANConfig(**base, adversarial=False, physics=False),
    }


def run_zone(df: pd.DataFrame, zone: str, feats: list[str], epochs: int, seed: int,
             min_rows: int, models: set[str], stride: int = 4, threads: int = 2, pred_rows=None):
    zdf = df[df["ZONE"] == zone]
    all_feats = feats + AUX
    lab = D.complete_rows(zdf, all_feats + [TARGET])
    counts = lab.groupby("WELL_NAME").size()
    wells = counts[counts >= min_rows].index.tolist()
    lab = lab[lab["WELL_NAME"].isin(wells)]
    print(f"\n=== Zone {zone}: {len(wells)} wells with measured {TARGET}, {len(lab)} rows ===")
    if len(wells) < 2:
        print("   not enough wells for LOWO")
        return []

    # Basic set B with the wide training wells (RF-II)
    basic_cols = AUX + [TARGET] + [c for c in D.FEATURES_BASE4 if c in df.columns]

    lab_ctx, ctx_feats = M.add_context(lab, all_feats)
    rows = []
    for w in wells:
        t0 = time.time()
        tr, te = lab[lab["WELL_NAME"] != w], lab[lab["WELL_NAME"] == w]
        preds = {}
        if "RF-II" in models:
            tr_t = D.complete_rows(zdf[zdf["WELL_NAME"] != w], basic_cols)
            preds["RF-II"] = pd.Series(
                M.rf_iterative(tr_t, te, basic_cols, TARGET, seed), index=te.index)
        if "RF-F" in models:
            preds["RF-F"] = pd.Series(M.rf_direct(tr, te, all_feats, TARGET, seed), index=te.index)
        if "RF-F-ctx" in models:
            trc, tec = lab_ctx.loc[tr.index], lab_ctx.loc[te.index]
            preds["RF-F-ctx"] = pd.Series(M.rf_direct(trc, tec, ctx_feats, TARGET, seed), index=te.index)
        cg = {k: v for k, v in cgan_variants(epochs, seed, threads).items() if k in models}
        if cg:
            Wtr = D.make_windows(tr, all_feats, TARGET, AUX, stride=stride)
            Wte = D.make_windows(te, all_feats, TARGET, AUX, stride=1)
            if len(Wtr.X) and len(Wte.X):
                for name, cfg in cg.items():
                    m = M.CGANImputer(cfg).fit(Wtr)
                    preds[name] = m.predict_rows(Wte)
                    if name == "cGAN-F":
                        preds["cGAN-F-stoch"] = m.predict_rows(Wte, stochastic=True)
        # score every model on the same rows
        common = te.index
        for p in preds.values():
            common = common.intersection(p.index)
        y = te.loc[common, TARGET].to_numpy()
        if pred_rows is not None:
            pr = te.loc[common, ["WELL_NAME", "DEPT", TARGET]].copy(); pr["ZONE"] = zone
            for name, p in preds.items():
                pr[name] = p.loc[common].to_numpy()
            pred_rows.append(pr)
        for name, p in preds.items():
            r = metrics(y, p.loc[common].to_numpy())
            r.update(zone=zone, well=w, model=name)
            rows.append(r)
        print(f"   {w}: n={len(common)} " +
              " ".join(f"{k}={r2_score(y, v.loc[common]):.3f}" for k, v in preds.items()) +
              f"  ({time.time()-t0:.0f}s)")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--las-dir"); ap.add_argument("--zones")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--pickle", help="pre-loaded dataframe (pandas pickle)")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--featset", choices=["base4", "full"], default="full")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-rows", type=int, default=100)
    ap.add_argument("--zone-list", default="A,B1,B2,C")
    ap.add_argument("--models", default="RF-II,RF-F,RF-F-ctx,cGAN-F,cGAN-F-noPhys,CNN-F-L1")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    if a.synthetic:
        df = D.synthetic_dataset()
    elif a.pickle:
        df = pd.read_pickle(a.pickle)
    else:
        df = D.load_wells(a.las_dir, a.zones)
    feats = D.usable_features(df, D.FEATURES_BASE4 if a.featset == "base4" else D.FEATURES_FULL)
    print("Preditores:", feats + AUX)
    os.makedirs(a.out, exist_ok=True)
    rows, preds = [], []
    for z in a.zone_list.split(","):
        rows += run_zone(df, z, feats, a.epochs, a.seed, a.min_rows, set(a.models.split(",")),
                         a.stride, a.threads, preds)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "lowo_per_well.csv"), index=False)
        if preds:
            pd.concat(preds).to_csv(os.path.join(a.out, "lowo_predictions.csv"), index=False)
    res = pd.DataFrame(rows)
    if res.empty:
        return
    summ = res.groupby(["zone", "model"])[["R2", "RMSE", "MAE"]].agg(["mean", "std", "median"])
    summ.to_csv(os.path.join(a.out, "lowo_summary.csv"))
    print("\n", res.groupby(["zone", "model"])[["R2", "RMSE"]].mean().unstack("model").round(3))
    json.dump(vars(a), open(os.path.join(a.out, "run_args.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
