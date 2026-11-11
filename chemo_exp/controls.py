"""Controls that separate the effects of the predictor set, the training wells and
the forest settings, plus the cGAN trained on the basic set (cGAN-B).

Same test rows and leave-one-well-out protocol as the main experiment.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import data as D
from . import models as M
from .run import AUX, TARGET, metrics


def rf20(train, test, feats, seed=42):
    rf = RandomForestRegressor(n_estimators=20, n_jobs=-1, random_state=seed)
    rf.fit(train[feats].to_numpy(), train[TARGET].to_numpy())
    return rf.predict(test[feats].to_numpy())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", help="pre-loaded dataframe (pandas pickle)")
    ap.add_argument("--las-dir"); ap.add_argument("--zones")
    ap.add_argument("--zone-list", default="A,B1,B2")
    ap.add_argument("--cgan-epochs", type=int, default=80)
    ap.add_argument("--no-cgan", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="controls")
    a = ap.parse_args(argv)
    df = pd.read_pickle(a.pickle) if a.pickle else D.load_wells(a.las_dir, a.zones)
    full = D.FEATURES_FULL + AUX
    base = D.FEATURES_BASE4 + AUX
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for zone in a.zone_list.split(","):
        zdf = df[df["ZONE"] == zone]
        lab_full = D.complete_rows(zdf, full + [TARGET])
        c = lab_full.groupby("WELL_NAME").size()
        wells = c[c >= 100].index.tolist()
        lab_full = lab_full[lab_full["WELL_NAME"].isin(wells)]
        lab_base = D.complete_rows(zdf, base + [TARGET])        # includes wells without NMR or spectral logs
        print(f"\n=== Zone {zone}: {len(wells)} test wells; basic-set training with "
              f"{lab_base['WELL_NAME'].nunique()} wells ===")
        for w in wells:
            t0 = time.time()
            te = lab_full[lab_full["WELL_NAME"] == w]
            tr_f = lab_full[lab_full["WELL_NAME"] != w]          # narrow training wells (set F)
            tr_b = lab_base[lab_base["WELL_NAME"] != w]          # wide training wells (set B)
            preds = {
                "RF-B": rf20(tr_b, te, base),
                "RF-B-narrow": rf20(tr_f, te, base),
                "RF-F20": rf20(tr_f, te, full),
                "RF-B300": M.rf_direct(tr_b, te, base, TARGET),
            }
            preds = {k: pd.Series(v, index=te.index) for k, v in preds.items()}
            if not a.no_cgan:
                Wtr = D.make_windows(tr_b, base, TARGET, AUX, stride=4)
                Wte = D.make_windows(te, base, TARGET, AUX, stride=1)
                cfg = M.CGANConfig(epochs=a.cgan_epochs, threads=a.threads)
                preds["cGAN-B"] = M.CGANImputer(cfg).fit(Wtr).predict_rows(Wte)
            common = te.index
            for p in preds.values():
                common = common.intersection(p.index)
            y = te.loc[common, TARGET].to_numpy()
            for k, p in preds.items():
                r = metrics(y, p.loc[common].to_numpy()); r.update(zone=zone, well=w, model=k)
                rows.append(r)
            print(f"   {w}: " + " ".join(f"{r['model']}={r['R2']:.3f}" for r in rows[-len(preds):])
                  + f" ({time.time()-t0:.0f}s)")
            pd.DataFrame(rows).to_csv(os.path.join(a.out, "controls_per_well.csv"), index=False)


if __name__ == "__main__":
    main()
