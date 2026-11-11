"""Loading of the Sapinhoá LAS files, zone assignment and depth windowing."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

NULLS = [-9999, -9999.25, -999.25]
ECS = ["DWCA", "DWSI", "DWAL", "DWMG", "DWFE", "DWTI"]

# Predictor sets
FEATURES_BASE4 = ["GR", "RHOB", "NPHI", "DT"]          # basic set B (plus Ca, Si, Al)
FEATURES_P13 = [                                         # conventional, NMR and spectral logs
    "GR", "RHOB", "NPHI", "DT", "PE",
    "NMRE_FINAL", "NMRFL_FINAL", "NMRT_FINAL", "T2LM", "MPERM",
    "HFK", "HTHO", "HURA",
]
# T2LM and MPERM are missing in several wells with DWMG (2D, 10, 19D, 21D, 24D);
# excluding them keeps more wells for leave-one-well-out validation.
FEATURES_FULL = [c for c in FEATURES_P13 if c not in ("T2LM", "MPERM")]


def load_wells(las_dir: str, zones_xlsx: str) -> pd.DataFrame:
    import lasio

    dfs = []
    for f in sorted(os.listdir(las_dir)):
        if not f.lower().endswith(".las"):
            continue
        las = lasio.read(os.path.join(las_dir, f))
        df = las.df().reset_index()
        if "DEPTH" in df.columns and "DEPT" not in df.columns:
            df = df.rename(columns={"DEPTH": "DEPT"})
        df["WELL_NAME"] = las.well.WELL.value if "WELL" in las.well else f
        df["ARQUIVO_ORIGEM"] = f
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.replace(NULLS, np.nan)
    # ECS zeros are kept: they occur in short interleaved runs and the smallest
    # positive value is 0.001, so they are values below the tool resolution.
    return assign_zones(df, pd.read_excel(zones_xlsx))


def assign_zones(df: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ZONE"] = "Sem_Zona"
    for _, r in zones.iterrows():
        m = (df["WELL_NAME"] == r["Well"]) & (df["DEPT"] >= r["Top"]) & (df["DEPT"] <= r["Bottom"])
        df.loc[m, "ZONE"] = r["Zone"]
    return df


def usable_features(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    return [c for c in wanted if c in df.columns]


def complete_rows(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.dropna(subset=cols)


# --------------------------------------------------------------------------
# Depth windows for the cGAN
# --------------------------------------------------------------------------
@dataclass
class Windows:
    X: np.ndarray        # (n, C, L) conditioning
    y: np.ndarray        # (n, 1, L) target
    aux: np.ndarray      # (n, A, L) observed ECS logs used in the penalties (original units)
    well: np.ndarray     # (n,) well name
    idx: np.ndarray      # (n, L) row index in the source dataframe


def contiguous_segments(depth: np.ndarray, step_tol: float = 1.5) -> list[np.ndarray]:
    """Split an interval wherever there is a depth gap (removed samples)."""
    if len(depth) == 0:
        return []
    d = np.diff(depth)
    med = np.median(d) if len(d) else 0.1524
    breaks = np.where(d > step_tol * med)[0] + 1
    return np.split(np.arange(len(depth)), breaks)


def make_windows(df: pd.DataFrame, feats: list[str], target: str, aux_cols: list[str],
                 L: int = 41, stride: int = 1) -> Windows:
    Xs, ys, auxs, wells, idxs = [], [], [], [], []
    for w, g in df.sort_values(["WELL_NAME", "DEPT"]).groupby("WELL_NAME", sort=False):
        dep = g["DEPT"].to_numpy()
        F = g[feats].to_numpy(np.float32)
        Y = g[[target]].to_numpy(np.float32)
        A = g[aux_cols].to_numpy(np.float32) if aux_cols else np.zeros((len(g), 0), np.float32)
        I = g.index.to_numpy()
        for seg in contiguous_segments(dep):
            if len(seg) < L:
                continue
            for s in range(0, len(seg) - L + 1, stride):
                sl = seg[s:s + L]
                Xs.append(F[sl].T); ys.append(Y[sl].T); auxs.append(A[sl].T)
                wells.append(w); idxs.append(I[sl])
    if not Xs:
        C = len(feats)
        return Windows(np.zeros((0, C, L), np.float32), np.zeros((0, 1, L), np.float32),
                       np.zeros((0, len(aux_cols), L), np.float32), np.array([]), np.zeros((0, L), int))
    return Windows(np.stack(Xs), np.stack(ys), np.stack(auxs), np.array(wells), np.stack(idxs))


# --------------------------------------------------------------------------
# Synthetic data to test the pipeline without real data
# --------------------------------------------------------------------------
def synthetic_dataset(n_wells: int = 8, n_per_zone: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    zones = ["A", "B1", "B2", "C"]
    for w in range(n_wells):
        depth0 = 5000.0
        for zi, z in enumerate(zones):
            n = n_per_zone if z != "C" else n_per_zone // 2
            t = np.arange(n)
            lat = np.cumsum(rng.normal(0, 0.15, n))           # latent geological signal
            lat2 = np.sin(t / (15 + 5 * w)) + 0.3 * rng.normal(size=n)
            ca = np.clip(0.30 - 0.04 * lat2 + 0.01 * rng.normal(size=n), 0.05, 0.45)
            si = np.clip(0.08 + 0.03 * lat2 + 0.01 * rng.normal(size=n), 0.005, 0.3)
            al = np.clip(0.007 + 0.002 * lat + 0.001 * rng.normal(size=n), 1e-4, 0.05)
            mg = np.clip(0.019 + 0.006 * np.tanh(lat) - 0.02 * (si - 0.08) + 0.002 * rng.normal(size=n), 1e-4, 0.08)
            d = {
                "DEPT": depth0 + t * 0.1524,
                "WELL_NAME": f"W{w:02d}", "ZONE": z,
                "GR": 20 + 400 * al + 3 * rng.normal(size=n),
                "RHOB": 2.71 - 2.0 * (si - 0.08) + 4 * (mg - 0.019) + 0.02 * rng.normal(size=n),
                "NPHI": 0.12 + 0.5 * (si - 0.08) + 0.01 * rng.normal(size=n),
                "DT": 60 + 50 * (si - 0.08) - 300 * (mg - 0.019) + rng.normal(size=n),
                "PE": 5.0 - 10 * (si - 0.08) - 15 * (mg - 0.019) + 0.1 * rng.normal(size=n),
                "NMRT_FINAL": 0.14 + 0.02 * lat2 + 0.01 * rng.normal(size=n),
                "T2LM": 50 + 10 * lat2 + 3 * rng.normal(size=n),
                "MPERM": np.exp(2 + lat2 + 0.3 * rng.normal(size=n)),
                "HFK": 0.3 + 30 * al + 0.05 * rng.normal(size=n),
                "HTHO": 1 + 100 * al + 0.2 * rng.normal(size=n),
                "HURA": 2 + 0.5 * rng.normal(size=n),
                "DWCA": ca, "DWSI": si, "DWAL": al, "DWMG": mg,
                "DWFE": np.clip(0.01 + 0.3 * al + 0.002 * rng.normal(size=n), 1e-4, None),
                "DWTI": np.clip(0.001 + 0.05 * al + 0.0002 * rng.normal(size=n), 1e-5, None),
            }
            d["NMRE_FINAL"] = d["NMRT_FINAL"] * 0.85
            d["NMRFL_FINAL"] = d["NMRE_FINAL"] * 0.6
            rows.append(pd.DataFrame(d))
            depth0 += n * 0.1524
    df = pd.concat(rows, ignore_index=True)
    # Wells 6 and 7 without measured DWMG (application cases)
    df.loc[df["WELL_NAME"].isin(["W06", "W07"]), "DWMG"] = np.nan
    return df
