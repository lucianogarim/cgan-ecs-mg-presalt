"""Compared models: random forests and the geochemically informed conditional GAN."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


# ==========================================================================
# RandomForest
# ==========================================================================
def rf_iterative(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], target: str,
                        seed: int = 42) -> np.ndarray:
    """Iterative imputer with a 20-tree random forest (RF-II).

    The held-out rows enter the fit with the target masked; since only the target
    is missing, this reduces to a single forest fitted on the training rows.
    """
    te = test[cols].copy()
    te[target] = np.nan
    stacked = pd.concat([train[cols], te])
    imp = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=20, n_jobs=-1, random_state=seed),
        max_iter=5, min_value=1e-5, random_state=seed)
    out = pd.DataFrame(imp.fit_transform(stacked), columns=cols, index=stacked.index)
    return out.loc[test.index, target].to_numpy()


def rf_direct(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], target: str,
              seed: int = 42, n_estimators: int = 300) -> np.ndarray:
    """Pointwise random forest regression fitted on the training wells only."""
    rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3, max_features=0.5,
                               n_jobs=-1, random_state=seed)
    rf.fit(train[feats].to_numpy(), train[target].to_numpy())
    return rf.predict(test[feats].to_numpy())


def add_context(df: pd.DataFrame, feats: list[str], half: int = 20) -> tuple[pd.DataFrame, list[str]]:
    """Moving mean and standard deviation per well (window of 2*half+1 = 41 samples, as in the cGAN)."""
    df = df.sort_values(["WELL_NAME", "DEPT"]).copy()
    new = []
    g = df.groupby("WELL_NAME")[feats]
    for k, fn in (("m", "mean"), ("s", "std")):
        r = g.rolling(2 * half + 1, center=True, min_periods=half + 1).agg(fn)
        r = r.reset_index(level=0, drop=True)
        for c in feats:
            df[f"{c}_{k}{2*half+1}"] = r[c]
            new.append(f"{c}_{k}{2*half+1}")
    return df, feats + new


# ==========================================================================
# cGAN
# ==========================================================================
@dataclass
class CGANConfig:
    L: int = 41
    z_dim: int = 16
    epochs: int = 300
    batch: int = 128
    lr: float = 2e-4
    lambda_l1: float = 80.0
    mu_tv: float = 2.0
    w_comp: float = 0.5
    w_corr: float = 0.2
    corr_thr: float = 0.3
    k_stoch: int = 5
    alpha_resid: float = 0.6
    adversarial: bool = True
    physics: bool = True
    seed: int = 42
    threads: int = 2
    log_every: int = 0
    history: list = field(default_factory=list)


def _build_nets(C: int, z: int):
    import torch.nn as nn

    class Gen(nn.Module):
        def __init__(self):
            super().__init__()
            def blk(i, o, s=1):
                return nn.Sequential(nn.Conv1d(i, o, 5, s, 2), nn.BatchNorm1d(o), nn.LeakyReLU(0.2))
            self.e1 = blk(C + z, 64)
            self.e2 = blk(64, 128, 2)
            self.e3 = blk(128, 128, 2)
            self.u2 = nn.Sequential(nn.Upsample(scale_factor=2), blk(128, 128))
            self.d2 = blk(256, 128)
            self.u1 = nn.Sequential(nn.Upsample(scale_factor=2), blk(128, 64))
            self.d1 = blk(128, 64)
            self.out = nn.Sequential(nn.Conv1d(64, 1, 3, 1, 1), nn.Sigmoid())

        def forward(self, x, zz):
            import torch
            import torch.nn.functional as F
            h = torch.cat([x, zz.unsqueeze(-1).expand(-1, -1, x.shape[-1])], 1)
            e1 = self.e1(h); e2 = self.e2(e1); e3 = self.e3(e2)
            u2 = self.u2(e3); u2 = F.interpolate(u2, size=e2.shape[-1])
            d2 = self.d2(torch.cat([u2, e2], 1))
            u1 = self.u1(d2); u1 = F.interpolate(u1, size=e1.shape[-1])
            d1 = self.d1(torch.cat([u1, e1], 1))
            return self.out(d1)

    class Disc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(C + 1, 64, 5, 2, 2), nn.LeakyReLU(0.2),
                nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
                nn.Conv1d(128, 1, 3, 1, 1))  # PatchGAN 1D

        def forward(self, x, y):
            import torch
            return self.net(torch.cat([x, y], 1))

    return Gen(), Disc()


def _pearson_rows(a, b, eps=1e-8):
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


class CGANImputer:
    """One-dimensional conditional GAN (pix2pix) with L1, first-order TV and geochemical losses.

    aux (original units) = observed [DWCA, DWSI, DWAL] in the window; the target is DWMG.
    Penalties:
      compositional sum: relu(Ca + Mg_hat + Si + Al - M), M = largest sum in the training wells
      anticorrelation:   relu(corr(Ca + Mg_hat, Si + Al) + thr), averaged over the batch
    """

    def __init__(self, cfg: CGANConfig):
        self.cfg = cfg

    # ---- scaling statistics from the training wells only ----
    def _fit_scalers(self, X, y):
        self.xm = X.mean(axis=(0, 2), keepdims=True)
        self.xs = X.std(axis=(0, 2), keepdims=True) + 1e-6
        self.ymin, self.ymax = float(y.min()), float(y.max())

    def _sx(self, X):
        return ((X - self.xm) / self.xs).astype(np.float32)

    def _sy(self, y):
        return ((y - self.ymin) / (self.ymax - self.ymin + 1e-12)).astype(np.float32)

    def _iy(self, y):
        return y * (self.ymax - self.ymin) + self.ymin

    def fit(self, W):
        import torch
        import torch.nn.functional as F
        c = self.cfg
        torch.manual_seed(c.seed); np.random.seed(c.seed)
        torch.set_num_threads(c.threads)
        self._fit_scalers(W.X, W.y)
        X = torch.from_numpy(self._sx(W.X)); Y = torch.from_numpy(self._sy(W.y))
        A = torch.from_numpy(W.aux.astype(np.float32))
        self.M = float((W.aux.sum(1) + W.y[:, 0]).max()) if W.aux.shape[1] else np.inf
        G, D = _build_nets(X.shape[1], c.z_dim)
        oG = torch.optim.Adam(G.parameters(), c.lr, betas=(0.5, 0.999))
        oD = torch.optim.Adam(D.parameters(), c.lr, betas=(0.5, 0.999))
        bce = torch.nn.BCEWithLogitsLoss()
        n = len(X)
        for ep in range(c.epochs):
            perm = torch.randperm(n)
            agg = np.zeros(6)
            for b in range(0, n, c.batch):
                ix = perm[b:b + c.batch]
                if len(ix) < 4:
                    continue
                x, y, a = X[ix], Y[ix], A[ix]
                z = torch.randn(len(ix), c.z_dim)
                yh = G(x, z)
                if c.adversarial:
                    oD.zero_grad()
                    pr, pf = D(x, y), D(x, yh.detach())
                    ld = 0.5 * (bce(pr, torch.ones_like(pr)) + bce(pf, torch.zeros_like(pf)))
                    ld.backward(); oD.step()
                oG.zero_grad()
                l1 = F.l1_loss(yh, y)
                tv = (yh[..., 1:] - yh[..., :-1]).abs().mean()
                loss = c.lambda_l1 * l1 + c.mu_tv * tv
                ladv = torch.tensor(0.0); lcomp = torch.tensor(0.0); lcorr = torch.tensor(0.0)
                if c.adversarial:
                    pf = D(x, yh)
                    ladv = bce(pf, torch.ones_like(pf)); loss = loss + ladv
                if c.physics and a.shape[1] >= 3:
                    mg = yh[:, 0] * (self.ymax - self.ymin) + self.ymin
                    s = a.sum(1) + mg
                    lcomp = F.relu(s - self.M).mean() / max(self.M, 1e-6)
                    carb = a[:, 0] + mg
                    sil = a[:, 1] + a[:, 2]
                    lcorr = F.relu(_pearson_rows(carb, sil) + c.corr_thr).mean()
                    loss = loss + c.w_comp * lcomp + c.w_corr * lcorr
                loss.backward(); oG.step()
                agg += [loss.item(), l1.item(), tv.item(), ladv.item(), lcomp.item(), lcorr.item()]
            c.history.append(agg)
            if c.log_every and (ep + 1) % c.log_every == 0:
                print(f"  ep {ep+1}: loss={agg[0]:.3f} l1={agg[1]:.4f}")
        self.G = G.eval()
        # training residuals (scaled to [0, 1]) for the stochastic realizations
        with torch.no_grad():
            p = self._raw_predict(X)
        self.resid = (Y.numpy() - p)[:, 0].ravel()
        return self

    def _raw_predict(self, Xs):
        import torch
        with torch.no_grad():
            out = []
            for b in range(0, len(Xs), 1024):
                x = Xs[b:b + 1024]
                out.append(self.G(x, torch.zeros(len(x), self.cfg.z_dim)).numpy())
        return np.concatenate(out) if out else np.zeros((0, 1, self.cfg.L), np.float32)

    def predict_rows(self, W, stochastic: bool = False):
        """Predict per window and average the overlapping windows at each depth."""
        import torch
        p = self._raw_predict(torch.from_numpy(self._sx(W.X)))[:, 0]
        if stochastic:
            rng = np.random.default_rng(self.cfg.seed)
            reals = [np.clip(p + self.cfg.alpha_resid * rng.choice(self.resid, p.shape), 0, 1)
                     for _ in range(self.cfg.k_stoch)]
            p = np.mean(reals, axis=0)
        p = self._iy(p)
        s = pd.Series(p.ravel()).groupby(W.idx.ravel()).mean()
        return s
