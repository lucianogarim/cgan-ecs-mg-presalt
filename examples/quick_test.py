"""Quick test of the full pipeline on a synthetic dataset (a few minutes on a CPU).

Run from the repository root:

    python examples/quick_test.py

It (1) runs the leave-one-well-out comparison of the random forest and cGAN
models on one zone of a synthetic dataset and (2) reconstructs DWMG in the two
synthetic wells without it. No real data are needed. The test passes if both
steps finish and write their output files.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chemo_exp import impute, run  # noqa: E402

OUT = os.path.join("runs", "quick_test")


def main():
    print("Step 1/2: leave-one-well-out comparison on synthetic zone B1 (5 epochs)")
    run.main(["--synthetic", "--zone-list", "B1", "--epochs", "5",
              "--models", "RF-II,RF-F,cGAN-F", "--out", OUT])
    per_well = pd.read_csv(os.path.join(OUT, "lowo_per_well.csv"))
    assert not per_well.empty, "no leave-one-well-out results were written"

    print("\nStep 2/2: reconstruction of DWMG in synthetic wells without it (5 epochs)")
    out_csv = os.path.join(OUT, "imputed_dwmg.csv")
    impute.main(["--synthetic", "--zone-list", "B1", "--epochs", "5", "--out", out_csv])
    imputed = pd.read_csv(out_csv)
    assert not imputed.empty, "no reconstructed samples were written"

    print("\nQuick test passed.")
    print("Median R2 per model in the held-out wells (5 epochs, not converged):")
    print(per_well.groupby("model")["R2"].median().round(3).to_string())
    print(f"Reconstructed {len(imputed)} samples in wells {sorted(imputed['WELL_NAME'].unique())}")
    print(f"Outputs in {OUT}/")


if __name__ == "__main__":
    main()
