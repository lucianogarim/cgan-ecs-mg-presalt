# Tutorial

Three typical use cases, from a quick check to applying the method to new wells.

## 1. Check the installation with synthetic data

The package includes a synthetic dataset (eight wells, four zones, two wells without DWMG) that runs in a few minutes on any computer.

```bash
pip install -r requirements.txt
python -m chemo_exp.run --synthetic --epochs 5 --zone-list B1 --out runs/synthetic
```

`runs/synthetic/lowo_per_well.csv` lists one row per held-out well and model. With 5 epochs the cGAN results are not converged; the run only checks that everything works.

## 2. Reconstruct DWMG in wells without it

```bash
python -m chemo_exp.impute --las-dir data/las --zones data/zone_tops.xlsx --zone-list A,B1,B2 --out runs/imputed_dwmg.csv
```

This trains one cGAN-B per zone on the 13 wells with DWMG and writes the reconstructed log of the other wells to `runs/imputed_dwmg.csv`. To use your own wells, put their LAS files in a folder, list their zone tops in a spreadsheet with the columns `Well`, `Zone`, `Top` and `Bottom`, and pass both paths. Wells without the seven logs of the basic set are skipped.

To plot one reconstructed well:

```python
import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv("runs/imputed_dwmg.csv")
w = d[d.WELL_NAME == d.WELL_NAME.iloc[0]].sort_values("DEPT")
plt.plot(w.DWMG_cGAN, w.DEPT)
plt.gca().invert_yaxis()
plt.xlabel("DWMG (dry-weight fraction)")
plt.ylabel("Depth (m)")
plt.show()
```

## 3. Evaluate the models on your own data

```bash
python -m chemo_exp.run --las-dir <your_las_folder> --zones <your_zone_tops.xlsx> --out runs/my_eval
python -m chemo_exp.controls --las-dir <your_las_folder> --zones <your_zone_tops.xlsx> --out runs/my_controls
```

Compare models with `runs/my_eval/lowo_summary.csv` and with paired tests over wells, for example:

```python
import pandas as pd
from scipy.stats import wilcoxon

r = pd.read_csv("runs/my_eval/lowo_per_well.csv")
p = r.pivot_table(index=["zone", "well"], columns="model", values="R2").dropna()
print(wilcoxon(p["cGAN-F"], p["RF-F"]))
```
