# cgan-ecs-mg-presalt

Code and data for the article *Reconstructing elemental capture spectroscopy magnesium logs in pre-salt carbonates: a leave-one-well-out comparison of a geochemically informed conditional GAN and random forests* (submitted to Applied Computing and Geosciences).

The repository reconstructs the ECS magnesium log (DWMG) of the Barra Velha Formation, Sapinhoá Field, Santos Basin, and compares random forests with a geochemically informed conditional GAN under leave-one-well-out validation inside each stratigraphic zone.

## Contents

```
chemo_exp/            Python package
  data.py             LAS loading, zone assignment, depth windows, synthetic test data
  models.py           Random forest models and the cGAN (1D U-Net generator, 1D PatchGAN discriminator)
  run.py              Main experiment: leave-one-well-out comparison per zone
  controls.py         Controls: predictor set, training wells and forest settings, cGAN-B
  depth_cases.py      Rerun of cGAN-B on three wells for the depth figure
  figures.py          Figures and tables of the article
  fig_arch.py         cGAN architecture diagram
data/
  las/                28 well files (LAS 2.0) of the Sapinhoá Field
  zone_tops.xlsx      Tops and bases of zones A, B1, B2 and C per well (columns Well, Top, Bottom, Zone)
results/              Per-well metrics and predictions used in the article
figures/              Figures of the article (PDF)
```

## Data

Each LAS file contains conventional logs (GR, RHOB, NPHI, DT, PE), spectral gamma ray (HFK, HTHO, HURA), NMR logs (NMRT_FINAL, NMRE_FINAL, NMRFL_FINAL, T2LM, MPERM) and ECS dry-weight fractions (DWCA, DWSI, DWAL, DWFE, DWTI and, in 13 wells, DWMG). Missing values are coded as -9999. Zero values in DWMG and DWAL are values below the tool resolution and are kept as observations.

## Installation

```bash
pip install -r requirements.txt
```

The experiments were run with Python 3.13, PyTorch 2.15 (CPU), scikit-learn 1.9 and pandas 3.0.

## Reproducing the article

Quick test with synthetic data (a few minutes):

```bash
python -m chemo_exp.run --synthetic --epochs 5 --out runs/synthetic
```

Main experiment (RF-II, RF-F, RF-F-ctx, cGAN-F, cGAN-F-stoch, cGAN-F-noPhys and CNN-F-L1), 31 folds:

```bash
python -m chemo_exp.run --las-dir data/las --zones data/zone_tops.xlsx --zone-list A,B1,B2 --out runs/main
```

Controls (RF-B, RF-B-narrow, RF-F20, RF-B300 and cGAN-B):

```bash
python -m chemo_exp.controls --las-dir data/las --zones data/zone_tops.xlsx --out runs/controls
```

Depth figure cases and figures:

```bash
python -m chemo_exp.depth_cases --las-dir data/las --zones data/zone_tops.xlsx --out runs
python -c "from chemo_exp import data as D; D.load_wells('data/las','data/zone_tops.xlsx').to_pickle('df.pkl')"
python -m chemo_exp.figures runs/main/lowo_per_well.csv runs/main/lowo_predictions.csv df.pkl figures runs/controls/controls_per_well.csv runs/depth_cgan_b.csv
python chemo_exp/fig_arch.py figures
```

To redraw the figures from the published results without rerunning the models, replace the `runs/...` paths with the files in `results/`.

With 80 epochs on a CPU, each cGAN fold takes about 2 minutes per variant; the full main experiment took about 3 hours on two cores. Results can vary slightly with the number of CPU threads, because the order of floating-point operations changes.

## Result files

| File | Content |
|---|---|
| `results/main_per_well.csv` | R², RMSE and MAE per held-out well and model, main experiment |
| `results/main_predictions.csv` | Measured and reconstructed DWMG per depth, main experiment |
| `results/controls_per_well.csv` | R², RMSE and MAE per held-out well, controls and cGAN-B |
| `results/shape_offset_per_well.csv` | Per-well correlation (shape) and mean error (offset) |
| `results/depth_cgan_b.csv` | cGAN-B reconstructions used in the depth figure |

Model names follow the article: `F` is the full set of 14 predictors (GR, RHOB, NPHI, DT, PE, NMRT, NMRE, NMRFL, HFK, HTHO, HURA, DWCA, DWSI, DWAL) and `B` the basic set (GR, RHOB, NPHI, DT, DWCA, DWSI, DWAL) trained on every well that has it.

## Citation

See `CITATION.cff`. The full reference will be added after publication.

## License

MIT (see `LICENSE`).
