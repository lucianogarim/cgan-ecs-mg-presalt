# cgan-ecs-mg-presalt

Code and data for the article *Reconstructing elemental capture spectroscopy magnesium logs in pre-salt carbonates: a leave-one-well-out comparison of a geochemically informed conditional GAN and random forests* (submitted to Computers & Geosciences).

## Purpose

Elemental capture spectroscopy (ECS) magnesium logs (DWMG) are missing in many wells. This code reconstructs DWMG from conventional and co-measured ECS logs with a geochemically informed conditional GAN, compares it with random forests under leave-one-well-out validation inside each stratigraphic zone, and reproduces every table and figure of the article. The data are 28 wells of the Barra Velha Formation, Sapinhoá Field, Santos Basin (Brazil).

## Installation

Python 3.10 or later is required (tested with Python 3.13).

```bash
git clone https://github.com/lucianogarim/cgan-ecs-mg-presalt
cd cgan-ecs-mg-presalt
pip install -r requirements.txt
```

## Quick test

From the repository root:

```bash
python examples/quick_test.py
```

The test uses a built-in synthetic dataset, so no real data are needed, and takes about two minutes on a CPU. It runs the leave-one-well-out comparison on one synthetic zone and then reconstructs DWMG in the two synthetic wells without it. It ends with the message `Quick test passed.` and writes its outputs to `runs/quick_test/` (`lowo_per_well.csv` with R², RMSE and MAE per held-out well and model, and `imputed_dwmg.csv` with the reconstructed log). With only five training epochs the cGAN is not converged; the test checks that the pipeline works, not the scores of the article.

## Contents

```
examples/quick_test.py  Quick test on synthetic data
chemo_exp/            Python package
  data.py             LAS loading, zone assignment, depth windows, synthetic test data
  impute.py           Reconstruct DWMG in wells where it was not measured
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
docs/USER_GUIDE.md    Inputs, outputs, options and expected behaviour of every command
docs/TUTORIAL.md      Step-by-step use cases
```

## Data

Each LAS file contains conventional logs (GR, RHOB, NPHI, DT, PE), spectral gamma ray (HFK, HTHO, HURA), NMR logs (NMRT_FINAL, NMRE_FINAL, NMRFL_FINAL, T2LM, MPERM) and ECS dry-weight fractions (DWCA, DWSI, DWAL, DWFE, DWTI and, in 13 wells, DWMG). Missing values are coded as -9999. Zero values in DWMG and DWAL are values below the tool resolution and are kept as observations.

The experiments of the article were run with Python 3.13, PyTorch 2.15 (CPU), scikit-learn 1.9 and pandas 3.0.

## Reconstructing DWMG in new wells

```bash
python -m chemo_exp.impute --las-dir data/las --zones data/zone_tops.xlsx --out runs/imputed_dwmg.csv
```

See `docs/TUTORIAL.md` for a walkthrough and `docs/USER_GUIDE.md` for all options.

## Computational requirements

All experiments run on a CPU. The synthetic test takes a few minutes; reconstructing DWMG for all zones takes a few minutes per zone; the full leave-one-well-out comparison took about 3 hours on two cores, and the controls about 1.5 hours.

## Reproducing the article

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
