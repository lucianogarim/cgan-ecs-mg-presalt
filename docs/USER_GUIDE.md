# User guide

This guide describes the inputs, outputs, options and expected behaviour of every command in `chemo_exp`.

## Inputs

### Well files (`--las-dir`)

A folder of LAS 2.0 files, one per well. The well name is read from the `WELL` field of the LAS header. The following mnemonics are used:

| Group | Mnemonics |
|---|---|
| Conventional | GR, RHOB, NPHI, DT, PE |
| Spectral gamma ray | HFK, HTHO, HURA |
| NMR | NMRT_FINAL, NMRE_FINAL, NMRFL_FINAL (T2LM and MPERM are read but not used) |
| ECS dry-weight fractions | DWCA, DWSI, DWAL, DWMG (target), DWFE, DWTI |

Missing values must be coded as -9999 or -9999.25. Zero values in ECS logs are kept as measurements (values below the tool resolution). Depth is read from `DEPT` or `DEPTH`.

### Zone tops (`--zones`)

A spreadsheet (xlsx) with one row per well and zone and the columns `Well`, `Zone`, `Top` and `Bottom` (measured depth, same unit as the LAS files). Well names must match the `WELL` field of the LAS files. Samples outside every listed interval are labelled `Sem_Zona` and ignored.

### Predictor sets (`--featset`)

- `base4` (set B): GR, RHOB, NPHI, DT, DWCA, DWSI, DWAL. Available in more wells; best results in the article.
- `full` (set F): set B plus PE, NMRT_FINAL, NMRE_FINAL, NMRFL_FINAL, HFK, HTHO, HURA.

Only depths where all predictors are present are used. Windows of 41 samples never cross a depth gap or a zone boundary.

## Commands

### `python -m chemo_exp.impute` — reconstruct DWMG in new wells

Trains one cGAN per zone on all wells with DWMG and reconstructs DWMG in the wells without it.

| Option | Default | Meaning |
|---|---|---|
| `--las-dir`, `--zones` | | input data (or `--synthetic`) |
| `--featset` | `base4` | predictor set |
| `--zone-list` | `A,B1,B2,C` | zones to process |
| `--epochs` | 80 | training epochs per zone |
| `--stride` | 4 | stride between training windows |
| `--seed` | 42 | random seed |
| `--threads` | 2 | CPU threads for PyTorch |
| `--out` | `runs/imputed_dwmg.csv` | output file |

Output: CSV with columns `WELL_NAME`, `DEPT`, `ZONE` and `DWMG_cGAN` (dry-weight fraction). Expected run time: a few minutes per zone on a CPU.

### `python -m chemo_exp.run` — leave-one-well-out comparison

Holds out each well with DWMG once per zone and compares the models of the article on the same rows.

| Option | Default | Meaning |
|---|---|---|
| `--las-dir`, `--zones`, `--synthetic`, `--pickle` | | input data |
| `--featset` | `full` | predictor set for RF-F and the cGAN-F variants |
| `--models` | `RF-II,RF-F,RF-F-ctx,cGAN-F,cGAN-F-noPhys,CNN-F-L1` | models to run (`cGAN-F-stoch` is added with `cGAN-F`) |
| `--epochs`, `--stride`, `--seed`, `--threads` | 80, 4, 42, 2 | cGAN training |
| `--min-rows` | 100 | minimum complete samples for a well to be held out |
| `--zone-list` | `A,B1,B2,C` | zones |
| `--out` | `results` | output folder |

Outputs: `lowo_per_well.csv` (R², RMSE, MAE and number of samples per zone, held-out well and model), `lowo_summary.csv` (mean, standard deviation and median per zone and model), `lowo_predictions.csv` (measured and reconstructed DWMG per depth) and `run_args.json`.

### `python -m chemo_exp.controls` — controls

Runs RF-B, RF-B-narrow, RF-F20, RF-B300 and cGAN-B on the same folds. Options as above, plus `--cgan-epochs` and `--no-cgan` (forests only, about one minute). Output: `controls_per_well.csv`.

### `python -m chemo_exp.depth_cases`, `python -m chemo_exp.figures`, `python chemo_exp/fig_arch.py`

Reproduce the figures of the article (see the README for the exact commands).

## Expected behaviour and limitations

- A zone needs at least two wells with DWMG for leave-one-well-out validation; zone C of the Sapinhoá data has only one.
- Results vary slightly with the number of CPU threads and the random seed (per-well R² changes of up to about 0.1 were observed); compare models with the paired results over all wells rather than single wells.
- The cGAN reproduces the vertical pattern of DWMG well but does not correct well-level offsets when a well belongs to a magnesium regime that is under-represented in training.
