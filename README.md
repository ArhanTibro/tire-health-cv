# Tire Health & Expiry Assessment System (Computer Vision)

Deep Learning-Based Tire Health and Expiry Assessment System Using Computer Vision.
Classifies tires into 5 visual condition classes and separately flags expiration via OCR
on the DOT/manufacturing-date code.

## Classes
- Healthy
- Zero Tread
- Low Tread
- Sidewall Damaged
- Uneven Wear

(Expiration is a separate binary flag, not a 6th visual class — see `src/ocr/` once added.)

## Project structure

```
data/        Dataset (not committed to git — see data/README.md)
notebooks/   Thin notebooks that call into src/ — runnable locally or on Colab
src/         Reusable training/eval code, shared by notebooks and scripts
configs/     YAML configs for each model (hyperparameters)
results/     Metrics/plots per model run, for side-by-side comparison
```

## Setup (local, RTX 3050 / any CUDA GPU)

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Then follow `data/README.md` to get the dataset onto disk.

## Setup (Colab, no local GPU needed)

1. Open the relevant notebook in `notebooks/` via Colab.
2. Run the first cell (clones this repo + installs requirements).
3. Mount Google Drive and point `DATA_DIR` at the shared dataset folder.
4. Run the rest of the notebook — it calls the same `src/` code as local runs.

## Workflow

1. `notebooks/01_data_prep.ipynb` — split dataset into train/val/test, sanity-check class balance
2. `notebooks/02_cnn_baseline.ipynb` — train the from-scratch CNN
3. `notebooks/03_yolo_transfer.ipynb` — fine-tune YOLO11-cls (transfer learning)
4. `notebooks/04_compare_results.ipynb` — load both models' metrics from `results/` and compare

## Team workflow

- `main` stays stable. Work on a feature branch (`git checkout -b your-feature`), open a PR to merge back.
- Each model run should save its metrics to `results/<model_name>/` in the same format (see `src/utils/metrics.py`) so results are directly comparable.
