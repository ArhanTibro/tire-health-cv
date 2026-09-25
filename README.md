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

## Setup (local to install packages)


```bash
python -m venv venv
venv\Scripts\activate on Windows to activate the environment and for mac : source venv/bin/activate 
python -m pip install -r requirements.txt
```

Then follow `data/README.md` to get the dataset onto disk.


## Workflow

1. `notebooks/01_data_prep.ipynb` — split dataset into train/val/test, sanity-check class balance
2. `notebooks/02_cnn_baseline.ipynb` — train the from-scratch CNN
3. `notebooks/03_yolo_transfer.ipynb` — fine-tune YOLO11-cls (transfer learning)
4. `notebooks/04_compare_results.ipynb` — load both models' metrics from `results/` and compare

## Team workflow

We **do not work directly on `main`**. `main` is always the clean and final version.



1. **Update `dev` before starting:**

   ```bash
   git checkout dev
   git pull origin dev
   ```

2. **Create your own branch from `dev`:**

   ```bash
   git checkout -b <your branch name>
   ```

   Examples: `arhan-cnn-training`, `arnob-data-prep`, `yolo-tuning`

3. **Work and commit regularly:**

   ```bash
   git add .
   git commit -m "Short description of what changed"
   ```

4. **Push your branch:**

   ```bash
   git push -u origin <your branch name>
   ```

5. **Open a Pull Request:**
   `your-branch → dev` **(never directly to `main`)**.
   Have at least one teammate review it when possible.

6. **After merging into `dev`:**
   Test the complete project to ensure everyone's changes work together.


## Stay updated with dev branch when working on you own branch :

  ```bash
  git checkout <your branch name>
  git pull origin dev
  ```
