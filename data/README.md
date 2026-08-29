# Dataset

The raw dataset lives in the shared Google Drive folder `softcom_dataset`, not in this repo
(image binaries don't belong in git — bloats the repo and makes clones slow).

**Drive folder:** softcom_dataset (ask a teammate for the share link if you don't have it)

## Structure (as collected)

```
softcom_dataset/
    healthy/
    low_tread/
    zero_tread/
    sidewall_damaged/
    uneven_wear/
    expired/          # separate OCR-based flag, not part of the 5-class visual split
```

## To use locally

1. Download `softcom_dataset/` from Drive.
2. Place it at `data/raw/` in this repo (this path is gitignored — it stays local only).
3. Run `notebooks/01_data_prep.ipynb` to generate the stratified train/val/test split at
   `data/processed/` (also gitignored).

## To use on Colab

Mount Drive directly in the notebook and point `DATA_DIR` at the `softcom_dataset` folder —
no need to download anything.

## Naming convention

`<class>_<collector-initials>_<number>.<ext>`, e.g. `healthy_AR_001.jpeg`. See
`src/data/rename_images.py` to enforce this on a folder of newly added images.
