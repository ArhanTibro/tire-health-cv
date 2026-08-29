# Dataset

The dataset lives in the shared Google Drive folder `softcom_dataset` and is
NOT part of this git repo (image files don't belong in git — bloats the repo
and makes clones slow).

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
2. Place it at `data/` in this repo (this path is gitignored — it stays local only).
3. Run `notebooks/01_data_prep.ipynb` to generate the stratified train/val/test split at
   `data/processed/` (also gitignored).


