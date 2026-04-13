# MaNGA Galaxy Spectral Clustering

A pipeline for downloading, preprocessing, and unsupervised clustering of MaNGA integral field spectroscopy data from the SDSS DR17 survey. Galaxies are represented as radially stacked spectra, reduced via PCA and UMAP, and clustered using HDBSCAN with Bayesian hyperparameter optimization.

---

## Overview

The pipeline consists of two stages:

1. **Preprocessing** (`cubes.py`) — Downloads MaNGA data cubes and DAP maps, stacks spaxel spectra into radial bins (in units of effective radius), reprojects onto a common wavelength grid, and saves the results as FITS tables.

2. **Clustering** (`clustering.py`) — Loads the preprocessed spectra, builds feature vectors, applies StandardScaler → PCA → UMAP for dimensionality reduction, and then runs HDBSCAN. Hyperparameters for UMAP and HDBSCAN are jointly optimized using Gaussian process minimization of the DBCV validity score.

---

## Requirements

- Python 3.8+
- [Marvin](https://sdss-marvin.readthedocs.io/) (SDSS MaNGA access)
- [specutils](https://specutils.readthedocs.io/)
- [astropy](https://www.astropy.org/)
- [umap-learn](https://umap-learn.readthedocs.io/)
- [hdbscan](https://hdbscan.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)
- [scikit-optimize](https://scikit-optimize.github.io/)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [tqdm](https://tqdm.github.io/)

Install dependencies:

```bash
pip install marvin specutils astropy umap-learn hdbscan scikit-learn scikit-optimize matplotlib numpy tqdm
```

---

## Data

This pipeline uses SDSS DR17 MaNGA data:

- **DRP cubes** (`LOGCUBE.fits.gz`) — raw spectral data cubes
- **DAP maps** (`MAPS-HYB10-MILESHC-MASTARHC2.fits.gz`) — derived analysis maps including elliptical coordinate maps used for radial binning
- **DRPall catalog** (`drpall-v3_1_1.fits`) — survey-wide catalog used to select the main MaNGA sample and retrieve NSA photometry

The sample is selected from the MaNGA primary, secondary, and color-enhanced targets at redshift `z < 0.1`.

---

## Usage

### Step 1: Preprocess data cubes

```bash
python cubes.py
```

This will download MaNGA cubes and maps directly from the SDSS SAS, preprocess them, and write the radially stacked spectra to `preprocessed_spectra/<plateifu>.fits`. Already-processed galaxies are skipped automatically.

Set `download=False` in `preprocess_all_cubes()` to use locally cached Marvin data instead.

Output directory structure expected:
```
cubes/
maps/
preprocessed_spectra/
plots/
```

### Step 2: Cluster the spectra

```bash
python clustering.py
```

By default, this runs Bayesian hyperparameter optimization (50 evaluations) over the UMAP `n_neighbors` and HDBSCAN `min_cluster_size` / `min_samples` parameters, maximizing the DBCV relative validity score.

To skip optimization and run clustering directly with fixed parameters, comment out the `parameter_optimization` call and uncomment the `run_umap` / `run_hdbscan` / `plot_umap` block at the bottom of the script.

---

## Pipeline Details

### Radial stacking (`cubes.py`)

Each galaxy cube is binned into six radial annuli spanning 0–1.5 $R_e$ in steps of 0.25 $R_e$. Spaxels flagged as bad by the DRP bitmask are excluded. The rest-frame spectra are interpolated onto a common wavelength grid from 3700–9400 Å.

### Feature vectors (`clustering.py`)

Each annular spectrum is normalized by its median flux in the 5500–5800 Å window. The six normalized spectra are concatenated into a single feature vector per galaxy. Vectors where any annulus fails normalization (zero flux) are discarded.

### Dimensionality reduction

Features are standardized, then reduced to 50 PCA components (retaining the bulk of variance), and optionally further reduced with UMAP prior to clustering.

### Clustering & optimization

HDBSCAN is run on the UMAP embedding. The objective function for optimization is the negative DBCV score (`relative_validity_`). The search space covers:

| Parameter | Range |
|---|---|
| `min_cluster_size` | 5–30 |
| `min_samples` | 5–30 |
| `n_neighbors` (UMAP) | 5–50 |

A constraint enforces `min_cluster_size ≤ min_samples`.

---

## Output

- `preprocessed_spectra/` — one FITS file per galaxy with columns `wav`, `flux1`–`flux6`
- `plots/umap.pdf` — 2D UMAP projection colored by cluster label; noise points shown in gray

---

## Notes

- The absolute magnitude feature (`nsa_elpetro_absmag` r-band) is present in the code but commented out. It can be re-enabled in `create_feature_vectors()` and the corresponding `feat` array size adjusted.
- The `min_dist` and `n_components` UMAP parameters are also present as commented-out optimization dimensions and can be included in the search space if desired.
- Marvin must be configured with valid SAS credentials for remote access. See the [Marvin documentation](https://sdss-marvin.readthedocs.io/en/stable/installation.html) for setup.
