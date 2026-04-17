# Bimodal Drift Rate Structure in FRB 20240114A

Code and data to reproduce the discovery of bimodal drift rate structure in the upward-drifting population of Fast Radio Burst source FRB 20240114A.

**Blog post:** [blankline.org/research/bimodal-drift-rates](https://www.blankline.org/research/bimodal-drift-rates)
**Built with:** [Primus v0.2](https://www.blankline.org/research/primus)
**Author:** Santosh Arron, Blankline (2026)

---

## What this is

Unsupervised clustering of 978 burst clusters from FRB 20240114A (Zhang et al. 2026, FAST telescope) identifies a previously unreported subpopulation of 45 "extreme-drift" bursts within the upward-drifting class. In the single-component (U1) subsample, the drift-rate distribution is bimodal at 9.2 sigma with modes at 113 and 300 MHz/ms, surviving every robustness test applied.

## Reproduce in one command

```bash
git clone https://github.com/blankline-org/frb-bimodal.git
cd frb-bimodal
pip install -r requirements.txt
python reproduce.py
```

This will regenerate every figure and statistical result under `results/`. Runtime is under five minutes on a laptop.

Reference figures (committed) are in `figures/` for comparison.

## What each script does

| Script | Purpose | Output |
|---|---|---|
| `discover_cluster.py` | UMAP + HDBSCAN on 8-feature space; identifies cluster C1 | `discover_cluster.png` |
| `characterize_cluster.py` | Compares C1 against rest-of-upward on each feature | `characterize_cluster.png` |
| `robustness.py` | 6x6 UMAP+HDBSCAN parameter grid, bootstrap 100 samples | `robustness.png` |
| `verify_bimodality.py` | GMM-BIC, Ashman's D, gap analysis on U1-only subset | `verify_bimodality.png` |

## Data

`data/raw/FRB20240114A_Morphology_Public_Dataset_20240312CST/All_Drifting_Burst-Cluster_Table.xlsx`

Published by Zhang et al. (2026), [DOI 10.3847/1538-4357/ae314a](https://doi.org/10.3847/1538-4357/ae314a), hosted on Science Data Bank (scidb.cn). Redistributed here under the dataset's public-release terms for reproducibility.

## Headline results

| Metric | Value |
|---|---|
| Burst clusters analysed | 978 (233 upward, 142 U1) |
| Cluster C1 size | 45 burst clusters, 100% U1 |
| C1 drift rate vs rest-of-upward | 245.6 vs 98.1 MHz/ms (p = 1.8e-5) |
| U1-only bimodality BIC (two vs one Gaussian) | Delta-BIC = 19.9 |
| U1-only Ashman's D | 2.71 |
| U1-only gap significance | 9.2 sigma |
| Minimum U1 drift rate | 87.1 MHz/ms (all above error floor) |
| UMAP parameter robustness | 6/6 configurations |
| HDBSCAN parameter robustness | 6/6 configurations |
| Bootstrap reproducibility | 98/100 resamples |
| Decorrelated-feature robustness | 4/4 variants (100% upward purity) |

## Method

Eight spectrotemporal features per burst cluster: bandwidth, effective width, peak frequency, drift rate, energy, flux, S/N, centre frequency. HDBSCAN (min_cluster_size=15, min_samples=5) identifies clusters directly on the standardised 8-dimensional feature space, without specifying their number. UMAP (n_neighbours=15, min_dist=0.1) projects the same features to 2D for post-hoc visualisation only. A second, independent analysis restricted to U1 (single-component) bursts confirms the bimodality in the cleanest subsample where every drift rate exceeds its measurement uncertainty.

## Citation

```
Arron, S. (2026). Discovery of bimodal drift rate structure in FRB 20240114A:
evidence for dual emission regions. Blankline.
https://www.blankline.org/research/bimodal-drift-rates
```

## Contact

research@blankline.org

## License

Code is released under the MIT License; see [`LICENSE`](LICENSE).

The dataset under `data/raw/` is redistributed under the terms of its original public release by Zhang et al. (2026). It is not covered by the MIT License.
