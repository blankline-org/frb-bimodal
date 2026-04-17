"""Bimodality tests on the upward-drifting drift-rate distribution.

Tests whether the upward-drifting bursts of FRB 20240114A form a single
continuous distribution or separate into two discrete modes. Applies four
independent diagnostics — Gaussian-mixture BIC comparison (one vs two vs
three components), Ashman's D for mode separation, the bimodality
coefficient, and a gap analysis on the sorted drift-rate sequence — and
renders a six-panel summary figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results")


def load_data():
    path = Path("data/raw/FRB20240114A_Morphology_Public_Dataset_20240312CST/All_Drifting_Burst-Cluster_Table.xlsx")
    df = pd.read_excel(path)
    df = df.sort_values('MJD_of_Burst-Clusters').reset_index(drop=True)
    df['drift_type'] = df['Morphology_DU'].str[0]
    df = df[df['drift_type'].isin(['U', 'D'])].reset_index(drop=True)
    return df


def select_u1(df):
    """Return the single-component upward-drifting subsample.

    Restricting to U1 bursts removes the definitional ambiguity of multi-
    component drift-rate measurements: every U1 drift rate refers to the
    intrinsic frequency-time slope of a single burst rather than to the
    apparent slope defined by sub-burst separations.
    """
    return df[df['Morphology_DU'] == 'U1'].copy()


def test_bimodality(data, name):
    """GMM, Ashman's D, and bimodality coefficient for a one-dimensional sample."""
    print(f"\n{'='*60}")
    print(f"Bimodality diagnostics: {name}")
    print(f"{'='*60}")

    data = data[~np.isnan(data)]
    n = len(data)
    print(f"N = {n}")

    from sklearn.mixture import GaussianMixture

    gmm1 = GaussianMixture(n_components=1, random_state=42)
    gmm1.fit(data.reshape(-1, 1))
    bic1 = gmm1.bic(data.reshape(-1, 1))
    aic1 = gmm1.aic(data.reshape(-1, 1))

    gmm2 = GaussianMixture(n_components=2, random_state=42)
    gmm2.fit(data.reshape(-1, 1))
    bic2 = gmm2.bic(data.reshape(-1, 1))
    aic2 = gmm2.aic(data.reshape(-1, 1))

    gmm3 = GaussianMixture(n_components=3, random_state=42)
    gmm3.fit(data.reshape(-1, 1))
    bic3 = gmm3.bic(data.reshape(-1, 1))
    aic3 = gmm3.aic(data.reshape(-1, 1))

    print(f"\nGaussian mixture model selection:")
    print(f"  1 component:  BIC={bic1:.1f}, AIC={aic1:.1f}")
    print(f"  2 components: BIC={bic2:.1f}, AIC={aic2:.1f}")
    print(f"  3 components: BIC={bic3:.1f}, AIC={aic3:.1f}")

    best_n = np.argmin([bic1, bic2, bic3]) + 1
    print(f"  Preferred model by BIC: {best_n} component(s).")

    delta_bic = bic1 - bic2
    print(f"  Delta BIC (1 vs 2): {delta_bic:.1f}")
    if delta_bic > 10:
        bimodal_evidence = "STRONG"
    elif delta_bic > 6:
        bimodal_evidence = "MODERATE"
    elif delta_bic > 2:
        bimodal_evidence = "WEAK"
    else:
        bimodal_evidence = "NONE"
    print(f"  Evidence for two components (Kass & Raftery 1995 scale): {bimodal_evidence.lower()}.")

    if best_n >= 2:
        means = gmm2.means_.flatten()
        stds = np.sqrt(gmm2.covariances_.flatten())

        d = abs(means[0] - means[1]) / np.sqrt((stds[0]**2 + stds[1]**2) / 2)
        print(f"\n  Ashman's D = {d:.2f} (D > 2 indicates clearly separated modes).")

        if d > 2:
            separation = "CLEAR"
        else:
            separation = "OVERLAPPING"
    else:
        separation = "N/A"
        d = 0

    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    bc = (skewness**2 + 1) / (kurtosis + 3)

    print(f"\n  Bimodality coefficient = {bc:.3f} (threshold 0.555).")
    bc_bimodal = bc > 0.555

    return {
        'n': n,
        'best_n_components': best_n,
        'delta_bic': delta_bic,
        'bimodal_evidence': bimodal_evidence,
        'ashman_d': d,
        'separation': separation,
        'bimodality_coef': bc,
        'bc_bimodal': bc_bimodal,
        'gmm2': gmm2
    }


def test_gap_statistic(data, name):
    """Locate the largest gap in the sorted sample and compare it to the gap distribution."""
    print(f"\n{'='*60}")
    print(f"Gap analysis: {name}")
    print(f"{'='*60}")

    data = np.sort(data[~np.isnan(data)])
    gaps = np.diff(data)

    n_top = 5
    top_gap_indices = np.argsort(gaps)[-n_top:][::-1]

    print(f"\nFive largest gaps between consecutive values:")
    for i, idx in enumerate(top_gap_indices):
        gap_size = gaps[idx]
        left_val = data[idx]
        right_val = data[idx + 1]
        z_score = (gap_size - np.mean(gaps)) / np.std(gaps)
        print(f"  {i+1}. {left_val:.1f} -> {right_val:.1f}  (gap={gap_size:.1f}, z={z_score:.1f})")

    largest_gap = gaps[top_gap_indices[0]]
    z_largest = (largest_gap - np.mean(gaps)) / np.std(gaps)

    print(f"\nLargest gap z-score: {z_largest:.1f} (relative to the mean gap).")
    gap_significant = z_largest > 3

    return {
        'largest_gap': largest_gap,
        'z_score': z_largest,
        'gap_significant': gap_significant,
        'gap_location': data[top_gap_indices[0]]
    }


def create_final_verdict_plot(df, up_bimodal_result, gap_result):
    """Assemble the six-panel bimodality summary figure."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    up_drift = select_u1(df)['Rd(MHz/ms)'].dropna().values
    down_drift = df.loc[df['drift_type'] == 'D', 'Rd(MHz/ms)'].dropna().values

    ax1 = axes[0, 0]
    ax1.hist(up_drift, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')

    x = np.linspace(up_drift.min(), up_drift.max(), 200)
    gmm = up_bimodal_result['gmm2']

    log_prob = gmm.score_samples(x.reshape(-1, 1))
    ax1.plot(x, np.exp(log_prob), 'r-', linewidth=2, label='2-component GMM')

    for i in range(2):
        mean = gmm.means_[i, 0]
        std = np.sqrt(gmm.covariances_[i, 0, 0])
        weight = gmm.weights_[i]
        component = weight * stats.norm.pdf(x, mean, std)
        ax1.plot(x, component, '--', linewidth=1.5, label=f'Component {i+1}')
        ax1.axvline(x=mean, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Drift rate (MHz/ms)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'U1 drift rates with 2-component GMM fit\n'
                  rf'$\Delta\mathrm{{BIC}}={up_bimodal_result["delta_bic"]:.1f}$')
    ax1.legend()

    ax2 = axes[0, 1]
    stats.probplot(up_drift, dist="norm", plot=ax2)
    ax2.set_title('Q-Q plot against a normal distribution (U1)')

    ax3 = axes[0, 2]
    sorted_drift = np.sort(up_drift)
    gaps = np.diff(sorted_drift)
    ax3.scatter(sorted_drift[:-1], gaps, alpha=0.5, s=20)
    ax3.axhline(y=gap_result['largest_gap'], color='red', linestyle='--',
                label=f'Largest gap (z={gap_result["z_score"]:.1f})')
    ax3.set_xlabel('Drift rate (MHz/ms)')
    ax3.set_ylabel('Gap to next value')
    ax3.set_title('Gap spectrum of the sorted sample')
    ax3.legend()

    ax4 = axes[1, 0]
    kde_up = gaussian_kde(up_drift)
    kde_down = gaussian_kde(down_drift)
    x_up = np.linspace(up_drift.min(), up_drift.max(), 200)
    x_down = np.linspace(down_drift.min(), down_drift.max(), 200)
    ax4.fill_between(x_up, kde_up(x_up), alpha=0.5, color='blue', label='Upward (U1)')
    ax4.fill_between(x_down, kde_down(x_down), alpha=0.5, color='red', label='Downward')
    ax4.set_xlabel('Drift rate (MHz/ms)')
    ax4.set_ylabel('Density')
    ax4.set_title('Kernel density: U1 upward vs downward populations')
    ax4.legend()

    ax5 = axes[1, 1]
    ax5.plot(np.sort(up_drift), np.linspace(0, 1, len(up_drift)), 'b-', linewidth=2, label='Upward (U1)')
    ax5.plot(np.sort(down_drift), np.linspace(0, 1, len(down_drift)), 'r-', linewidth=2, label='Downward')
    ax5.set_xlabel('Drift rate (MHz/ms)')
    ax5.set_ylabel('Cumulative probability')
    ax5.set_title('Empirical cumulative distributions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = axes[1, 2]
    ax6.axis('off')

    evidence_for = []
    evidence_against = []

    if up_bimodal_result['delta_bic'] > 6:
        evidence_for.append(f"BIC favours 2 components (Delta BIC = {up_bimodal_result['delta_bic']:.1f})")
    else:
        evidence_against.append("BIC does not favour bimodality")

    if up_bimodal_result['ashman_d'] > 2:
        evidence_for.append(f"Ashman's D = {up_bimodal_result['ashman_d']:.2f} (> 2)")
    else:
        evidence_against.append(f"Ashman's D = {up_bimodal_result['ashman_d']:.2f} (< 2)")

    if up_bimodal_result['bc_bimodal']:
        evidence_for.append(f"Bimodality coefficient = {up_bimodal_result['bimodality_coef']:.3f} (> 0.555)")
    else:
        evidence_against.append(f"Bimodality coefficient = {up_bimodal_result['bimodality_coef']:.3f} (< 0.555)")

    if gap_result['gap_significant']:
        evidence_for.append(f"Largest gap at z = {gap_result['z_score']:.1f}")
    else:
        evidence_against.append("No gap exceeds z = 3 in the sorted sample")

    verdict_text = "Bimodality diagnostics\n" + "-" * 40 + "\n\n"

    verdict_text += "Consistent with two components:\n"
    if evidence_for:
        for e in evidence_for:
            verdict_text += f"  - {e}\n"
    else:
        verdict_text += "  (none)\n"

    verdict_text += "\nConsistent with a single component:\n"
    if evidence_against:
        for e in evidence_against:
            verdict_text += f"  - {e}\n"
    else:
        verdict_text += "  (none)\n"

    score = len(evidence_for) - len(evidence_against)
    verdict_text += "\n" + "-" * 40 + "\n"
    verdict_text += f"Score (supporting minus opposing): {score:+d}\n"

    ax6.text(0.05, 0.95, verdict_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Bimodality verification for the U1 drift-rate distribution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'verify_bimodality.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {OUTPUT_DIR / 'verify_bimodality.png'}.")

    return score


def main():
    print("=" * 70)
    print("Bimodality verification for FRB 20240114A U1 drift rates")
    print("=" * 70)

    df = load_data()
    u1 = select_u1(df)
    up_drift = u1['Rd(MHz/ms)'].dropna().values

    print(f"\nU1 (single-component upward) sample: N = {len(up_drift)}.")
    print(f"Drift-rate range: [{up_drift.min():.1f}, {up_drift.max():.1f}] MHz/ms.")
    print(f"Mean = {up_drift.mean():.1f}, median = {np.median(up_drift):.1f}, "
          f"std = {up_drift.std():.1f} MHz/ms.")

    bimodal_result = test_bimodality(up_drift, "U1 drift rates")

    gap_result = test_gap_statistic(up_drift, "U1 drift rates")

    down_drift = df.loc[df['drift_type'] == 'D', 'Rd(MHz/ms)'].dropna().values
    down_bimodal = test_bimodality(down_drift, "downward drift rates (comparison)")

    score = create_final_verdict_plot(df, bimodal_result, gap_result)

    print("\n" + "=" * 70)
    print("Summary (U1 sample)")
    print("=" * 70)
    print(f"\nDelta BIC (1 vs 2 components): {bimodal_result['delta_bic']:.1f}")
    print(f"Ashman's D:                     {bimodal_result['ashman_d']:.2f}")
    print(f"Bimodality coefficient:         {bimodal_result['bimodality_coef']:.3f}")
    print(f"Largest-gap z-score:            {gap_result['z_score']:.1f}")
    print(f"Supporting diagnostics minus opposing: {score:+d}")
    print(f"\nResults written to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
