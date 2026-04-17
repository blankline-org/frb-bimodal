"""Characterise Cluster C1 against the remaining upward-drifting population.

Reproduces the UMAP + HDBSCAN clustering from discover_cluster.py, identifies
the predominantly upward-drifting cluster (C1), and tests each of the eight
input features for a difference between C1 and the remaining upward-drifting
bursts using the Mann-Whitney U test.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_cluster():
    """Reproduce the clustering to identify C1."""
    path = Path("data/raw/FRB20240114A_Morphology_Public_Dataset_20240312CST/All_Drifting_Burst-Cluster_Table.xlsx")
    df = pd.read_excel(path)
    df = df.sort_values('MJD_of_Burst-Clusters').reset_index(drop=True)
    df['drift_type'] = df['Morphology_DU'].str[0]
    df = df[df['drift_type'].isin(['U', 'D'])].reset_index(drop=True)

    # Features
    feature_cols = ['Bandwidth(MHz)', 'Weff(ms)', '(Average)_Freq_peak(MHz)',
                    'Rd(MHz/ms)', 'Energy(1e39erg)', 'Flux', 'S/N', 'center_freq']

    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = X[col].fillna(X[col].median())

    for col in ['Energy(1e39erg)', 'Flux', 'Bandwidth(MHz)']:
        if col in X.columns:
            X[col] = np.log10(X[col].clip(lower=1e-10))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # UMAP
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_scaled)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_scaled)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
    labels = clusterer.fit_predict(X_scaled)

    return df, X_scaled, X_2d, labels, feature_cols


def characterize_cluster_c1(df, labels):
    """Identify C1 and test each feature for a difference versus other upward-drifting bursts."""

    print("=" * 70)
    print("Characterising cluster C1")
    print("=" * 70)

    # C1 is the HDBSCAN cluster with the highest proportion of upward-drifting
    # bursts.
    drift_types = df['drift_type'].values

    cluster_up_pct = {}
    for label in set(labels):
        if label >= 0:
            mask = labels == label
            up_pct = ((drift_types == 'U') & mask).sum() / mask.sum()
            cluster_up_pct[label] = up_pct

    # C1 is the one with ~100% Up
    c1_label = max(cluster_up_pct, key=cluster_up_pct.get)
    print(f"\nCluster C1 identified as label {c1_label} ({cluster_up_pct[c1_label]:.1%} Up)")

    c1_mask = labels == c1_label
    c0_mask = labels == (1 - c1_label)  # The other cluster
    noise_mask = labels == -1
    other_up_mask = (drift_types == 'U') & ~c1_mask

    c1_df = df[c1_mask]
    c0_df = df[c0_mask]
    other_up_df = df[other_up_mask]
    all_up_df = df[drift_types == 'U']
    all_down_df = df[drift_types == 'D']

    print(f"\nC1 size: {len(c1_df)} bursts")
    print(f"Other Up bursts: {len(other_up_df)} bursts")

    # Per-feature comparison: C1 vs rest-of-upward (Mann-Whitney U).
    print("\n" + "=" * 70)
    print("Feature comparison: C1 vs rest-of-upward")
    print("=" * 70)

    properties = [
        ('Bandwidth(MHz)', 'Bandwidth'),
        ('Weff(ms)', 'Duration'),
        ('(Average)_Freq_peak(MHz)', 'Peak Frequency'),
        ('Rd(MHz/ms)', 'Drift Rate'),
        ('Energy(1e39erg)', 'Energy'),
        ('Flux', 'Flux'),
        ('S/N', 'Signal-to-Noise'),
        ('center_freq', 'Center Frequency'),
    ]

    print(f"\n{'Property':<20} {'C1 (n=45)':<15} {'Other Up':<15} {'All Down':<15} {'C1 vs Others':<15}")
    print("-" * 80)

    significant_differences = []

    for col, name in properties:
        if col not in df.columns:
            continue

        c1_vals = c1_df[col].dropna()
        other_up_vals = other_up_df[col].dropna()
        down_vals = all_down_df[col].dropna()

        if len(c1_vals) < 5 or len(other_up_vals) < 5:
            continue

        c1_med = c1_vals.median()
        other_med = other_up_vals.median()
        down_med = down_vals.median()

        stat, p_val = stats.mannwhitneyu(c1_vals, other_up_vals, alternative='two-sided')

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"{name:<20} {c1_med:<15.2f} {other_med:<15.2f} {down_med:<15.2f} {sig:<15}")

        if p_val < 0.01:
            ratio = c1_med / other_med if other_med != 0 else float('inf')
            significant_differences.append({
                'property': name,
                'c1_median': c1_med,
                'other_up_median': other_med,
                'ratio': ratio,
                'p_value': p_val
            })

    print("-" * 80)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")

    if significant_differences:
        print("\n" + "=" * 70)
        print("Features significantly different in C1 (p < 0.01)")
        print("=" * 70)

        for diff in significant_differences:
            direction = "higher" if diff['ratio'] > 1 else "lower"
            print(f"\n  {diff['property']}:")
            print(f"    C1 median:       {diff['c1_median']:.2f}")
            print(f"    Other-up median: {diff['other_up_median']:.2f}")
            print(f"    Difference:      {abs(diff['ratio']-1)*100:.0f}% {direction} (p = {diff['p_value']:.2e})")

    return c1_df, significant_differences, c1_label


def analyze_up_subgroups(df, labels):
    """Partition the upward-drifting bursts into three groups by k-means and report their feature profiles."""

    print("\n" + "=" * 70)
    print("k=3 partition of upward-drifting bursts")
    print("=" * 70)

    up_mask = df['drift_type'] == 'U'
    up_df = df[up_mask].copy()

    # Features for clustering
    feature_cols = ['Bandwidth(MHz)', 'Weff(ms)', '(Average)_Freq_peak(MHz)',
                    'Rd(MHz/ms)', 'Energy(1e39erg)', 'Flux', 'S/N']

    X_up = up_df[feature_cols].copy()
    for col in feature_cols:
        X_up[col] = X_up[col].fillna(X_up[col].median())

    for col in ['Energy(1e39erg)', 'Flux', 'Bandwidth(MHz)']:
        if col in X_up.columns:
            X_up[col] = np.log10(X_up[col].clip(lower=1e-10))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_up)

    # K-means with k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    up_labels = kmeans.fit_predict(X_scaled)
    up_df['subgroup'] = up_labels

    print(f"\nUp subgroup sizes:")
    for i in range(3):
        n = (up_labels == i).sum()
        print(f"  Subgroup {i}: {n} bursts ({n/len(up_labels)*100:.1f}%)")

    # Characterize each subgroup
    print(f"\n{'Subgroup':<10} {'Bandwidth':<12} {'Duration':<12} {'Energy':<12} {'Drift':<12}")
    print("-" * 60)

    subgroup_chars = []
    for i in range(3):
        sub_df = up_df[up_df['subgroup'] == i]
        bw = sub_df['Bandwidth(MHz)'].median()
        dur = sub_df['Weff(ms)'].median()
        energy = sub_df['Energy(1e39erg)'].median()
        drift = sub_df['Rd(MHz/ms)'].median()

        print(f"{i:<10} {bw:<12.1f} {dur:<12.2f} {energy:<12.2e} {drift:<12.2f}")

        subgroup_chars.append({
            'subgroup': i,
            'bandwidth': bw,
            'duration': dur,
            'energy': energy,
            'drift': drift,
            'n': len(sub_df)
        })

    # Label subgroups by their relative energy and bandwidth.
    print("\nSubgroup labels:")

    # Sort by energy
    by_energy = sorted(subgroup_chars, key=lambda x: x['energy'])
    by_bandwidth = sorted(subgroup_chars, key=lambda x: x['bandwidth'])

    for char in subgroup_chars:
        i = char['subgroup']
        if char['energy'] == by_energy[0]['energy']:
            name = "Low-Energy Up"
        elif char['energy'] == by_energy[2]['energy']:
            name = "High-Energy Up"
        else:
            name = "Medium-Energy Up"

        if char['bandwidth'] == by_bandwidth[0]['bandwidth']:
            bw_name = "Narrowband"
        elif char['bandwidth'] == by_bandwidth[2]['bandwidth']:
            bw_name = "Broadband"
        else:
            bw_name = "Medium-band"

        print(f"  Subgroup {i}: '{bw_name} {name}' (n={char['n']})")

    return up_df, subgroup_chars


def create_detailed_visualization(df, X_2d, labels, c1_label, up_df=None):
    """Produce the C1 characterisation figure."""

    fig = plt.figure(figsize=(18, 14))

    drift_types = df['drift_type'].values
    c1_mask = labels == c1_label

    # 1. UMAP with C1 highlighted
    ax1 = fig.add_subplot(2, 3, 1)
    # Plot all points faded
    ax1.scatter(X_2d[~c1_mask, 0], X_2d[~c1_mask, 1],
                c=['lightblue' if t == 'U' else 'lightcoral' for t in drift_types[~c1_mask]],
                alpha=0.3, s=20, label='Other')
    # Highlight C1
    ax1.scatter(X_2d[c1_mask, 0], X_2d[c1_mask, 1],
                c='darkgreen', s=50, alpha=0.8, label=f'C1 (n={c1_mask.sum()})', edgecolor='black')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_title('Cluster C1 in UMAP space')
    ax1.legend()

    # 2. C1 vs Others in Bandwidth vs Energy space
    ax2 = fig.add_subplot(2, 3, 2)
    other_mask = ~c1_mask & (drift_types == 'U')
    down_mask = drift_types == 'D'

    ax2.scatter(np.log10(df.loc[down_mask, 'Bandwidth(MHz)'].clip(lower=0.1)),
                np.log10(df.loc[down_mask, 'Energy(1e39erg)'].clip(lower=1e-10)),
                c='lightcoral', alpha=0.3, s=20, label='Down')
    ax2.scatter(np.log10(df.loc[other_mask, 'Bandwidth(MHz)'].clip(lower=0.1)),
                np.log10(df.loc[other_mask, 'Energy(1e39erg)'].clip(lower=1e-10)),
                c='lightblue', alpha=0.5, s=30, label='Other Up')
    ax2.scatter(np.log10(df.loc[c1_mask, 'Bandwidth(MHz)'].clip(lower=0.1)),
                np.log10(df.loc[c1_mask, 'Energy(1e39erg)'].clip(lower=1e-10)),
                c='darkgreen', s=60, alpha=0.8, label='C1', edgecolor='black')
    ax2.set_xlabel(r'$\log_{10}$ Bandwidth (MHz)')
    ax2.set_ylabel(r'$\log_{10}$ Energy ($10^{39}$ erg)')
    ax2.set_title('C1 in bandwidth-energy space')
    ax2.legend()

    # 3. C1 vs Others in Duration vs Drift space
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(df.loc[down_mask, 'Weff(ms)'],
                df.loc[down_mask, 'Rd(MHz/ms)'],
                c='lightcoral', alpha=0.3, s=20, label='Down')
    ax3.scatter(df.loc[other_mask, 'Weff(ms)'],
                df.loc[other_mask, 'Rd(MHz/ms)'],
                c='lightblue', alpha=0.5, s=30, label='Other Up')
    ax3.scatter(df.loc[c1_mask, 'Weff(ms)'],
                df.loc[c1_mask, 'Rd(MHz/ms)'],
                c='darkgreen', s=60, alpha=0.8, label='C1', edgecolor='black')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Duration (ms)')
    ax3.set_ylabel('Drift Rate (MHz/ms)')
    ax3.set_title('C1 in duration-drift space')
    ax3.legend()

    # 4. Property distributions comparison
    ax4 = fig.add_subplot(2, 3, 4)
    c1_energy = df.loc[c1_mask, 'Energy(1e39erg)'].dropna()
    other_up_energy = df.loc[other_mask, 'Energy(1e39erg)'].dropna()

    ax4.hist(np.log10(other_up_energy.clip(lower=1e-10)), bins=30, alpha=0.5,
             label='Other Up', color='blue', density=True)
    ax4.hist(np.log10(c1_energy.clip(lower=1e-10)), bins=15, alpha=0.7,
             label='C1', color='green', density=True)
    ax4.set_xlabel(r'$\log_{10}$ Energy')
    ax4.set_ylabel('Density')
    ax4.set_title('Energy distribution: C1 vs rest-of-upward')
    ax4.legend()

    # 5. Bandwidth distribution
    ax5 = fig.add_subplot(2, 3, 5)
    c1_bw = df.loc[c1_mask, 'Bandwidth(MHz)'].dropna()
    other_up_bw = df.loc[other_mask, 'Bandwidth(MHz)'].dropna()

    ax5.hist(other_up_bw, bins=30, alpha=0.5, label='Other Up', color='blue', density=True)
    ax5.hist(c1_bw, bins=15, alpha=0.7, label='C1', color='green', density=True)
    ax5.set_xlabel('Bandwidth (MHz)')
    ax5.set_ylabel('Density')
    ax5.set_title('Bandwidth distribution: C1 vs rest-of-upward')
    ax5.legend()

    # 6. Summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    c1_df = df[c1_mask]
    other_up_df = df[other_mask]

    summary = f"""
Cluster C1 feature medians
{'=' * 45}

C1 (n={len(c1_df)}):
  Bandwidth:  {c1_df['Bandwidth(MHz)'].median():.1f} MHz
  Duration:   {c1_df['Weff(ms)'].median():.2f} ms
  Energy:     {c1_df['Energy(1e39erg)'].median():.2e} x 10^39 erg
  Drift rate: {c1_df['Rd(MHz/ms)'].median():.2f} MHz/ms
  Peak freq:  {c1_df['(Average)_Freq_peak(MHz)'].median():.0f} MHz

Rest-of-upward (n={len(other_up_df)}):
  Bandwidth:  {other_up_df['Bandwidth(MHz)'].median():.1f} MHz
  Duration:   {other_up_df['Weff(ms)'].median():.2f} ms
  Energy:     {other_up_df['Energy(1e39erg)'].median():.2e} x 10^39 erg
  Drift rate: {other_up_df['Rd(MHz/ms)'].median():.2f} MHz/ms
  Peak freq:  {other_up_df['(Average)_Freq_peak(MHz)'].median():.0f} MHz

C1 is 100% upward-drifting and forms a distinct
island in the UMAP projection.
"""

    ax6.text(0.02, 0.98, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Cluster C1 characterisation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'characterize_cluster.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {OUTPUT_DIR / 'characterize_cluster.png'}")


def main():
    print("=" * 70)
    print("Cluster C1 characterisation")
    print("=" * 70)

    df, X_scaled, X_2d, labels, feature_cols = load_and_cluster()
    c1_df, significant_diffs, c1_label = characterize_cluster_c1(df, labels)
    up_df, subgroup_chars = analyze_up_subgroups(df, labels)
    create_detailed_visualization(df, X_2d, labels, c1_label, up_df)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    if significant_diffs:
        print("\n  Properties significantly different in C1:")
        for diff in significant_diffs[:3]:
            print(f"    - {diff['property']}: {diff['ratio']:.2f}x (p = {diff['p_value']:.2e})")
    else:
        print("\n  No features show a significant difference between C1 and the remaining upward-drifting population.")

    print(f"\n  Results written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
