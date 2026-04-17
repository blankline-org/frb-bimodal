"""Unsupervised cluster discovery in the FRB 20240114A feature space.

Applies UMAP (n_neighbours=15, min_dist=0.1) and HDBSCAN (min_cluster_size=15,
min_samples=5) to an eight-feature space of burst-cluster properties published
by Zhang et al. (2026). Identifies subpopulations without specifying their
number in advance, and tests whether discovered clusters reproduce the
published Up/Down drift-direction classification or reveal hidden structure
within it.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Check for UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed. Using PCA instead.")
    print("Install with: pip install umap-learn")

# Check for HDBSCAN
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("WARNING: hdbscan not installed. Using DBSCAN instead.")
    from sklearn.cluster import DBSCAN

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load data and select features for clustering."""
    path = Path("data/raw/FRB20240114A_Morphology_Public_Dataset_20240312CST/All_Drifting_Burst-Cluster_Table.xlsx")
    df = pd.read_excel(path)

    # Sort by time
    df = df.sort_values('MJD_of_Burst-Clusters').reset_index(drop=True)

    # Create Up/Down labels
    df['drift_type'] = df['Morphology_DU'].str[0]
    df = df[df['drift_type'].isin(['U', 'D'])].reset_index(drop=True)

    print(f"Loaded {len(df)} bursts")
    print(f"  Up: {(df['drift_type'] == 'U').sum()}")
    print(f"  Down: {(df['drift_type'] == 'D').sum()}")

    return df


def select_features(df):
    """Select and clean features for clustering."""

    # Physical features for the "fingerprint"
    feature_cols = [
        'Bandwidth(MHz)',           # Frequency spread
        'Weff(ms)',                  # Duration
        '(Average)_Freq_peak(MHz)', # Peak frequency
        'Rd(MHz/ms)',               # Drift rate
        'Energy(1e39erg)',          # Energy
        'Flux',                     # Brightness
        'S/N',                      # Signal quality
        'center_freq',              # Center frequency
    ]

    # Check which columns exist and have data
    available_cols = []
    for col in feature_cols:
        if col in df.columns:
            n_valid = df[col].notna().sum()
            if n_valid > len(df) * 0.5:  # At least 50% data
                available_cols.append(col)
                print(f"  Using: {col} ({n_valid} values)")
            else:
                print(f"  Skipping: {col} (only {n_valid} values)")
        else:
            print(f"  Missing: {col}")

    print(f"\nUsing {len(available_cols)} features for clustering")

    # Extract feature matrix
    X = df[available_cols].copy()

    # Handle missing values - use median imputation
    for col in available_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)

    # Log-transform skewed features
    for col in ['Energy(1e39erg)', 'Flux', 'Bandwidth(MHz)']:
        if col in X.columns:
            X[col] = np.log10(X[col].clip(lower=1e-10))

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, available_cols


def run_dimensionality_reduction(X_scaled):
    """Reduce to 2D for visualization."""

    if HAS_UMAP:
        print("\nRunning UMAP dimensionality reduction...")
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=42
        )
        X_2d = reducer.fit_transform(X_scaled)
        method = "UMAP"
    else:
        print("\nRunning PCA dimensionality reduction...")
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X_scaled)
        method = "PCA"
        print(f"  Variance explained: {reducer.explained_variance_ratio_.sum():.1%}")

    return X_2d, method


def run_clustering(X_scaled, X_2d):
    """Find natural clusters in the data."""

    if HAS_HDBSCAN:
        print("\nRunning HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=15,
            min_samples=5,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(X_scaled)
        method = "HDBSCAN"
    else:
        print("\nRunning DBSCAN clustering...")
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        labels = clusterer.fit_predict(X_scaled)
        method = "DBSCAN"

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    print(f"  Found {n_clusters} clusters")
    print(f"  Noise points: {n_noise}")

    # Calculate silhouette score (excluding noise)
    if n_clusters > 1:
        mask = labels >= 0
        if mask.sum() > n_clusters:
            sil_score = silhouette_score(X_scaled[mask], labels[mask])
            print(f"  Silhouette score: {sil_score:.3f}")
        else:
            sil_score = 0
    else:
        sil_score = 0

    return labels, n_clusters, method, sil_score


def analyze_clusters_vs_drift_type(df, labels):
    """Test whether clusters reproduce the Up/Down classification or reveal hidden structure."""
    print("\n" + "=" * 70)
    print("Cluster vs drift-type analysis")
    print("=" * 70)

    drift_types = df['drift_type'].values

    # Create contingency table
    unique_labels = sorted(set(labels))

    print("\nContingency Table (rows=clusters, cols=Up/Down):")
    print("-" * 50)
    print(f"{'Cluster':<10} {'Up':<10} {'Down':<10} {'Total':<10} {'%Up':<10}")
    print("-" * 50)

    contingency = []
    for label in unique_labels:
        mask = labels == label
        n_up = ((drift_types == 'U') & mask).sum()
        n_down = ((drift_types == 'D') & mask).sum()
        total = mask.sum()
        pct_up = n_up / total * 100 if total > 0 else 0

        label_name = f"Noise" if label == -1 else f"C{label}"
        print(f"{label_name:<10} {n_up:<10} {n_down:<10} {total:<10} {pct_up:<.1f}%")

        if label >= 0:
            contingency.append([n_up, n_down])

    print("-" * 50)

    # Chi-squared test: Are clusters independent of Up/Down?
    if len(contingency) >= 2:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-squared test (Clusters vs Up/Down):")
        print(f"  Chi² = {chi2:.2f}, p = {p_value:.2e}")

        if p_value < 0.001:
            print("\n  Cluster assignments are not simply a rediscovery of Up/Down.")
        elif p_value > 0.05:
            print("\n  Clusters track the Up/Down split closely.")

    # Mixed-cluster search: clusters containing both Up and Down bursts
    print("\n" + "=" * 70)
    print("Mixed-cluster search")
    print("=" * 70)

    third_species_candidates = []

    for label in unique_labels:
        if label == -1:
            continue

        mask = labels == label
        n_up = ((drift_types == 'U') & mask).sum()
        n_down = ((drift_types == 'D') & mask).sum()
        total = mask.sum()

        if total < 10:
            continue

        pct_up = n_up / total
        pct_down = n_down / total

        # Pure clusters exceed 90% one type; mixed clusters are candidates for
        # further investigation.
        if 0.2 < pct_up < 0.8:
            third_species_candidates.append({
                'cluster': label,
                'n_up': n_up,
                'n_down': n_down,
                'total': total,
                'pct_up': pct_up,
                'entropy': -pct_up*np.log2(pct_up+1e-10) - pct_down*np.log2(pct_down+1e-10)
            })

    if third_species_candidates:
        print("\n  Mixed-composition clusters:")
        for cand in third_species_candidates:
            print(f"\n    Cluster {cand['cluster']}:")
            print(f"      Size: {cand['total']} bursts")
            print(f"      Composition: {cand['pct_up']:.1%} Up, {1-cand['pct_up']:.1%} Down")
    else:
        print("\n  No mixed clusters detected; clusters align with Up/Down.")

    return third_species_candidates


def analyze_subpopulations_within_type(df, X_scaled, drift_type='U'):
    """Search for subpopulations within a single drift-direction class."""
    print(f"\n" + "=" * 70)
    print(f"Subpopulation search within drift-type '{drift_type}'")
    print("=" * 70)

    mask = df['drift_type'] == drift_type
    X_subset = X_scaled[mask]

    print(f"\nAnalyzing {X_subset.shape[0]} {drift_type}-type bursts")

    if X_subset.shape[0] < 30:
        print("  Too few bursts for subpopulation analysis")
        return None

    # Try different numbers of clusters
    best_k = 1
    best_score = -1

    for k in range(2, min(6, X_subset.shape[0] // 10)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        sub_labels = kmeans.fit_predict(X_subset)

        if len(set(sub_labels)) > 1:
            score = silhouette_score(X_subset, sub_labels)
            print(f"  k={k}: silhouette = {score:.3f}")

            if score > best_score:
                best_score = score
                best_k = k

    # Statistical test: Is k>1 significantly better than k=1?
    print(f"\n  Best k = {best_k} (silhouette = {best_score:.3f})")

    if best_k > 1 and best_score > 0.2:
        print(f"\n  Subpopulation structure detected in {drift_type} bursts ({best_k} groups).")

        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        sub_labels = kmeans.fit_predict(X_subset)

        return sub_labels, best_k, best_score
    else:
        print(f"\n  No subpopulation structure detected in {drift_type} bursts.")
        return None


def hunt_drift_reversals(df):
    """Identify bursts near zero drift, candidates for drift-sign reversal."""
    print("\n" + "=" * 70)
    print("Near-zero drift rate candidates")
    print("=" * 70)

    drift = df['Rd(MHz/ms)'].values

    # Look for bursts near zero drift
    zero_drift_mask = np.abs(drift) < 5  # Within ±5 MHz/ms
    n_near_zero = zero_drift_mask.sum()

    print(f"\nBursts with |drift| < 5 MHz/ms: {n_near_zero}")

    if n_near_zero > 0:
        near_zero_df = df[zero_drift_mask]

        print(f"\n  Drift rate range: [{drift[zero_drift_mask].min():.2f}, {drift[zero_drift_mask].max():.2f}] MHz/ms")
        print(f"  Mean bandwidth: {near_zero_df['Bandwidth(MHz)'].mean():.1f} MHz")
        print(f"  Mean energy: {near_zero_df['Energy(1e39erg)'].mean():.2e} x 10^39 erg")

        up_energy = df.loc[df['drift_type'] == 'U', 'Energy(1e39erg)'].mean()
        down_energy = df.loc[df['drift_type'] == 'D', 'Energy(1e39erg)'].mean()
        zero_energy = near_zero_df['Energy(1e39erg)'].mean()

        print(f"\n  Energy comparison:")
        print(f"    Up bursts:        {up_energy:.2e}")
        print(f"    Down bursts:      {down_energy:.2e}")
        print(f"    Near-zero drift:  {zero_energy:.2e}")

        if zero_energy > max(up_energy, down_energy) * 1.5:
            print("\n  Near-zero drift bursts carry higher mean energy than either class.")

        return near_zero_df

    return None


def create_visualization(df, X_2d, labels, method, feature_cols):
    """Create comprehensive visualization."""

    fig = plt.figure(figsize=(18, 12))

    drift_types = df['drift_type'].values
    colors_type = ['blue' if t == 'U' else 'red' for t in drift_types]

    # 1. UMAP/PCA colored by Up/Down
    ax1 = fig.add_subplot(2, 3, 1)
    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=colors_type, alpha=0.5, s=20)
    ax1.set_xlabel(f'{method} 1')
    ax1.set_ylabel(f'{method} 2')
    ax1.set_title(f'{method} - Colored by Drift Type\n(Blue=Up, Red=Down)')

    # 2. UMAP/PCA colored by cluster
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.5, s=20)
    ax2.set_xlabel(f'{method} 1')
    ax2.set_ylabel(f'{method} 2')
    ax2.set_title(f'{method} - Colored by Cluster\n(Gray=Noise)')
    plt.colorbar(scatter2, ax=ax2)

    # 3. Drift rate vs Energy
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(df['Rd(MHz/ms)'], np.log10(df['Energy(1e39erg)'].clip(lower=1e-10)),
                c=colors_type, alpha=0.5, s=20)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Drift Rate (MHz/ms)')
    ax3.set_ylabel('log₁₀(Energy)')
    ax3.set_title('Drift Rate vs Energy\n(Blue=Up, Red=Down)')

    # 4. Bandwidth vs Duration
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(df['Weff(ms)'], df['Bandwidth(MHz)'], c=colors_type, alpha=0.5, s=20)
    ax4.set_xlabel('Duration (ms)')
    ax4.set_ylabel('Bandwidth (MHz)')
    ax4.set_title('Duration vs Bandwidth\n(Blue=Up, Red=Down)')

    # 5. Cluster composition
    ax5 = fig.add_subplot(2, 3, 5)
    unique_labels = sorted([l for l in set(labels) if l >= 0])

    if unique_labels:
        up_counts = []
        down_counts = []
        for label in unique_labels:
            mask = labels == label
            up_counts.append(((drift_types == 'U') & mask).sum())
            down_counts.append(((drift_types == 'D') & mask).sum())

        x = np.arange(len(unique_labels))
        width = 0.35
        ax5.bar(x - width/2, up_counts, width, label='Up', color='blue', alpha=0.7)
        ax5.bar(x + width/2, down_counts, width, label='Down', color='red', alpha=0.7)
        ax5.set_xlabel('Cluster')
        ax5.set_ylabel('Count')
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'C{l}' for l in unique_labels])
        ax5.legend()
        ax5.set_title('Cluster Composition\n(Mixed clusters = Third Species?)')

    # 6. Summary text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    summary = f"""
FRB 20240114A: feature-space summary
{'=' * 40}

Bursts analysed: {len(df)}
  Up:   {(drift_types == 'U').sum()}
  Down: {(drift_types == 'D').sum()}

Features used: {len(feature_cols)}
  {', '.join(feature_cols[:4])}
  {', '.join(feature_cols[4:]) if len(feature_cols) > 4 else ''}

Clustering:
  Method:         HDBSCAN
  Clusters found: {n_clusters}
  Noise points:   {n_noise}
"""

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.suptitle('FRB 20240114A: unsupervised cluster discovery', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'discover_cluster.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {OUTPUT_DIR / 'discover_cluster.png'}")


def main():
    print("=" * 70)
    print("FRB 20240114A: unsupervised cluster discovery")
    print("=" * 70)

    df = load_and_prepare_data()

    print("\n" + "=" * 70)
    print("Feature selection")
    print("=" * 70)
    X_scaled, feature_cols = select_features(df)

    X_2d, dr_method = run_dimensionality_reduction(X_scaled)
    labels, n_clusters, cl_method, sil_score = run_clustering(X_scaled, X_2d)

    third_species = analyze_clusters_vs_drift_type(df, labels)

    print("\n")
    up_subpop = analyze_subpopulations_within_type(df, X_scaled, 'U')

    print("\n")
    down_subpop = analyze_subpopulations_within_type(df, X_scaled, 'D')

    reversals = hunt_drift_reversals(df)

    create_visualization(df, X_2d, labels, dr_method, feature_cols)

    print("\n" + "=" * 70)
    print("Analysis summary")
    print("=" * 70)

    findings = []

    if third_species:
        findings.append(f"Mixed-composition clusters: {len(third_species)} candidate(s)")

    if up_subpop:
        findings.append(f"Up bursts contain {up_subpop[1]} distinct subgroups")

    if down_subpop:
        findings.append(f"Down bursts contain {down_subpop[1]} distinct subgroups")

    if reversals is not None and len(reversals) > 10:
        findings.append(f"{len(reversals)} near-zero drift candidates")

    if findings:
        print("\n  Findings:")
        for f in findings:
            print(f"    - {f}")
    else:
        print("\n  No structure beyond the published Up/Down classification was detected.")

    print(f"\n  Results written to: {OUTPUT_DIR}")

    return {
        'df': df,
        'labels': labels,
        'X_2d': X_2d,
        'third_species': third_species,
        'up_subpop': up_subpop,
        'down_subpop': down_subpop,
        'reversals': reversals
    }


if __name__ == "__main__":
    results = main()
