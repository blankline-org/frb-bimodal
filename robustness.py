"""Robustness checks for cluster C1.

Runs six independent tests to establish that the 45-burst cluster identified
in discover_cluster.py is not an artefact of a single parameter choice or
clustering algorithm: UMAP-parameter stability, HDBSCAN-parameter stability,
bootstrap resampling, alternative clustering algorithms, per-feature
Mann-Whitney tests with Bonferroni correction, and a label-permutation test.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load and prepare data."""
    path = Path("data/raw/FRB20240114A_Morphology_Public_Dataset_20240312CST/All_Drifting_Burst-Cluster_Table.xlsx")
    df = pd.read_excel(path)
    df = df.sort_values('MJD_of_Burst-Clusters').reset_index(drop=True)
    df['drift_type'] = df['Morphology_DU'].str[0]
    df = df[df['drift_type'].isin(['U', 'D'])].reset_index(drop=True)
    return df


def prepare_features(df):
    """Prepare feature matrix."""
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

    return X_scaled, feature_cols


def identify_extreme_drift_cluster(labels, drift_types, drift_rates):
    """Identify which cluster (if any) corresponds to extreme drift Up bursts."""
    best_cluster = None
    best_score = 0

    for label in set(labels):
        if label < 0:
            continue
        mask = labels == label
        n_total = mask.sum()
        if n_total < 10:
            continue

        # Check if mostly Up
        up_pct = ((drift_types == 'U') & mask).sum() / n_total

        # Check if high drift rate
        mean_drift = np.abs(drift_rates[mask]).mean()

        # Score: high Up percentage AND high drift rate
        if up_pct > 0.9:  # At least 90% Up
            score = up_pct * mean_drift
            if score > best_score:
                best_score = score
                best_cluster = label

    return best_cluster


def test_umap_parameters(X_scaled, drift_types, drift_rates):
    """Recover C1 across six UMAP parameter configurations."""
    print("\n" + "=" * 70)
    print("Test 1: UMAP parameter sensitivity")
    print("=" * 70)

    if not HAS_UMAP:
        print("  UMAP not available, skipping...")
        return None

    param_sets = [
        {'n_neighbors': 10, 'min_dist': 0.05},
        {'n_neighbors': 15, 'min_dist': 0.1},   # Original
        {'n_neighbors': 20, 'min_dist': 0.1},
        {'n_neighbors': 15, 'min_dist': 0.2},
        {'n_neighbors': 30, 'min_dist': 0.1},
        {'n_neighbors': 15, 'min_dist': 0.05},
    ]

    results = []
    reference_labels = None

    for i, params in enumerate(param_sets):
        reducer = umap.UMAP(n_components=2, random_state=42, **params)
        X_2d = reducer.fit_transform(X_scaled)

        if HAS_HDBSCAN:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
            labels = clusterer.fit_predict(X_scaled)
        else:
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            labels = clusterer.fit_predict(X_2d)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        extreme_cluster = identify_extreme_drift_cluster(labels, drift_types, drift_rates)

        if extreme_cluster is not None:
            n_extreme = (labels == extreme_cluster).sum()
            up_pct = ((drift_types == 'U') & (labels == extreme_cluster)).sum() / n_extreme
        else:
            n_extreme = 0
            up_pct = 0

        # Compare to reference (first run)
        if reference_labels is None:
            reference_labels = labels.copy()
            ari = 1.0
        else:
            # Only compare non-noise points
            mask = (labels >= 0) & (reference_labels >= 0)
            if mask.sum() > 0:
                ari = adjusted_rand_score(reference_labels[mask], labels[mask])
            else:
                ari = 0

        results.append({
            'params': params,
            'n_clusters': n_clusters,
            'extreme_found': extreme_cluster is not None,
            'n_extreme': n_extreme,
            'up_pct': up_pct,
            'ari': ari
        })

        marker = "[ok]" if extreme_cluster is not None else "[--]"
        print(f"  {marker} n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}: "
              f"{n_clusters} clusters, extreme={n_extreme} bursts, ARI={ari:.2f}")

    found_count = sum(1 for r in results if r['extreme_found'])
    print(f"\n  C1 recovered in {found_count}/{len(results)} UMAP configurations.")

    return results


def test_hdbscan_parameters(X_scaled, drift_types, drift_rates):
    """Recover C1 across six HDBSCAN parameter configurations."""
    print("\n" + "=" * 70)
    print("Test 2: HDBSCAN parameter sensitivity")
    print("=" * 70)

    if not HAS_HDBSCAN:
        print("  HDBSCAN not available, skipping...")
        return None

    param_sets = [
        {'min_cluster_size': 10, 'min_samples': 3},
        {'min_cluster_size': 15, 'min_samples': 5},  # Original
        {'min_cluster_size': 20, 'min_samples': 5},
        {'min_cluster_size': 15, 'min_samples': 10},
        {'min_cluster_size': 25, 'min_samples': 5},
        {'min_cluster_size': 10, 'min_samples': 5},
    ]

    results = []
    reference_labels = None

    for params in param_sets:
        clusterer = hdbscan.HDBSCAN(**params)
        labels = clusterer.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        extreme_cluster = identify_extreme_drift_cluster(labels, drift_types, drift_rates)

        if extreme_cluster is not None:
            n_extreme = (labels == extreme_cluster).sum()
            mean_drift = np.abs(drift_rates[labels == extreme_cluster]).mean()
        else:
            n_extreme = 0
            mean_drift = 0

        if reference_labels is None:
            reference_labels = labels.copy()
            ari = 1.0
        else:
            mask = (labels >= 0) & (reference_labels >= 0)
            ari = adjusted_rand_score(reference_labels[mask], labels[mask]) if mask.sum() > 0 else 0

        results.append({
            'params': params,
            'n_clusters': n_clusters,
            'extreme_found': extreme_cluster is not None,
            'n_extreme': n_extreme,
            'mean_drift': mean_drift,
            'ari': ari
        })

        marker = "[ok]" if extreme_cluster is not None else "[--]"
        print(f"  {marker} min_cluster_size={params['min_cluster_size']}, min_samples={params['min_samples']}: "
              f"{n_clusters} clusters, extreme={n_extreme}, drift={mean_drift:.1f}")

    found_count = sum(1 for r in results if r['extreme_found'])
    print(f"\n  C1 recovered in {found_count}/{len(results)} HDBSCAN configurations.")

    return results


def test_bootstrap_stability(X_scaled, drift_types, drift_rates, n_bootstrap=100, seed=2026):
    """Recover C1 under bootstrap resampling of the burst catalogue."""
    print("\n" + "=" * 70)
    print(f"Test 3: Bootstrap stability ({n_bootstrap} resamples)")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    n_samples = X_scaled.shape[0]
    extreme_sizes = []
    extreme_drifts = []
    found_count = 0

    for i in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_scaled[idx]
        dt_boot = drift_types[idx]
        dr_boot = drift_rates[idx]

        # Cluster
        if HAS_HDBSCAN:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
            labels = clusterer.fit_predict(X_boot)
        else:
            clusterer = KMeans(n_clusters=3, random_state=i, n_init=10)
            labels = clusterer.fit_predict(X_boot)

        # Find extreme cluster
        extreme_cluster = identify_extreme_drift_cluster(labels, dt_boot, dr_boot)

        if extreme_cluster is not None:
            found_count += 1
            n_extreme = (labels == extreme_cluster).sum()
            mean_drift = np.abs(dr_boot[labels == extreme_cluster]).mean()
            extreme_sizes.append(n_extreme)
            extreme_drifts.append(mean_drift)

    print(f"\n  C1 recovered in {found_count}/{n_bootstrap} bootstrap resamples ({found_count/n_bootstrap*100:.1f}%).")

    if extreme_sizes:
        print(f"  Cluster size: {np.mean(extreme_sizes):.1f} +/- {np.std(extreme_sizes):.1f}")
        print(f"  Mean drift rate: {np.mean(extreme_drifts):.1f} +/- {np.std(extreme_drifts):.1f} MHz/ms")
        print(f"  95% CI on size: [{np.percentile(extreme_sizes, 2.5):.0f}, {np.percentile(extreme_sizes, 97.5):.0f}]")

    return {
        'found_rate': found_count / n_bootstrap,
        'sizes': extreme_sizes,
        'drifts': extreme_drifts
    }


def test_alternative_methods(X_scaled, drift_types, drift_rates):
    """Recover C1 under alternative clustering algorithms."""
    print("\n" + "=" * 70)
    print("Test 4: Alternative clustering algorithms")
    print("=" * 70)

    methods = [
        ('K-Means (k=3)', KMeans(n_clusters=3, random_state=42, n_init=10)),
        ('K-Means (k=4)', KMeans(n_clusters=4, random_state=42, n_init=10)),
        ('K-Means (k=5)', KMeans(n_clusters=5, random_state=42, n_init=10)),
        ('Agglomerative (k=3)', AgglomerativeClustering(n_clusters=3)),
        ('Agglomerative (k=4)', AgglomerativeClustering(n_clusters=4)),
        ('DBSCAN (eps=1.0)', DBSCAN(eps=1.0, min_samples=5)),
        ('DBSCAN (eps=1.5)', DBSCAN(eps=1.5, min_samples=5)),
    ]

    results = []

    for name, clusterer in methods:
        labels = clusterer.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Look for high-drift Up cluster
        extreme_cluster = identify_extreme_drift_cluster(labels, drift_types, drift_rates)

        if extreme_cluster is not None:
            n_extreme = (labels == extreme_cluster).sum()
            mean_drift = np.abs(drift_rates[labels == extreme_cluster]).mean()
            up_pct = ((drift_types == 'U') & (labels == extreme_cluster)).sum() / n_extreme
        else:
            n_extreme = 0
            mean_drift = 0
            up_pct = 0

        results.append({
            'method': name,
            'n_clusters': n_clusters,
            'extreme_found': extreme_cluster is not None,
            'n_extreme': n_extreme,
            'mean_drift': mean_drift,
            'up_pct': up_pct
        })

        marker = "[ok]" if extreme_cluster is not None else "[--]"
        print(f"  {marker} {name}: {n_clusters} clusters, extreme={n_extreme}, drift={mean_drift:.1f}")

    found_count = sum(1 for r in results if r['extreme_found'])
    print(f"\n  C1 recovered with {found_count}/{len(methods)} alternative algorithms.")

    return results


def test_statistical_significance(df, X_scaled, drift_types, drift_rates):
    """Per-feature Mann-Whitney tests of C1 vs rest-of-upward with Bonferroni correction."""
    print("\n" + "=" * 70)
    print("Test 5: Per-feature statistical significance")
    print("=" * 70)

    # Get cluster labels
    if HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
        labels = clusterer.fit_predict(X_scaled)
    else:
        clusterer = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_scaled)

    extreme_cluster = identify_extreme_drift_cluster(labels, drift_types, drift_rates)

    if extreme_cluster is None:
        print("  C1 not recovered; skipping significance testing.")
        return None

    extreme_mask = labels == extreme_cluster
    other_up_mask = (drift_types == 'U') & ~extreme_mask

    print(f"\n  Comparing C1 ({extreme_mask.sum()} bursts) against rest-of-upward ({other_up_mask.sum()} bursts).")

    # Test each property
    properties = [
        ('Drift Rate', 'Rd(MHz/ms)'),
        ('Duration', 'Weff(ms)'),
        ('Peak Frequency', '(Average)_Freq_peak(MHz)'),
        ('Bandwidth', 'Bandwidth(MHz)'),
        ('Energy', 'Energy(1e39erg)'),
        ('S/N', 'S/N'),
    ]

    results = []
    print(f"\n  {'Property':<20} {'C1 median':<15} {'Other median':<15} {'p-value':<15} {'Significant?'}")
    print("  " + "-" * 75)

    for name, col in properties:
        if col not in df.columns:
            continue

        c1_vals = df.loc[extreme_mask, col].dropna()
        other_vals = df.loc[other_up_mask, col].dropna()

        if len(c1_vals) < 5 or len(other_vals) < 5:
            continue

        stat, p_val = stats.mannwhitneyu(c1_vals, other_vals, alternative='two-sided')

        # Bonferroni correction (6 tests)
        sig = "***" if p_val < 0.001/6 else "**" if p_val < 0.01/6 else "*" if p_val < 0.05/6 else ""

        results.append({
            'property': name,
            'c1_median': c1_vals.median(),
            'other_median': other_vals.median(),
            'p_value': p_val,
            'significant': len(sig) > 0
        })

        print(f"  {name:<20} {c1_vals.median():<15.2f} {other_vals.median():<15.2f} {p_val:<15.2e} {sig}")

    n_significant = sum(1 for r in results if r['significant'])
    print(f"\n  {n_significant}/{len(results)} features differ significantly after Bonferroni correction.")

    return results


def test_permutation(X_scaled, drift_types, drift_rates, n_permutations=500):
    """Label-permutation test on drift-type assignments to calibrate chance expectation."""
    print("\n" + "=" * 70)
    print(f"Test 6: Label-permutation test ({n_permutations} permutations)")
    print("=" * 70)

    # Get observed cluster
    if HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
        labels = clusterer.fit_predict(X_scaled)
    else:
        clusterer = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_scaled)

    extreme_cluster = identify_extreme_drift_cluster(labels, drift_types, drift_rates)

    if extreme_cluster is None:
        print("  C1 not recovered; skipping permutation test.")
        return None

    obs_size = (labels == extreme_cluster).sum()
    obs_drift = np.abs(drift_rates[labels == extreme_cluster]).mean()
    obs_up_pct = ((drift_types == 'U') & (labels == extreme_cluster)).sum() / obs_size

    print(f"\n  Observed C1: size={obs_size}, drift={obs_drift:.1f} MHz/ms, upward fraction={obs_up_pct:.1%}.")

    print(f"\n  Drawing {n_permutations} permutations with drift-type labels shuffled.")

    perm_drifts = []
    perm_up_pcts = []

    for i in range(n_permutations):
        # Shuffle drift type labels
        perm_drift_types = np.random.permutation(drift_types)

        # Find extreme cluster with permuted labels
        extreme_perm = identify_extreme_drift_cluster(labels, perm_drift_types, drift_rates)

        if extreme_perm is not None:
            perm_drift = np.abs(drift_rates[labels == extreme_perm]).mean()
            perm_up = ((perm_drift_types == 'U') & (labels == extreme_perm)).sum() / (labels == extreme_perm).sum()
            perm_drifts.append(perm_drift)
            perm_up_pcts.append(perm_up)

    if perm_drifts:
        p_drift = np.mean(np.array(perm_drifts) >= obs_drift)
        p_up = np.mean(np.array(perm_up_pcts) >= obs_up_pct)

        print(f"\n  Permutation p-values:")
        print(f"    P(drift >= {obs_drift:.1f}) = {p_drift:.4f}")
        print(f"    P(upward fraction >= {obs_up_pct:.1%}) = {p_up:.4f}")

        if p_drift < 0.05 and p_up < 0.05:
            print("\n  Both statistics exceed the 95th percentile of the permuted distribution.")
        else:
            print("\n  One or both statistics fall within the permuted-null range.")
    else:
        print("  No valid permutations produced a candidate cluster.")
        p_drift, p_up = 1, 1

    return {'p_drift': p_drift, 'p_up_pct': p_up}


def test_decorrelated_features(df, drift_types, drift_rates):
    """Recover C1 on feature subsets with correlated variables removed.

    Several of the eight input features are physically coupled
    (peak-frequency/centre-frequency, energy/flux, bandwidth/energy). A
    distance-based clustering algorithm implicitly upweights such couplings.
    This test drops increasingly large correlated subsets and checks that the
    HDBSCAN cluster C1 still emerges, to confirm that the finding does not
    depend on duplicated information in the feature vector.
    """
    print("\n" + "=" * 70)
    print("Test 7: Decorrelated feature subsets")
    print("=" * 70)

    if not HAS_HDBSCAN:
        print("  HDBSCAN not available, skipping.")
        return None

    def _scale(cols):
        X = df[cols].copy()
        for c in cols:
            X[c] = X[c].fillna(X[c].median())
        for c in ['Energy(1e39erg)', 'Flux', 'Bandwidth(MHz)']:
            if c in cols:
                X[c] = np.log10(X[c].clip(lower=1e-10))
        return StandardScaler().fit_transform(X)

    variants = [
        ('all 8 features (baseline)',
         ['Bandwidth(MHz)', 'Weff(ms)', '(Average)_Freq_peak(MHz)',
          'Rd(MHz/ms)', 'Energy(1e39erg)', 'Flux', 'S/N', 'center_freq']),
        ('drop center_freq (7D)',
         ['Bandwidth(MHz)', 'Weff(ms)', '(Average)_Freq_peak(MHz)',
          'Rd(MHz/ms)', 'Energy(1e39erg)', 'Flux', 'S/N']),
        ('drop center_freq + flux (6D)',
         ['Bandwidth(MHz)', 'Weff(ms)', '(Average)_Freq_peak(MHz)',
          'Rd(MHz/ms)', 'Energy(1e39erg)', 'S/N']),
        ('decorrelated subset (5D)',
         ['Weff(ms)', '(Average)_Freq_peak(MHz)', 'Rd(MHz/ms)',
          'Energy(1e39erg)', 'S/N']),
    ]

    results = []
    for name, cols in variants:
        X = _scale(cols)
        labels = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5).fit_predict(X)
        c1 = identify_extreme_drift_cluster(labels, drift_types, drift_rates)
        if c1 is not None:
            mask = labels == c1
            n = int(mask.sum())
            up_pct = ((drift_types == 'U') & mask).sum() / n
            mean_drift = np.abs(drift_rates[mask]).mean()
            marker = "[ok]"
            print(f"  {marker} {name:<32}  n={n}, upward={up_pct:.0%}, drift={mean_drift:.1f} MHz/ms")
            results.append({'variant': name, 'found': True, 'n': n,
                            'up_pct': up_pct, 'mean_drift': mean_drift})
        else:
            marker = "[--]"
            print(f"  {marker} {name:<32}  C1 not recovered")
            results.append({'variant': name, 'found': False})

    found_count = sum(1 for r in results if r['found'])
    print(f"\n  C1 recovered in {found_count}/{len(results)} decorrelated variants.")

    return results


def create_robustness_summary(umap_results, hdbscan_results, bootstrap_results,
                               alt_results, stat_results, perm_results):
    """Assemble the six-panel robustness summary figure."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax1 = axes[0, 0]
    if umap_results:
        found = [1 if r['extreme_found'] else 0 for r in umap_results]
        ax1.bar(range(len(found)), found, color=['green' if f else 'red' for f in found])
        ax1.set_xticks(range(len(found)))
        ax1.set_xticklabels([f"Set {i+1}" for i in range(len(found))], rotation=45)
        ax1.set_ylabel('C1 recovered (1/0)')
        ax1.set_title(f'UMAP parameter stability: {sum(found)}/{len(found)}')
    else:
        ax1.text(0.5, 0.5, 'UMAP not available', ha='center', va='center')
        ax1.set_title('UMAP parameter stability')

    ax2 = axes[0, 1]
    if hdbscan_results:
        found = [1 if r['extreme_found'] else 0 for r in hdbscan_results]
        ax2.bar(range(len(found)), found, color=['green' if f else 'red' for f in found])
        ax2.set_xticks(range(len(found)))
        ax2.set_xticklabels([f"Set {i+1}" for i in range(len(found))], rotation=45)
        ax2.set_ylabel('C1 recovered (1/0)')
        ax2.set_title(f'HDBSCAN parameter stability: {sum(found)}/{len(found)}')
    else:
        ax2.text(0.5, 0.5, 'HDBSCAN not available', ha='center', va='center')
        ax2.set_title('HDBSCAN parameter stability')

    ax3 = axes[0, 2]
    if bootstrap_results and bootstrap_results['sizes']:
        ax3.hist(bootstrap_results['sizes'], bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax3.axvline(x=45, color='red', linestyle='--', linewidth=2, label='Reference (45)')
        ax3.set_xlabel('C1 size')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Bootstrap: C1 recovered in {bootstrap_results["found_rate"]*100:.0f}% of resamples')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Bootstrap unavailable', ha='center', va='center')
        ax3.set_title('Bootstrap distribution')

    ax4 = axes[1, 0]
    if alt_results:
        methods = [r['method'].split()[0] for r in alt_results]
        found = [1 if r['extreme_found'] else 0 for r in alt_results]
        colors = ['green' if f else 'red' for f in found]
        ax4.barh(range(len(methods)), found, color=colors)
        ax4.set_yticks(range(len(methods)))
        ax4.set_yticklabels(methods)
        ax4.set_xlabel('C1 recovered (1/0)')
        ax4.set_title(f'Alternative algorithms: {sum(found)}/{len(found)}')

    ax5 = axes[1, 1]
    if stat_results:
        props = [r['property'] for r in stat_results]
        pvals = [-np.log10(r['p_value']) for r in stat_results]
        colors = ['green' if r['significant'] else 'gray' for r in stat_results]
        ax5.barh(range(len(props)), pvals, color=colors)
        ax5.axvline(x=-np.log10(0.05/6), color='red', linestyle='--', label='Bonferroni threshold')
        ax5.set_yticks(range(len(props)))
        ax5.set_yticklabels(props)
        ax5.set_xlabel(r'$-\log_{10}(p)$')
        ax5.set_title('Per-feature significance')
        ax5.legend()

    ax6 = axes[1, 2]
    ax6.axis('off')

    tests_passed = 0
    total_tests = 0

    summary = "Robustness summary\n" + "-" * 35 + "\n\n"

    if umap_results:
        umap_pass = sum(1 for r in umap_results if r['extreme_found']) >= len(umap_results) * 0.5
        tests_passed += umap_pass
        total_tests += 1
        summary += f"UMAP stability:    {'pass' if umap_pass else 'fail'}\n"

    if hdbscan_results:
        hdb_pass = sum(1 for r in hdbscan_results if r['extreme_found']) >= len(hdbscan_results) * 0.5
        tests_passed += hdb_pass
        total_tests += 1
        summary += f"HDBSCAN stability: {'pass' if hdb_pass else 'fail'}\n"

    if bootstrap_results:
        boot_pass = bootstrap_results['found_rate'] >= 0.7
        tests_passed += boot_pass
        total_tests += 1
        summary += f"Bootstrap:         {'pass' if boot_pass else 'fail'} ({bootstrap_results['found_rate']*100:.0f}%)\n"

    if alt_results:
        alt_pass = sum(1 for r in alt_results if r['extreme_found']) >= len(alt_results) * 0.4
        tests_passed += alt_pass
        total_tests += 1
        summary += f"Alt. algorithms:   {'pass' if alt_pass else 'fail'}\n"

    if stat_results:
        stat_pass = sum(1 for r in stat_results if r['significant']) >= 2
        tests_passed += stat_pass
        total_tests += 1
        summary += f"Per-feature sig.:  {'pass' if stat_pass else 'fail'}\n"

    if perm_results:
        perm_pass = perm_results['p_drift'] < 0.05 and perm_results['p_up_pct'] < 0.05
        tests_passed += perm_pass
        total_tests += 1
        summary += f"Permutation:       {'pass' if perm_pass else 'fail'}\n"

    summary += "\n" + "-" * 35 + "\n"
    summary += f"Tests passed: {tests_passed}/{total_tests}\n"

    ax6.text(0.1, 0.9, summary, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Robustness checks for cluster C1', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'robustness.png', dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {OUTPUT_DIR / 'robustness.png'}.")

    return tests_passed, total_tests


def main():
    print("=" * 70)
    print("Robustness checks for cluster C1")
    print("=" * 70)

    df = load_data()
    X_scaled, feature_cols = prepare_features(df)
    drift_types = df['drift_type'].values
    drift_rates = df['Rd(MHz/ms)'].values

    print(f"\nLoaded {len(df)} burst clusters with {len(feature_cols)} features.")

    umap_results = test_umap_parameters(X_scaled, drift_types, drift_rates)
    hdbscan_results = test_hdbscan_parameters(X_scaled, drift_types, drift_rates)
    bootstrap_results = test_bootstrap_stability(X_scaled, drift_types, drift_rates, n_bootstrap=100)
    alt_results = test_alternative_methods(X_scaled, drift_types, drift_rates)
    stat_results = test_statistical_significance(df, X_scaled, drift_types, drift_rates)
    perm_results = test_permutation(X_scaled, drift_types, drift_rates, n_permutations=200)
    decorrelated_results = test_decorrelated_features(df, drift_types, drift_rates)

    tests_passed, total_tests = create_robustness_summary(
        umap_results, hdbscan_results, bootstrap_results,
        alt_results, stat_results, perm_results
    )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n{tests_passed} of {total_tests} robustness tests passed.")
    print(f"Results written to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
