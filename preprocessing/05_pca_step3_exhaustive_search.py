"""
================================================================================
PCA OPTIMIZATION - STEP 3: EXHAUSTIVE VARIABLE SEARCH (p=4 to p=8)
================================================================================

PURPOSE:
    Test all combinations of variables from 4 to 8 variables per surface.
    Find best combination for EACH p value to enable evidence-based decision.
    
    This is Step 3 of the methodology journey:
    - Step 1: Tennis-informed (8 vars) → 49-55% variance
    - Step 2: Statistical selection (4 vars) → 63-78% variance
    - Step 3: Exhaustive search → finds optimal combinations
    - Step 4: Final optimized (4 vars) → 83-90% variance

OUTPUT (results/step3_exhaustive_search_p4to8/):
    - optimization_summary_by_p.csv: Best combination for each p value per surface
    - optimization_[surface]_top10.csv: Top 10 combinations overall
    - optimization_[surface]_best_by_p.csv: Best per p for each surface

USAGE:
    python 05_pca_step3_exhaustive_search.py
================================================================================
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
INPUT_FILE = "tennis_metrics_by_player.csv"
INPUT_PATH = DATA_DIR / INPUT_FILE

OUTPUT_DIR = ROOT_DIR / "results" / "step3_exhaustive_search_p4to8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PCA OPTIMIZATION - STEP 3: EXHAUSTIVE SEARCH")
print("p = 4 to 8 variables")
print("=" * 60)
print(f"Input: {INPUT_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"n=21 players per surface")
print("")

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(INPUT_PATH, sep=';')

ALL_VARS = [c for c in df.columns 
            if c not in ['player', 'surface'] 
            and pd.api.types.is_numeric_dtype(df[c])]

print(f"Total variables available: {len(ALL_VARS)}")
print(f"Surfaces: {df['surface'].unique()}")
print("")

# =============================================================================
# OPTIMIZATION FUNCTION
# =============================================================================

def test_combinations_for_surface(df_surface, var_list, min_vars=4, max_vars=8):
    """Test all combinations and return results DataFrame."""
    results = []
    
    for p in range(min_vars, max_vars + 1):
        combos = list(combinations(var_list, p))
        
        for combo in combos:
            vars_used = list(combo)
            X = df_surface[vars_used].values
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=2)
            pca.fit(X_scaled)
            
            total_variance = pca.explained_variance_ratio_.sum()
            pc1_variance = pca.explained_variance_ratio_[0]
            pc2_variance = pca.explained_variance_ratio_[1]
            
            results.append({
                'n_variables': p,
                'variables': ', '.join(vars_used),
                'pc1_variance': pc1_variance,
                'pc2_variance': pc2_variance,
                'total_variance': total_variance,
                'n_p_ratio': round(len(df_surface) / p, 2)
            })
    
    results_df = pd.DataFrame(results)
    return results_df

# =============================================================================
# RUN OPTIMIZATION FOR EACH SURFACE
# =============================================================================

surfaces = df['surface'].unique()
best_by_p_results = {}

for surface in surfaces:
    print(f"\n{'='*50}")
    print(f"OPTIMIZING: {surface}")
    print(f"{'='*50}")
    
    df_surface = df[df['surface'] == surface]
    
    # Test all combinations
    results_df = test_combinations_for_surface(df_surface, ALL_VARS, min_vars=4, max_vars=8)
    
    # Save top 10 overall
    top10 = results_df.sort_values('total_variance', ascending=False).head(10)
    top10.to_csv(OUTPUT_DIR / f"optimization_{surface}_top10.csv", index=False)
    
    # Find best for each p
    best_by_p = []
    for p in range(4, 9):
        p_results = results_df[results_df['n_variables'] == p]
        if not p_results.empty:
            best = p_results.loc[p_results['total_variance'].idxmax()]
            best_by_p.append({
                'surface': surface,
                'n_variables': p,
                'n_p_ratio': best['n_p_ratio'],
                'total_variance': best['total_variance'],
                'pc1_variance': best['pc1_variance'],
                'pc2_variance': best['pc2_variance'],
                'variables': best['variables']
            })
    
    best_by_p_df = pd.DataFrame(best_by_p)
    best_by_p_df.to_csv(OUTPUT_DIR / f"optimization_{surface}_best_by_p.csv", index=False)
    best_by_p_results[surface] = best_by_p_df
    
    # Print summary for this surface
    print(f"\n  BEST BY NUMBER OF VARIABLES:")
    print(f"  {'p':<3} {'n/p':<6} {'PC1':<8} {'PC2':<8} {'Total':<8}")
    print(f"  {'-'*40}")
    for _, row in best_by_p_df.iterrows():
        print(f"  {row['n_variables']:<3} {row['n_p_ratio']:<6.2f} {row['pc1_variance']:<8.2%} {row['pc2_variance']:<8.2%} {row['total_variance']:<8.2%}")

# =============================================================================
# SAVE COMBINED SUMMARY (All Surfaces, All p)
# =============================================================================

all_summary = []
for surface, df_summary in best_by_p_results.items():
    all_summary.append(df_summary)

combined_summary = pd.concat(all_summary, ignore_index=True)
combined_summary.to_csv(OUTPUT_DIR / "optimization_summary_by_p.csv", index=False)

print("\n" + "=" * 60)
print("FINAL SUMMARY - BEST BY NUMBER OF VARIABLES")
print("=" * 60)
print(f"\n{'surface':<8} {'p':<3} {'n/p':<6} {'PC1':<8} {'PC2':<8} {'Total':<8}")
print(f"{'-'*60}")

for _, row in combined_summary.iterrows():
    print(f"{row['surface']:<8} {row['n_variables']:<3} {row['n_p_ratio']:<6.2f} {row['pc1_variance']:<8.2%} {row['pc2_variance']:<8.2%} {row['total_variance']:<8.2%}")

# =============================================================================
# RECOMMENDATION TABLE
# =============================================================================

print("\n" + "=" * 60)
print("EVIDENCE-BASED DECISION TABLE")
print("=" * 60)

decision_data = []
for surface in surfaces:
    surface_summary = combined_summary[combined_summary['surface'] == surface]
    
    for _, row in surface_summary.iterrows():
        p = row['n_variables']
        variance = row['total_variance']
        np_ratio = row['n_p_ratio']
        
        # Statistical assessment
        if np_ratio >= 5:
            stat_status = "GOOD"
        elif np_ratio >= 4:
            stat_status = "ACCEPTABLE"
        elif np_ratio >= 3:
            stat_status = "BORDERLINE"
        else:
            stat_status = "POOR"
        
        # Variance assessment
        if variance >= 0.85:
            var_status = "EXCELLENT"
        elif variance >= 0.80:
            var_status = "GOOD"
        elif variance >= 0.70:
            var_status = "ACCEPTABLE"
        else:
            var_status = "POOR"
        
        decision_data.append({
            'surface': surface,
            'p': p,
            'n/p_ratio': np_ratio,
            'variance': variance,
            'stat_status': stat_status,
            'var_status': var_status,
            'recommended': "YES" if (np_ratio >= 4 and variance >= 0.80) else "NO"
        })

decision_df = pd.DataFrame(decision_data)
print(decision_df.to_string(index=False))

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)

for surface in surfaces:
    surface_decisions = decision_df[decision_df['surface'] == surface]
    best_tradeoff = surface_decisions[surface_decisions['recommended'] == 'YES']
    
    if not best_tradeoff.empty:
        print(f"\n{surface}:")
        for _, row in best_tradeoff.iterrows():
            print(f"  p={row['p']} (n/p={row['n/p_ratio']:.2f}:1) - Variance={row['variance']:.2%}")

print("\n" + "=" * 60)
print("STEP 3 COMPLETE")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nProceed to Step 4 (final optimized PCA).")
print("=" * 60)