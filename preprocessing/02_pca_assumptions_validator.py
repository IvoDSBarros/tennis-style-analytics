"""
================================================================================
PCA ASSUMPTION VALIDATOR - STEP 2: STATISTICAL VARIABLE SELECTION
================================================================================

PURPOSE:
    This script validates PCA assumptions (KMO, Bartlett, VIF) and selects
    variables based on statistical rules (variance thresholds).
    
    It is part of the methodology journey:
    - Step 1: Tennis-informed (8 vars) → 49-55% variance
    - Step 2: Statistical selection (4 vars) → 65-72% variance
    - Step 3: Exhaustive search → finds optimal combinations
    - Step 4: Final optimized (4 vars) → 83-90% variance

METHODOLOGY REFERENCES:
    - Kaiser (1974): KMO > 0.60 acceptable
    - Bartlett (1950): p < 0.05 indicates correlations exist
    - Hair et al. (2010): n/p ratio > 5:1 ideal

OUTPUT:
    - Console: Summary of assumption tests per surface
    - CSV: results/step2_validator_4vars/diagnostics_report.csv

USAGE:
    python 02_pca_assumptions_validator.py
================================================================================
"""

import pandas as pd
import sys
from pathlib import Path

try:
    from factor_analyzer.factor_analyzer import calculate_kmo
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install factor-analyzer statsmodels scikit-learn")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
INPUT_FILE = "tennis_metrics_by_player.csv"
INPUT_PATH = DATA_DIR / INPUT_FILE

OUTPUT_DIR = ROOT_DIR / "results" / "step2_validator_4vars"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "diagnostics_report.csv"

THRESHOLDS = {
    "kmo": {"n_lt_30": 0.55, "n_lt_100": 0.60, "n_ge_100": 0.65},
    "bartlett_alpha": {"n_lt_30": 0.10, "n_ge_30": 0.05},
    "vif_max": {"n_lt_30": 10, "n_lt_100": 8, "n_ge_100": 6},
    "low_variance": 0.001,
    "zero_variance": 0.0001
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def get_threshold(n, thresholds, key):
    if n < 30:
        return thresholds[key]["n_lt_30"]
    elif n < 100:
        return thresholds[key]["n_lt_100"]
    else:
        return thresholds[key]["n_ge_100"]


def get_bartlett_alpha(n, thresholds):
    if n < 30:
        return thresholds["bartlett_alpha"]["n_lt_30"]
    else:
        return thresholds["bartlett_alpha"]["n_ge_30"]


def select_variables(df_metrics, target_p, thresholds):
    variances = df_metrics.var()
    
    constant_vars = variances[variances < thresholds["zero_variance"]].index.tolist()
    low_var_vars = variances[variances < thresholds["low_variance"]].index.tolist()
    
    candidates = [v for v in df_metrics.columns if v not in constant_vars and v not in low_var_vars]
    
    if len(candidates) > target_p:
        candidates_sorted = sorted(candidates, key=lambda x: variances[x], reverse=True)
        selected = candidates_sorted[:target_p]
    else:
        selected = candidates.copy()
    
    return selected


def run_assumption_tests(df, variables, surface_name, thresholds):
    df_subset = df[variables].dropna()
    n = df_subset.shape[0]
    p = len(variables)
    
    kmo_threshold = get_threshold(n, thresholds, "kmo")
    bartlett_alpha = get_bartlett_alpha(n, thresholds)
    vif_threshold = get_threshold(n, thresholds, "vif_max")
    
    results = {
        "surface": surface_name,
        "n": n,
        "p": p,
        "n_p_ratio": round(n / p, 2),
        "variables": ", ".join(variables)
    }
    
    try:
        _, kmo_value = calculate_kmo(df_subset)
        results["kmo"] = round(kmo_value, 3)
        results["kmo_pass"] = kmo_value >= kmo_threshold
    except Exception:
        results["kmo"] = None
        results["kmo_pass"] = False
    
    try:
        _, p_value = calculate_bartlett_sphericity(df_subset)
        results["bartlett_p"] = round(p_value, 4)
        results["bartlett_pass"] = p_value < bartlett_alpha
    except Exception:
        results["bartlett_p"] = None
        results["bartlett_pass"] = False
    
    try:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_subset)
        vifs = [round(variance_inflation_factor(scaled, i), 2) for i in range(p)]
        results["vif_max"] = max(vifs)
        results["vif_pass"] = results["vif_max"] < vif_threshold
    except Exception:
        results["vif_max"] = None
        results["vif_pass"] = False
    
    if results["kmo_pass"] and results["bartlett_pass"] and results["vif_pass"]:
        results["verdict"] = "PASS"
        results["action"] = "RUN PCA"
    elif results["kmo_pass"] and (results["bartlett_pass"] or results["vif_pass"]):
        results["verdict"] = "BORDERLINE"
        results["action"] = "RUN PCA WITH CAUTION"
    else:
        results["verdict"] = "FAIL"
        results["action"] = "DO NOT RUN PCA"
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("PCA ASSUMPTION VALIDATOR - DEMONSTRATIVE PURPOSES ONLY")
    print("=" * 60)
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_FILE}")
    print("-" * 60)
    
    if not INPUT_PATH.exists():
        print(f"Error: File not found: {INPUT_PATH}")
        print("Run 01_data_preparation.py first to generate the input file.")
        sys.exit(1)
    
    df = pd.read_csv(INPUT_PATH, sep=';')
    
    if 'surface' not in df.columns or 'player' not in df.columns:
        print("Error: Missing 'surface' or 'player' column")
        sys.exit(1)
    
    metric_cols = [c for c in df.columns 
                   if c not in ['player', 'surface'] 
                   and pd.api.types.is_numeric_dtype(df[c])]
    
    surfaces = df['surface'].unique()
    all_results = []
    
    for surface in surfaces:
        df_surface = df[df['surface'] == surface][metric_cols]
        n = df_surface.shape[0]
        p_orig = len(metric_cols)
        
        if n >= 100:
            target_p = p_orig
        else:
            target_p = max(2, min(15, int(n / 5)))
        
        if target_p < p_orig:
            selected_vars = select_variables(df_surface, target_p, THRESHOLDS)
        else:
            selected_vars = metric_cols.copy()
        
        results = run_assumption_tests(df_surface, selected_vars, surface, THRESHOLDS)
        all_results.append(results)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\nRESULTS SUMMARY")
    print("-" * 60)
    for r in all_results:
        print(f"\n{r['surface']}:")
        print(f"  n={r['n']}, p={r['p']}, n/p={r['n_p_ratio']}:1")
        print(f"  KMO={r['kmo']} ({'PASS' if r['kmo_pass'] else 'FAIL'})")
        print(f"  Bartlett p={r['bartlett_p']} ({'PASS' if r['bartlett_pass'] else 'FAIL'})")
        print(f"  Max VIF={r['vif_max']} ({'PASS' if r['vif_pass'] else 'FAIL'})")
        print(f"  Verdict: {r['verdict']} -> {r['action']}")
    
    print("\n" + "-" * 60)
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 60)
    
    return results_df

if __name__ == "__main__":
    results = main()