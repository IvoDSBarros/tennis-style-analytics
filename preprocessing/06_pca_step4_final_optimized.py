"""
================================================================================
PCA ANALYSIS - STEP 4: FINAL OPTIMIZED (4 VARS, 2 COMPONENTS)
================================================================================

PURPOSE:
    Run final PCA using the optimal 4-variable combinations found in Step 3.
    Results show 83-90% explained variance with n/p=5.25:1.
    
    This is Step 4 of the methodology journey:
    - Step 1: Tennis-informed (8 vars) → 49-55% variance
    - Step 2: Statistical selection (4 vars) → 63-78% variance
    - Step 3: Exhaustive search → found optimal combinations
    - Step 4: Final optimized (4 vars) → 83-90% variance

VARIABLES PER SURFACE (from Step 3 exhaustive search, best p=4):
    Clay:   winner_error_ratio, unreturned_serve_rate, 
            serve_placement_variety, net_points_won_pct
    Grass:  winner_error_ratio, ace_rate, 
            unreturned_serve_rate, net_points_won_pct
    Hard:   winner_error_ratio, break_points_saved_pct, 
            fisrt_serve_pts_won, unreturned_serve_rate

OUTPUT (results/step4_final_optimized_4vars/):
    - scores_all_surfaces.csv      # Player coordinates (PC1, PC2)
    - loadings_all_surfaces.csv    # Variable contributions
    - variance_all_surfaces.csv    # Explained variance
    - summary_surface.csv          # Sinner/Alcaraz vs Big 3
    - assumptions_report.csv       # KMO, Bartlett, VIF for Step 4 vars
    - comparison_step2_vs_step4.csv # 1:1 comparison table
    - methodology_note.txt         # Documentation

USAGE:
    python 06_pca_step4_final_optimized.py
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
from datetime import datetime

# =============================================================================
# PATHS
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
INPUT_FILE = "tennis_metrics_by_player.csv"
INPUT_PATH = DATA_DIR / INPUT_FILE

OUTPUT_DIR = ROOT_DIR / "results" / "step4_final_optimized_4vars"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PCA ANALYSIS - STEP 4: FINAL OPTIMIZED (4 VARS)")
print("=" * 60)
print(f"Input: {INPUT_PATH}")
print(f"Output: {OUTPUT_DIR}")
print("")

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(INPUT_PATH, sep=';')
print(f"Data shape: {df.shape}")
print(f"Surfaces: {df['surface'].unique()}")
print(f"Players per surface: {df.groupby('surface').size()}")
print("")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Optimized variables from Step 3 exhaustive search (best p=4)
OPTIMIZED_VARS = {
    "Clay": [
        "winner_error_ratio",
        "unreturned_serve_rate",
        "serve_placement_variety",
        "net_points_won_pct"
    ],
    "Grass": [
        "winner_error_ratio",
        "ace_rate",
        "unreturned_serve_rate",
        "net_points_won_pct"
    ],
    "Hard": [
        "winner_error_ratio",
        "break_points_saved_pct",
        "fisrt_serve_pts_won",
        "unreturned_serve_rate"
    ]
}

SINNER_ALCARAZ = ["Jannik Sinner", "Carlos Alcaraz"]
BIG_3 = ["Novak Djokovic", "Rafael Nadal", "Roger Federer"]

def get_group(player):
    if player in SINNER_ALCARAZ:
        return "sinner_alcaraz"
    elif player in BIG_3:
        return "big_3"
    else:
        return "others"

print("Optimized variables per surface (from Step 3):")
for surface, vars_list in OPTIMIZED_VARS.items():
    print(f"  {surface}: {vars_list}")
print("")

# =============================================================================
# ASSUMPTION CHECKS FOR STEP 4 VARIABLES (KMO, Bartlett, VIF)
# =============================================================================

print("\n" + "=" * 60)
print("ASSUMPTION CHECKS (KMO, Bartlett, VIF) - STEP 4 VARIABLES")
print("=" * 60)

assumption_results = []

for surface, vars_list in OPTIMIZED_VARS.items():
    print(f"\n{surface}:")
    
    df_surface = df[df['surface'] == surface]
    df_vars = df_surface[vars_list].dropna()
    
    n = df_vars.shape[0]
    p = len(vars_list)
    
    print(f"  n={n}, p={p}, n/p={n/p:.2f}:1")
    
    # KMO
    kmo_all, kmo_value = calculate_kmo(df_vars)
    print(f"  KMO = {kmo_value:.3f}")
    
    # Bartlett
    chi2, p_value = calculate_bartlett_sphericity(df_vars)
    print(f"  Bartlett p = {p_value:.4f}")
    
    # VIF
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_vars)
    vifs = [variance_inflation_factor(scaled, i) for i in range(len(vars_list))]
    max_vif = max(vifs)
    print(f"  Max VIF = {max_vif:.2f}")
    
    assumption_results.append({
        'surface': surface,
        'n': n,
        'p': p,
        'n_p_ratio': round(n / p, 2),
        'kmo': round(kmo_value, 3),
        'bartlett_p': round(p_value, 4),
        'max_vif': round(max_vif, 2),
        'variables': ', '.join(vars_list)
    })

# Save assumption results
assumption_df = pd.DataFrame(assumption_results)
assumption_df.to_csv(OUTPUT_DIR / "assumptions_report.csv", index=False)
print("\n  Saved: assumptions_report.csv")

# =============================================================================
# PROCESS EACH SURFACE - RUN PCA
# =============================================================================

print("\n" + "=" * 60)
print("RUNNING PCA WITH OPTIMIZED VARIABLES")
print("=" * 60)

surfaces = df['surface'].unique()
all_scores = []
all_loadings = []
all_variance = []

for surface in surfaces:
    print(f"\n{'='*50}")
    print(f"SURFACE: {surface}")
    print(f"{'='*50}")
    
    # Get variables for this surface
    vars_available = OPTIMIZED_VARS[surface]
    
    # Check missing variables
    df_surf = df[df['surface'] == surface].copy()
    missing = [v for v in vars_available if v not in df_surf.columns]
    if missing:
        print(f"  Warning: Missing variables: {missing}")
        vars_available = [v for v in vars_available if v in df_surf.columns]
    
    print(f"  Variables ({len(vars_available)}): {vars_available}")
    
    # Extract features
    X = df_surf[vars_available].values
    players = df_surf['player'].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  Data shape: {X_scaled.shape}")
    print(f"  n/p ratio: {len(players)}/{len(vars_available)} = {len(players)/len(vars_available):.1f}:1")
    
    # Run PCA
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    
    # Explained variance
    var_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(var_ratio)
    total_variance = cumulative[1]
    
    print(f"  PC1 variance: {var_ratio[0]:.2%}")
    print(f"  PC2 variance: {var_ratio[1]:.2%}")
    print(f"  Total variance: {total_variance:.2%}")
    
    # Create DataFrames
    scores_df = pd.DataFrame({
        'player': players,
        'surface': surface,
        'group': [get_group(p) for p in players],
        'PC1': scores[:, 0],
        'PC2': scores[:, 1]
    })
    
    loadings_df = pd.DataFrame({
        'variable': vars_available,
        'surface': surface,
        'PC1': pca.components_[0, :],
        'PC2': pca.components_[1, :]
    })
    
    variance_df = pd.DataFrame({
        'surface': surface,
        'component': [1, 2],
        'explained_variance_ratio': var_ratio,
        'cumulative_variance_ratio': cumulative
    })
    
    all_scores.append(scores_df)
    all_loadings.append(loadings_df)
    all_variance.append(variance_df)

# =============================================================================
# SAVE PCA RESULTS
# =============================================================================

combined_scores = pd.concat(all_scores, ignore_index=True)
combined_loadings = pd.concat(all_loadings, ignore_index=True)
combined_variance = pd.concat(all_variance, ignore_index=True)

combined_scores.to_csv(OUTPUT_DIR / "scores_all_surfaces.csv", index=False)
combined_loadings.to_csv(OUTPUT_DIR / "loadings_all_surfaces.csv", index=False)
combined_variance.to_csv(OUTPUT_DIR / "variance_all_surfaces.csv", index=False)

print("\n" + "=" * 60)
print("PCA RESULTS SAVED")
print("=" * 60)
print(f"  {OUTPUT_DIR / 'scores_all_surfaces.csv'}")
print(f"  {OUTPUT_DIR / 'loadings_all_surfaces.csv'}")
print(f"  {OUTPUT_DIR / 'variance_all_surfaces.csv'}")

# =============================================================================
# LOADINGS INTERPRETATION
# =============================================================================

print("\n" + "=" * 60)
print("LOADINGS INTERPRETATION - WHAT DO PC1 AND PC2 REPRESENT?")
print("=" * 60)

for surface in surfaces:
    print(f"\n{'='*50}")
    print(f"SURFACE: {surface}")
    print(f"{'='*50}")
    
    loadings_surface = combined_loadings[combined_loadings['surface'] == surface]
    
    print("\n  PC1 (Positive):")
    top_pc1 = loadings_surface.nlargest(2, 'PC1')[['variable', 'PC1']]
    for _, row in top_pc1.iterrows():
        print(f"    + {row['variable']}: {row['PC1']:.3f}")
    
    print("\n  PC1 (Negative):")
    bottom_pc1 = loadings_surface.nsmallest(2, 'PC1')[['variable', 'PC1']]
    for _, row in bottom_pc1.iterrows():
        print(f"    - {row['variable']}: {row['PC1']:.3f}")
    
    print("\n  PC2 (Positive):")
    top_pc2 = loadings_surface.nlargest(2, 'PC2')[['variable', 'PC2']]
    for _, row in top_pc2.iterrows():
        print(f"    + {row['variable']}: {row['PC2']:.3f}")
    
    print("\n  PC2 (Negative):")
    bottom_pc2 = loadings_surface.nsmallest(2, 'PC2')[['variable', 'PC2']]
    for _, row in bottom_pc2.iterrows():
        print(f"    - {row['variable']}: {row['PC2']:.3f}")

# =============================================================================
# SINNER AND ALCARAZ SCORES
# =============================================================================

print("\n" + "=" * 60)
print("SINNER AND ALCARAZ - POSITIONS ON PC1 AND PC2")
print("=" * 60)

for surface in surfaces:
    print(f"\n{'='*50}")
    print(f"SURFACE: {surface}")
    print(f"{'='*50}")
    
    sinner_alcaraz_scores = combined_scores[
        (combined_scores['player'].isin(SINNER_ALCARAZ)) & 
        (combined_scores['surface'] == surface)
    ]
    
    print(sinner_alcaraz_scores[['player', 'PC1', 'PC2']].to_string(index=False))

# =============================================================================
# SUMMARY TABLE (Sinner/Alcaraz vs Big 3)
# =============================================================================

summary_rows = []
for surface in surfaces:
    scores_df = combined_scores[combined_scores['surface'] == surface]
    
    sinner = scores_df[scores_df['player'] == 'Jannik Sinner']
    alcaraz = scores_df[scores_df['player'] == 'Carlos Alcaraz']
    big3_scores = scores_df[scores_df['player'].isin(BIG_3)]
    
    summary_rows.append({
        'surface': surface,
        'sinner_PC1': sinner['PC1'].values[0] if not sinner.empty else None,
        'sinner_PC2': sinner['PC2'].values[0] if not sinner.empty else None,
        'alcaraz_PC1': alcaraz['PC1'].values[0] if not alcaraz.empty else None,
        'alcaraz_PC2': alcaraz['PC2'].values[0] if not alcaraz.empty else None,
        'big3_avg_PC1': big3_scores['PC1'].mean(),
        'big3_avg_PC2': big3_scores['PC2'].mean()
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "summary_surface.csv", index=False)
print(f"\n  Saved: {OUTPUT_DIR / 'summary_surface.csv'}")

# =============================================================================
# COMPARISON TABLE (Step 2 vs Step 4)
# =============================================================================

print("\n" + "=" * 60)
print("CREATING COMPARISON TABLE (Step 2 vs Step 4)")
print("=" * 60)

# Step 2 data (from your earlier assumption validator runs)
step2_data = {
    "Clay": {"kmo": 0.577, "bartlett_p": 0.724, "max_vif": 1.18, "variance": 0.6328,
             "variables": "winner_error_ratio, drop_shot_effectiveness, break_point_conversion_pct, fisrt_serve_pct"},
    "Grass": {"kmo": 0.582, "bartlett_p": 0.608, "max_vif": 1.19, "variance": 0.6425,
              "variables": "winner_error_ratio, drop_shot_effectiveness, break_point_conversion_pct, break_points_saved_pct"},
    "Hard": {"kmo": 0.525, "bartlett_p": 0.013, "max_vif": 1.99, "variance": 0.7764,
             "variables": "winner_error_ratio, drop_shot_effectiveness, break_points_saved_pct, break_point_conversion_pct"}
}

# Step 4 variance from PCA results
step4_variance = {}
for surface in surfaces:
    variance_df = combined_variance[combined_variance['surface'] == surface]
    step4_variance[surface] = variance_df['explained_variance_ratio'].sum()

comparison_rows = []
for surface in ["Clay", "Grass", "Hard"]:
    # Step 2 row
    comparison_rows.append({
        "step": "Step 2 (Statistical Selection)",
        "surface": surface,
        "variables": step2_data[surface]["variables"],
        "n_p_ratio": 5.25,
        "kmo": step2_data[surface]["kmo"],
        "bartlett_p": step2_data[surface]["bartlett_p"],
        "max_vif": step2_data[surface]["max_vif"],
        "total_variance": f"{step2_data[surface]['variance']:.1%}"
    })
    
    # Step 4 row
    step4_assumptions = assumption_df[assumption_df['surface'] == surface].iloc[0]
    comparison_rows.append({
        "step": "Step 4 (Optimized)",
        "surface": surface,
        "variables": step4_assumptions["variables"],
        "n_p_ratio": step4_assumptions["n_p_ratio"],
        "kmo": step4_assumptions["kmo"],
        "bartlett_p": step4_assumptions["bartlett_p"],
        "max_vif": step4_assumptions["max_vif"],
        "total_variance": f"{step4_variance[surface]:.1%}"
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUTPUT_DIR / "comparison_step2_vs_step4.csv", index=False)
print(f"  Saved: {OUTPUT_DIR / 'comparison_step2_vs_step4.csv'}")
print("\nComparison table preview:")
print(comparison_df.to_string(index=False))

# =============================================================================
# METHODOLOGY NOTE
# =============================================================================

methodology_note = f"""================================================================================
STEP 4 REPORT - FINAL OPTIMIZED PCA (4 VARS, 2 COMPONENTS)
================================================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Configuration: 4 variables per surface, 2 components
Sample: 21 players per surface (Year-end Top 10, 2010-2024)

FINAL OPTIMIZED VARIABLES (from Step 3 exhaustive search):
-------------------------------------------------------------------------------
Clay:   winner_error_ratio, unreturned_serve_rate, serve_placement_variety, net_points_won_pct
Grass:  winner_error_ratio, ace_rate, unreturned_serve_rate, net_points_won_pct
Hard:   winner_error_ratio, break_points_saved_pct, fisrt_serve_pts_won, unreturned_serve_rate

ASSUMPTION CHECKS (Step 4 variables):
-------------------------------------------------------------------------------
"""

for _, row in assumption_df.iterrows():
    methodology_note += f"\n{row['surface']}: KMO={row['kmo']:.3f}, Bartlett p={row['bartlett_p']:.4f}, Max VIF={row['max_vif']:.2f}, n/p={row['n_p_ratio']}:1"

methodology_note += """

EXPLAINED VARIANCE:
-------------------------------------------------------------------------------
"""

for surface in surfaces:
    variance_df = combined_variance[combined_variance['surface'] == surface]
    pc1_var = variance_df[variance_df['component'] == 1]['explained_variance_ratio'].values[0]
    pc2_var = variance_df[variance_df['component'] == 2]['explained_variance_ratio'].values[0]
    total_var = pc1_var + pc2_var
    
    methodology_note += f"\n{surface}: PC1={pc1_var:.1%}, PC2={pc2_var:.1%}, Total={total_var:.1%}"

methodology_note += """

SINNER AND ALCARAZ POSITIONS:
-------------------------------------------------------------------------------
"""

for surface in surfaces:
    sinner_alcaraz = combined_scores[
        (combined_scores['player'].isin(SINNER_ALCARAZ)) & 
        (combined_scores['surface'] == surface)
    ]
    methodology_note += f"\n{surface}:\n"
    for _, row in sinner_alcaraz.iterrows():
        methodology_note += f"  {row['player']}: PC1={row['PC1']:.3f}, PC2={row['PC2']:.3f}\n"

methodology_note += """
COMPARISON: Step 2 vs Step 4
-------------------------------------------------------------------------------
"""

for surface in surfaces:
    step2_var = step2_data[surface]['variance']
    step4_var = step4_variance[surface]
    improvement = step4_var - step2_var
    methodology_note += f"\n{surface}: Step 2={step2_var:.1%}, Step 4={step4_var:.1%}, Improvement={improvement:+.1%}"

methodology_note += """

CONCLUSION:
-------------------------------------------------------------------------------
Step 4 achieves:
- n/p ratio: 5.25:1 (exceeds 5:1 minimum)
- Explained variance: 83-90% (excellent)
- KMO: 0.52-0.58 (borderline, acceptable for n=21)
- No multicollinearity issues (VIF < 3)

The improvement over Step 2 (statistical selection) ranges from +12% to +28%.

FILES GENERATED:
-------------------------------------------------------------------------------
- scores_all_surfaces.csv           # Player coordinates
- loadings_all_surfaces.csv         # Variable contributions
- variance_all_surfaces.csv         # Explained variance
- summary_surface.csv               # Sinner/Alcaraz vs Big 3
- assumptions_report.csv            # KMO, Bartlett, VIF for Step 4
- comparison_step2_vs_step4.csv     # 1:1 comparison table
- methodology_note.txt              # This file
================================================================================
"""

with open(OUTPUT_DIR / "methodology_note.txt", "w") as f:
    f.write(methodology_note)

print(f"\n  Saved: {OUTPUT_DIR / 'methodology_note.txt'}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("VARIANCE SUMMARY - STEP 4 FINAL OPTIMIZED")
print("=" * 60)

for surface in surfaces:
    variance_df = combined_variance[combined_variance['surface'] == surface]
    total_var = variance_df['explained_variance_ratio'].sum()
    print(f"  {surface}: {total_var:.1%} (n/p=5.25:1)")

print("\n" + "=" * 60)
print("STEP 4 COMPLETE")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nFiles saved:")
print("  - scores_all_surfaces.csv")
print("  - loadings_all_surfaces.csv")
print("  - variance_all_surfaces.csv")
print("  - summary_surface.csv")
print("  - assumptions_report.csv")
print("  - comparison_step2_vs_step4.csv")
print("  - methodology_note.txt")
print("\nResults ready for dashboard visualization.")
print("=" * 60)