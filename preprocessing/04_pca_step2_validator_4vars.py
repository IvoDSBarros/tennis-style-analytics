"""
================================================================================
PCA ANALYSIS - STEP 2: STATISTICAL SELECTION (4 VARS, 2 COMPONENTS)
================================================================================

PURPOSE:
    PCA using 4 variables per surface selected by statistical rules (variance thresholds).
    Results show 65-72% explained variance, improving from Step 1 (49-55%).
    
    This is Step 2 of the methodology journey:
    - Step 1: Tennis-informed (8 vars) → 49-55% variance
    - Step 2: Statistical selection (4 vars) → 65-72% variance
    - Step 3: Exhaustive search → finds optimal combinations
    - Step 4: Final optimized (4 vars) → 83-90% variance

VARIABLES PER SURFACE (from assumption validator, NO entropy protection):
    Clay:   winner_error_ratio, drop_shot_effectiveness, 
            break_point_conversion_pct, fisrt_serve_pct
    Grass:  winner_error_ratio, drop_shot_effectiveness, 
            break_point_conversion_pct, break_points_saved_pct
    Hard:   winner_error_ratio, drop_shot_effectiveness, 
            break_points_saved_pct, break_point_conversion_pct

OUTPUT (results/step2_validator_4vars/):
    - scores_all_surfaces.csv      # Player coordinates (PC1, PC2)
    - loadings_all_surfaces.csv    # Variable contributions
    - variance_all_surfaces.csv    # Explained variance
    - summary_surface.csv          # Sinner/Alcaraz vs Big 3
    - comparison_report.csv        # Step 1 vs Step 2 comparison
    - methodology_note.txt         # Documentation

USAGE:
    python 04_pca_step2_validator_4vars.py
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime

# =============================================================================
# PATHS
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT_DIR / "data"
INPUT_FILE = "tennis_metrics_by_player.csv"
INPUT_PATH = DATA_DIR / INPUT_FILE

OUTPUT_DIR = ROOT_DIR / "results" / "step2_validator_4vars"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PCA ANALYSIS - STEP 2: STATISTICAL SELECTION (4 VARS)")
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

# Surface-specific variables (from assumption validator, NO entropy protection)
VARS_BY_SURFACE = {
    "Clay": [
        "winner_error_ratio",
        "drop_shot_effectiveness",
        "break_point_conversion_pct",
        "fisrt_serve_pct"
    ],
    "Grass": [
        "winner_error_ratio",
        "drop_shot_effectiveness",
        "break_point_conversion_pct",
        "break_points_saved_pct"
    ],
    "Hard": [
        "winner_error_ratio",
        "drop_shot_effectiveness",
        "break_points_saved_pct",
        "break_point_conversion_pct"
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

print("Variables per surface:")
for surface, vars_list in VARS_BY_SURFACE.items():
    print(f"  {surface}: {vars_list}")
print("")

# =============================================================================
# BASELINE VARIANCE (from Step 1: 8-variable tennis-informed analysis)
# =============================================================================

BASELINE_VARIANCE = {
    "Clay": 0.5548,
    "Grass": 0.5125,
    "Hard": 0.4939
}

print("Baseline variance (Step 1 - 8 vars):")
for surface, var in BASELINE_VARIANCE.items():
    print(f"  {surface}: {var:.1%}")
print("")

# =============================================================================
# PROCESS EACH SURFACE
# =============================================================================

surfaces = df['surface'].unique()
all_scores = []
all_loadings = []
all_variance = []
comparison_results = []

for surface in surfaces:
    print(f"\n{'='*50}")
    print(f"SURFACE: {surface}")
    print(f"{'='*50}")
    
    # Get variables for this surface
    vars_available = VARS_BY_SURFACE[surface]
    
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
    
    # Compare to baseline (Step 1)
    baseline = BASELINE_VARIANCE[surface]
    improvement = total_variance - baseline
    print(f"  Improvement vs Step 1 (8 vars): {improvement:+.1%}")
    
    # Store comparison
    comparison_results.append({
        'surface': surface,
        'n_variables': len(vars_available),
        'n_p_ratio': round(len(players) / len(vars_available), 2),
        'step1_variance': baseline,
        'step2_variance': total_variance,
        'improvement': improvement,
        'variables_used': ', '.join(vars_available)
    })
    
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
# SAVE RESULTS
# =============================================================================

combined_scores = pd.concat(all_scores, ignore_index=True)
combined_loadings = pd.concat(all_loadings, ignore_index=True)
combined_variance = pd.concat(all_variance, ignore_index=True)
comparison_df = pd.DataFrame(comparison_results)

combined_scores.to_csv(OUTPUT_DIR / "scores_all_surfaces.csv", index=False)
combined_loadings.to_csv(OUTPUT_DIR / "loadings_all_surfaces.csv", index=False)
combined_variance.to_csv(OUTPUT_DIR / "variance_all_surfaces.csv", index=False)
comparison_df.to_csv(OUTPUT_DIR / "comparison_report.csv", index=False)

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)
print(f"  {OUTPUT_DIR / 'scores_all_surfaces.csv'}")
print(f"  {OUTPUT_DIR / 'loadings_all_surfaces.csv'}")
print(f"  {OUTPUT_DIR / 'variance_all_surfaces.csv'}")
print(f"  {OUTPUT_DIR / 'comparison_report.csv'}")

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
print(f"\n  {OUTPUT_DIR / 'summary_surface.csv'}")

# =============================================================================
# METHODOLOGY NOTE
# =============================================================================

methodology_note = f"""================================================================================
STEP 2 REPORT - STATISTICAL SELECTION (4 VARS, 2 COMPONENTS)
================================================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Configuration: 4 variables per surface, 2 components
Sample: 21 players per surface (Year-end Top 10, 2010-2024)

RESULTS SUMMARY:
-------------------------------------------------------------------------------
{comparison_df[['surface', 'n_p_ratio', 'step1_variance', 'step2_variance', 'improvement']].to_string(index=False)}

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
LOADINGS INTERPRETATION:
-------------------------------------------------------------------------------
"""

for surface in surfaces:
    loadings_surface = combined_loadings[combined_loadings['surface'] == surface]
    
    methodology_note += f"\n{surface}:\n"
    methodology_note += f"  PC1 defined by: {loadings_surface.nlargest(1, 'PC1')['variable'].values[0]} ({loadings_surface.nlargest(1, 'PC1')['PC1'].values[0]:.2f}) vs {loadings_surface.nsmallest(1, 'PC1')['variable'].values[0]} ({loadings_surface.nsmallest(1, 'PC1')['PC1'].values[0]:.2f})\n"
    methodology_note += f"  PC2 defined by: {loadings_surface.nlargest(1, 'PC2')['variable'].values[0]} ({loadings_surface.nlargest(1, 'PC2')['PC2'].values[0]:.2f}) vs {loadings_surface.nsmallest(1, 'PC2')['variable'].values[0]} ({loadings_surface.nsmallest(1, 'PC2')['PC2'].values[0]:.2f})\n"

methodology_note += """
CONCLUSION:
-------------------------------------------------------------------------------
Step 2 achieves better n/p ratio (5.25:1 vs 2.6:1) 
and higher explained variance (65-72% vs 49-55%) compared to Step 1.

This demonstrates that statistical variable selection improves PCA performance
with elite sample constraints (n=21).

Proceed to Step 3 (exhaustive search) to find optimal combinations.

FILES GENERATED:
-------------------------------------------------------------------------------
- scores_all_surfaces.csv      # Player coordinates
- loadings_all_surfaces.csv    # Variable contributions
- variance_all_surfaces.csv    # Explained variance
- summary_surface.csv          # Sinner/Alcaraz vs Big 3
- comparison_report.csv        # Step 1 vs Step 2 comparison
- methodology_note.txt         # This file
================================================================================
"""

with open(OUTPUT_DIR / "methodology_note.txt", "w") as f:
    f.write(methodology_note)

print(f"  {OUTPUT_DIR / 'methodology_note.txt'}")

# =============================================================================
# PRINT COMPARISON SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("COMPARISON SUMMARY: Step 1 (8 vars) vs Step 2 (4 vars)")
print("=" * 60)

for _, row in comparison_df.iterrows():
    print(f"\n{row['surface']}:")
    print(f"  Step 1 (8 vars): {row['step1_variance']:.1%}")
    print(f"  Step 2 (4 vars): {row['step2_variance']:.1%}")
    print(f"  Improvement:     {row['improvement']:+.1%}")
    print(f"  n/p ratio:       2.6:1 → {row['n_p_ratio']}:1")

print("\n" + "=" * 60)
print("STEP 2 COMPLETE")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nProceed to Step 3 (exhaustive search).")
print("=" * 60)