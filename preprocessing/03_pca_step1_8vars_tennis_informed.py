"""
================================================================================
PCA ANALYSIS - STEP 1: TENNIS-INFORMED (8 VARS, 2 COMPONENTS)
================================================================================

PURPOSE:
    Baseline PCA using 8 tennis-informed variables.
    Results show 49-55% explained variance due to n=21 constraint.
    This demonstrates why optimization was necessary.

DIAGNOSTIC NOTE (for methodology):
    Due to elite sample constraints (n=21 players per surface), 
    2 components capture only 49-55% of playing style variance (n/p=2.6:1 ratio).
    Results are exploratory and should be interpreted as directional patterns.

INPUT:
    tennis_metrics_by_player.csv (21 players x 3 surfaces x 14 metrics)

OUTPUT (results/step1_tennis_informed_8vars/):
    - scores_all_surfaces.csv      # Player coordinates (PC1, PC2)
    - loadings_all_surfaces.csv    # Variable contributions
    - variance_all_surfaces.csv    # Explained variance
    - summary_surface.csv          # Sinner/Alcaraz vs Big 3
    - diagnostics_report.csv       # Variance summary with notes
    - methodology_note.txt         # Limitation statement

USAGE:
    python 03_pca_step1_tennis_informed.py
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

OUTPUT_DIR = ROOT_DIR / "results" / "step1_tennis_informed_8vars"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PCA ANALYSIS - STEP 1: TENNIS-INFORMED (8 VARS)")
print("=" * 60)
print(f"Input: {INPUT_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"Configuration: 8 variables, 2 components")
print(f"Diagnostic note: n=21 players per surface (n/p=2.6:1)")
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

SELECTED_VARS = [
    "winner_error_ratio",
    "ace_rate",
    "return_points_won_pct",
    "net_points_won_pct",
    "break_point_conversion_pct",
    "serve_placement_variety",
    "slice_rate",
    "crosscourt_ratio"
]

SINNER_ALCARAZ = ["Jannik Sinner", "Carlos Alcaraz"]
BIG_3 = ["Novak Djokovic", "Rafael Nadal", "Roger Federer"]

def get_group(player):
    if player in SINNER_ALCARAZ:
        return "sinner_alcaraz"
    elif player in BIG_3:
        return "big_3"
    else:
        return "others"

print(f"Variables ({len(SELECTED_VARS)}): {SELECTED_VARS}")
print("")

# =============================================================================
# PROCESS EACH SURFACE
# =============================================================================

surfaces = df['surface'].unique()
all_scores = []
all_loadings = []
all_variance = []
variance_notes = []

for surface in surfaces:
    print(f"\nProcessing: {surface}")
    
    df_surf = df[df['surface'] == surface].copy()
    vars_available = [v for v in SELECTED_VARS if v in df_surf.columns]
    
    X = df_surf[vars_available].values
    players = df_surf['player'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    
    var_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(var_ratio)
    total_variance = cumulative[1]
    
    print(f"  n={len(players)}, p={len(vars_available)}, n/p={len(players)/len(vars_available):.1f}:1")
    print(f"  PC1: {var_ratio[0]:.2%}, PC2: {var_ratio[1]:.2%}, Total: {total_variance:.2%}")
    
    variance_notes.append({
        'surface': surface,
        'n_players': len(players),
        'n_variables': len(vars_available),
        'n_p_ratio': round(len(players) / len(vars_available), 2),
        'pc1_variance': round(var_ratio[0], 4),
        'pc2_variance': round(var_ratio[1], 4),
        'total_variance_2pc': round(total_variance, 4)
    })
    
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
# SAVE COMBINED RESULTS
# =============================================================================

combined_scores = pd.concat(all_scores, ignore_index=True)
combined_loadings = pd.concat(all_loadings, ignore_index=True)
combined_variance = pd.concat(all_variance, ignore_index=True)
diagnostics_df = pd.DataFrame(variance_notes)

combined_scores.to_csv(OUTPUT_DIR / "scores_all_surfaces.csv", index=False)
combined_loadings.to_csv(OUTPUT_DIR / "loadings_all_surfaces.csv", index=False)
combined_variance.to_csv(OUTPUT_DIR / "variance_all_surfaces.csv", index=False)
diagnostics_df.to_csv(OUTPUT_DIR / "diagnostics_report.csv", index=False)

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)
print(f"  {OUTPUT_DIR / 'scores_all_surfaces.csv'}")
print(f"  {OUTPUT_DIR / 'loadings_all_surfaces.csv'}")
print(f"  {OUTPUT_DIR / 'variance_all_surfaces.csv'}")
print(f"  {OUTPUT_DIR / 'diagnostics_report.csv'}")

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
print(f"  {OUTPUT_DIR / 'summary_surface.csv'}")

# =============================================================================
# METHODOLOGY NOTE
# =============================================================================

methodology_note = f"""================================================================================
STEP 1: TENNIS-INFORMED PCA (8 VARS, 2 COMPONENTS)
================================================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Configuration: 8 variables, 2 components, 3 surfaces
Sample: 21 players per surface (Year-end Top 10, 2010-2024)

DIAGNOSTIC FINDINGS:
-------------------------------------------------------------------------------
{diagnostics_df[['surface', 'total_variance_2pc', 'n_p_ratio']].to_string(index=False)}

LIMITATION STATEMENT:
-------------------------------------------------------------------------------
Due to elite sample constraints (n=21 players per surface), 
2 components capture only 49-55% of playing style variance (n/p=2.6:1 ratio).

This baseline demonstrates that tennis-informed variable selection alone
is insufficient for statistical robustness with this sample size.

VARIABLES USED:
-------------------------------------------------------------------------------
{', '.join(SELECTED_VARS)}

CONCLUSION:
-------------------------------------------------------------------------------
Step 1 shows low explained variance. Step 2 will use statistically-selected
variables (assumption validator). Step 3 will perform exhaustive search
to find optimal variable combinations.

FILES GENERATED:
-------------------------------------------------------------------------------
- scores_all_surfaces.csv      # Player coordinates on PC1/PC2
- loadings_all_surfaces.csv    # Variable contributions
- variance_all_surfaces.csv    # Explained variance
- summary_surface.csv          # Sinner/Alcaraz vs Big 3
- diagnostics_report.csv       # Variance summary
- methodology_note.txt         # This file
================================================================================
"""

with open(OUTPUT_DIR / "methodology_note.txt", "w") as f:
    f.write(methodology_note)

print(f"  {OUTPUT_DIR / 'methodology_note.txt'}")

# =============================================================================
# PRINT SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("VARIANCE SUMMARY")
print("=" * 60)
for _, row in diagnostics_df.iterrows():
    print(f"  {row['surface']}: {row['total_variance_2pc']:.1%} (n/p={row['n_p_ratio']}:1)")

print("\n" + "=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nProceed to Step 2 (assumption validator) or Step 3 (exhaustive search).")
print("=" * 60)