"""
================================================================================
FINAL PCA DATA PREPARATION
================================================================================
PURPOSE: 
    Combine Step 2 (Clay + Grass) and Step 4 (Hard) PCA results
    into a single dataset for dashboard visualization.

INPUT:
    results/step2_validator_4vars/scores_all_surfaces.csv
    results/step2_validator_4vars/loadings_all_surfaces.csv
    results/step4_final_optimized_4vars/scores_all_surfaces.csv
    results/step4_final_optimized_4vars/loadings_all_surfaces.csv

OUTPUT:
    data/pca_scores_by_surface.csv
    data/pca_loadings_by_surface.csv

USAGE:
    python 07_preprocessing_final_pca.py
"""

import pandas as pd
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
ROOT_DIR = Path(__file__).parent.parent.absolute()
PATH_RESULTS = ROOT_DIR / "results"
PATH_RESULTS_STEP2 = PATH_RESULTS / "step2_validator_4vars"
PATH_RESULTS_STEP4 = PATH_RESULTS / "step4_final_optimized_4vars"
OUTPUT_DIR = ROOT_DIR / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File names
FILE_NAME_SCORES = 'scores_all_surfaces.csv'
FILE_NAME_LOADINGS = 'loadings_all_surfaces.csv'
FILE_NAME_VARIANCE = 'variance_all_surfaces.csv'

print("=" * 60)
print("FINAL PCA DATA PREPARATION (Step 7)")
print("=" * 60)
print(f"Step 2 input: {PATH_RESULTS_STEP2}")
print(f"Step 4 input: {PATH_RESULTS_STEP4}")
print(f"Output dir: {OUTPUT_DIR}")
print("-" * 60)

# =============================================================================
# LOAD DATA
# =============================================================================
try:
    # Step 2 (Clay + Grass)
    df_scores_step2 = pd.read_csv(
        PATH_RESULTS_STEP2 / FILE_NAME_SCORES, sep=','
    )
    df_loadings_step2 = pd.read_csv(
        PATH_RESULTS_STEP2 / FILE_NAME_LOADINGS, sep=','
    )    
    df_variance_step2 = pd.read_csv(
        PATH_RESULTS_STEP2 / FILE_NAME_VARIANCE, sep=','
    )
    print("✓ Loaded Step 2 results")

    # Step 4 (Hard)
    df_scores_step4 = pd.read_csv(
        PATH_RESULTS_STEP4 / FILE_NAME_SCORES, sep=','
    )
    df_loadings_step4 = pd.read_csv(
        PATH_RESULTS_STEP4 / FILE_NAME_LOADINGS, sep=','
    )
    df_variance_step4 = pd.read_csv(
        PATH_RESULTS_STEP4 / FILE_NAME_VARIANCE, sep=','
    )
    print("✓ Loaded Step 4 results")

except FileNotFoundError as e:
    print(f"\nError loading files: {e}")
    print("\nMake sure you have run:")
    print("  - 04_pca_step2_validator_4vars.py")
    print("  - 06_pca_step4_final_optimized.py")
    exit()

# =============================================================================
# COMBINE SCORES: Step 2 (Clay + Grass) + Step 4 (Hard)
# =============================================================================
df_scores_step2_filtered = df_scores_step2[
    df_scores_step2['surface'].isin(['Clay', 'Grass'])
]
df_scores_step4_filtered = df_scores_step4[
    df_scores_step4['surface'] == 'Hard'
]
df_scores_final = pd.concat([df_scores_step2_filtered, df_scores_step4_filtered], axis=0)

# =============================================================================
# COMBINE LOADINGS: Step 2 (Clay + Grass) + Step 4 (Hard)
# =============================================================================
df_loadings_step2_filtered = df_loadings_step2[
    df_loadings_step2['surface'].isin(['Clay', 'Grass'])
]
df_loadings_step4_filtered = df_loadings_step4[
    df_loadings_step4['surface'] == 'Hard'
]
df_loadings_final = pd.concat([df_loadings_step2_filtered, df_loadings_step4_filtered], axis=0)

# =============================================================================
# COMBINE VARIANCE: Step 2 (Clay + Grass) + Step 4 (Hard)
# =============================================================================
df_variance_step2_filtered = df_variance_step2[
    df_variance_step2['surface'].isin(['Clay', 'Grass'])
]
df_variance_step4_filtered = df_variance_step4[
    df_variance_step4['surface'] == 'Hard'
]
df_variance_final = pd.concat([df_variance_step2_filtered, df_variance_step4_filtered], axis=0)

# =============================================================================
# SAVE FINAL DATASETS FOR VISUALIZATION
# =============================================================================
df_scores_final.to_json(
    OUTPUT_DIR / 'pca_scores_by_surface.json', orient='records'
    )

df_loadings_final.to_json(
    OUTPUT_DIR / 'pca_loadings_by_surface.json', orient='records'
    )

df_variance_final.to_json(
    OUTPUT_DIR / 'pca_variance_by_surface.json', orient='records'
    )

print("-" * 60)
print("SAVED:")
print(f"  {OUTPUT_DIR / 'pca_scores_by_surface.json'}")
print(f"  {OUTPUT_DIR / 'pca_loadings_by_surface.json'}")
print(f"  {OUTPUT_DIR / 'pca_variance_by_surface.json'}")
print("-" * 60)

#------------------------------------------------------------------------------
# SUMMARY
#------------------------------------------------------------------------------
print("\nFINAL DATASET SUMMARY:")
print(f"  Scores: {len(df_scores_final)} rows")
print(f"    - Clay: {len(df_scores_final[df_scores_final['surface'] == 'Clay'])}")
print(f"    - Grass: {len(df_scores_final[df_scores_final['surface'] == 'Grass'])}")
print(f"    - Hard: {len(df_scores_final[df_scores_final['surface'] == 'Hard'])}")
print(f"  Loadings: {len(df_loadings_final)} rows")
print("=" * 60)
print("\nReady for dashboard (app.py)")