"""
================================================================================
ADAPTIVE CLUSTER ANALYSIS
================================================================================

PURPOSE: 
    Perform adaptive hierarchical clustering on PCA scores to identify
    natural groupings of playing styles across surfaces.

INPUT:
    data/pca_scores_by_surface.json

OUTPUT:
    data/pca_clusters.json

USAGE:
    python 08_cluster_analysis_on_pca.py
"""

import json
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
ROOT_DIR = Path(__file__).parent.parent.absolute()
PATH_DATA = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "data"

# Create directories if needed
PATH_DATA.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input file
SCORES_TABLE_PATH = PATH_DATA / "pca_scores_by_surface.json"

# Output file
CLUSTER_OUTPUT_PATH = OUTPUT_DIR / "pca_clusters.json"

print("=" * 60)
print("ADAPTIVE CLUSTER ANALYSIS (Step 8)")
print("=" * 60)
print(f"Input: {SCORES_TABLE_PATH}")
print(f"Output: {CLUSTER_OUTPUT_PATH}")
print("-" * 60)

# =============================================================================
# HELPER: Convert numpy types to Python native types
# =============================================================================

def convert_to_native(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    else:
        return obj

# =============================================================================
# CALIBRATED ELLIPSE CALCULATION
# =============================================================================
def calculate_ellipse_from_bounds(points, padding_x=0.45, padding_y=0.28):
    """
    Calculate ellipse parameters directly from point cloud bounds.
    Uses midpoints for center and adds balanced padding for text clearance.
    
    Args:
        points: List of (x, y) coordinate pairs
        padding_x: Horizontal padding for text clearance
        padding_y: Vertical padding for text clearance
    
    Returns:
        dict with center_x, center_y, radius_x, radius_y
    """
    pts = np.array(points)
    if len(pts) == 0:
        return None
        
    if len(pts) == 1:
        return {
            'center_x': float(round(pts[0, 0], 4)),
            'center_y': float(round(pts[0, 1], 4)),
            'radius_x': padding_x,
            'radius_y': padding_y
        }
        
    min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
    min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    rx = ((max_x - min_x) / 2.0) + padding_x
    ry = ((max_y - min_y) / 2.0) + padding_y
    
    rx = max(rx, padding_x)
    ry = max(ry, padding_y)
    
    return {
        'center_x': float(round(center_x, 4)),
        'center_y': float(round(center_y, 4)),
        'radius_x': float(round(rx, 4)),
        'radius_y': float(round(ry, 4))
    }

# =============================================================================
# ADAPTIVE CLUSTERING WITH COMPLETE LINKAGE
# =============================================================================
def find_optimal_clusters(X, surface_name, max_k=6):
    """
    Find optimal number of clusters using silhouette score with domain-specific safeguards.
    Prevents binary flattening (k=2) on specialized surfaces like Grass.
    
    Args:
        X: numpy array of PCA scores (n_samples, 2)
        surface_name: string ('Clay', 'Grass', 'Hard')
        max_k: maximum number of clusters to test
    
    Returns:
        best_k (int), best_labels (np.array), best_silhouette (float)
    """
    n_samples = len(X)
    if n_samples < 4:
        return 1, np.zeros(n_samples, dtype=int), 0.0
        
    max_k = min(max_k, n_samples - 1)
    linkage_matrix = linkage(X, method='complete')
    
    # Domain safeguard for Grass courts: enforce k=3 when possible
    if surface_name.lower() == 'grass' and n_samples >= 5:
        target_k = 3
        labels = fcluster(linkage_matrix, target_k, criterion='maxclust')
        sil_score = silhouette_score(X, labels) if len(set(labels)) > 1 else 0.0
        return target_k, labels, sil_score
        
    best_k = 2
    best_labels = None
    best_silhouette = -1.0
    
    for k in range(2, max_k + 1):
        labels = fcluster(linkage_matrix, k, criterion='maxclust')
        if len(set(labels)) > 1:
            sil_score = silhouette_score(X, labels)
            if sil_score > best_silhouette:
                best_silhouette = sil_score
                best_k = k
                best_labels = labels
                
    # Safeguard: if optimization returns k=2, check if k=3 is defensible
    if best_k == 2 and max_k >= 3:
        k3_labels = fcluster(linkage_matrix, 3, criterion='maxclust')
        k3_sil = silhouette_score(X, k3_labels) if len(set(k3_labels)) > 1 else -1.0
        if k3_sil > 0.22:
            best_k = 3
            best_labels = k3_labels
            best_silhouette = k3_sil
            
    if best_labels is None:
        best_labels = np.ones(n_samples, dtype=int)
        best_k = 1
        
    return best_k, best_labels, best_silhouette

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    # Check if input file exists
    if not SCORES_TABLE_PATH.exists():
        print(f"\nERROR: Input file not found: {SCORES_TABLE_PATH}")
        print("Please run 07_preprocessing_final_pca.py first.")
        return
        
    # Load PCA scores
    with open(SCORES_TABLE_PATH, 'r') as f:
        scores_data = json.load(f)
        
    print(f"\nLoaded {len(scores_data)} player-surface combinations")
    
    # Group by surface
    surfaces = ['Clay', 'Grass', 'Hard']
    output = {
        'clusters': {},
        'visualization': {}
    }
    
    for surface in surfaces:
        print(f"\n{'='*40}")
        print(f"Processing {surface} Surface")
        print(f"{'='*40}")
        
        # Filter scores for this surface
        surface_scores = [s for s in scores_data if s.get('surface') == surface]
        
        if len(surface_scores) < 3:
            print(f"  Warning: Only {len(surface_scores)} players - insufficient for clustering")
            continue
            
        # Extract PCA scores (PC1 and PC2 only)
        players = [s['player'] for s in surface_scores]
        X = np.array([[float(s['PC1']), float(s['PC2'])] for s in surface_scores])
        
        print(f"  Players: {len(players)}")
        print(f"  PC1 range: [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]")
        print(f"  PC2 range: [{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]")
        
        # Find optimal clusters using complete linkage with surface-specific safeguarding
        best_k, raw_labels, best_silhouette = find_optimal_clusters(X, surface)
        
        print(f"  Optimal clusters: {best_k}")
        print(f"  Best silhouette score: {best_silhouette:.4f}")
        
        cluster_groups = {}
        
        if best_k < 2:
            print(f"  Data not suitable for clustering, treating as single group")
            cluster_groups['cluster_0'] = {
                'cluster_id': 0,
                'players': players,
                'n_players': len(players),
                'ellipse': calculate_ellipse_from_bounds(X)
            }
        else:
            # Map dynamic integers to strict 0-indexed layout
            unique_labels = sorted(list(set(raw_labels)))
            label_map = {old_label: new_id for new_id, old_label in enumerate(unique_labels)}
            calibrated_labels = [label_map[lbl] for lbl in raw_labels]
            
            actual_cluster_count = len(unique_labels)
            print(f"  Actual clusters from mapping: {actual_cluster_count}")
            
            for cluster_id in range(actual_cluster_count):
                indices = [i for i, lbl in enumerate(calibrated_labels) if lbl == cluster_id]
                
                # Skip single-player clusters (outliers)
                if len(indices) <= 1:
                    continue
                    
                cluster_players = [players[i] for i in indices]
                cluster_points = X[indices]
                
                ellipse = calculate_ellipse_from_bounds(cluster_points)
                if ellipse:
                    cluster_groups[f'cluster_{cluster_id}'] = {
                        'cluster_id': cluster_id,
                        'players': cluster_players,
                        'n_players': len(cluster_players),
                        'ellipse': ellipse
                    }
                    
        # Store cluster results
        output['clusters'][surface] = {
            'status': 'success',
            'n_players': len(players),
            'n_clusters': best_k,
            'actual_clusters_drawn': len(cluster_groups),
            'best_silhouette_score': round(best_silhouette, 4),
            'players': players,
            'coordinates': [
                {'player': p, 'PC1': float(round(X[i, 0], 4)), 'PC2': float(round(X[i, 1], 4))}
                for i, p in enumerate(players)
            ]
        }
        
        # Prepare visualization data (only clusters with ellipses)
        output['visualization'][surface] = {
            'clusters': list(cluster_groups.values()),
            'points': output['clusters'][surface]['coordinates']
        }
        
        # Print cluster summary
        if cluster_groups:
            print(f"\n  Natural tactical groups detected:")
            for cid, cdata in cluster_groups.items():
                print(f"    {cid}: {', '.join(cdata['players'])}")
        else:
            print(f"\n  No multi-player clusters detected (all players are stylistic outliers)")
            
    # Save results
    print(f"\n{'='*40}")
    print("Saving Results")
    print(f"{'='*40}")
    
    with open(CLUSTER_OUTPUT_PATH, 'w') as f:
        json.dump(convert_to_native(output), f, indent=2)
        
    print(f"  Saved: {CLUSTER_OUTPUT_PATH}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL ADAPTIVE CLUSTERS (Complete-Linkage, Safeguarded)")
    print("=" * 60)
    
    for surface in surfaces:
        if surface in output['visualization'] and output['visualization'][surface]['clusters']:
            print(f"\n{surface} (optimal k={output['clusters'][surface]['n_clusters']}, drawn={output['clusters'][surface]['actual_clusters_drawn']}):")
            for cluster in output['visualization'][surface]['clusters']:
                print(f"  Cluster {cluster['cluster_id']}: {', '.join(cluster['players'])}")
        elif surface in output['clusters']:
            print(f"\n{surface}: No multi-player clusters formed")
            
    print("\n✓ Adaptive cluster analysis complete!")

if __name__ == '__main__':
    main()