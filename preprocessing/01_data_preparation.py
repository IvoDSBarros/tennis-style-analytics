"""
================================================
PCA DATA PREPARATION
===============================================

DESCRIPTION:
    This script processes Match Charting Project data from Jeff Sackmann to
    create a PCA-ready dataset of playing style metrics for ATP players.

ACKNOWLEDGMENT:
    Data sourced from Jeff Sackmann's Tennis Match Charting Project
    https://github.com/JeffSackmann/tennis_MatchChartingProject
    Used under the terms of the MIT License

METHODOLOGY:
    1. Identify year-end Top 10 players (2010-2024)
    2. Filter matches: Grand Slams + Masters 1000 + Grass 500 (Round of 32 onwards)
    3. Select players with >=5 matches on each surface
    4. Stratified sampling: 8 matches per player per surface
    5. Calculate 14 playing style metrics
    6. Output: 21 players x 3 surfaces = 63 rows

REQUIRED DATA:
    Download from Jeff Sackmann's GitHub repositories:
    - https://github.com/JeffSackmann/tennis_atp
    - https://github.com/JeffSackmann/tennis_MatchChartingProject

    Place all CSV files in ./data/raw/ directory

OUTPUT:
    tennis_metrics_by_player.csv - Ready for PCA analysis
        Columns: surface, player, and 14 playing style metrics

USAGE:
    python 01_data_preparation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories (relative to script location)
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / "data"                 
DATA_RAW_DIR = DATA_DIR / "raw"               # Raw Sackmann files here

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

# Output file name
OUTPUT_FILE_CSV = "tennis_metrics_by_player.csv"
OUTPUT_FILE_JSON = "tennis_metrics_by_player.json"

# File names (expected in DATA_RAW_DIR)
FILES = {
    'players': 'atp_players.csv',
    'matches': 'charting-m-matches.csv',
    'overview': 'charting-m-stats-Overview.csv',
    'serve': 'charting-m-stats-KeyPointsServe.csv',
    'serve_basics': 'charting-m-stats-ServeBasics.csv',
    'return': 'charting-m-stats-KeyPointsReturn.csv',
    'net': 'charting-m-stats-NetPoints.csv',
    'shot_dir': 'charting-m-stats-ShotDirection.csv',
    'shot_types': 'charting-m-stats-ShotTypes.csv'
}

# Row/Set contexts (defines which rows to aggregate)
CONTEXT = {
    'overview': 'Total',           # column: 'set'
    'serve': 'STotal',             # column: 'row'
    'serve_basics': 'Total',       # column: 'row'
    'return': 'RTotal',            # column: 'row'
    'break_points': 'BPO',         # column: 'row'
    'net': 'NetPoints',            # column: 'row'
    'shot_dir': 'Total',           # column: 'row'
    'shot_types': 'Total',         # column: 'row'
    'slice': 'Sl',                 # column: 'row'
    'drop_shots': 'Dr'             # column: 'row'
}

# Fields to extract from each file
FIELDS = {
    'overview': ['match_id', 'player', 'bk_pts', 'bp_saved', 'winners', 'unforced'],
    'serve': ['match_id', 'player', 'pts', 'pts_won', 'first_in', 'aces'],
    'serve_basics': ['match_id', 'player', 'pts', 'unret', 'wide', 'body', 't'],
    'return': ['match_id', 'player', 'pts', 'pts_won'],
    'break_points': ['match_id', 'player', 'pts', 'pts_won'],
    'net': ['match_id', 'player', 'net_pts', 'pts_won'],
    'shot_dir': ['match_id', 'player', 'crosscourt', 'down_the_line'],
    'shot_types': ['match_id', 'player', 'shots']
}

# Tournament filters
TOURNAMENTS = {
    'grand_slams': ['Australian Open', 'US Open', 'Roland Garros', 'Wimbledon'],
    'masters': ['Paris Masters', 'Shanghai Masters', 'Cincinnati Masters', 'Canada Masters',
                'Rome Masters', 'Madrid Masters', 'Monte Carlo Masters', 'Miami Masters',
                'Indian Wells Masters', 'Paris_Masters', 'Miami_Masters', 'Madrid_Masters',
                'Cincinnati_Masters'],
    'grass_500': ['Halle', 'Queens Club', "Queen's Club"]
}
TARGET_TOURNAMENTS = TOURNAMENTS['grand_slams'] + TOURNAMENTS['masters'] + TOURNAMENTS['grass_500']
TARGET_ROUNDS = ['R32', 'R16', 'QF', 'SF', 'F']
MISSING_STATS = ['20120124-M-Australian_Open-QF-David_Ferrer-Novak_Djokovic']

# Sampling parameters
RANDOM_SEED = 42
SAMPLE_SIZE = 8
MIN_MATCHES_PER_SURFACE = 5
START_YEAR = '2010-01-01'

# ============================================================================
# DATA LOADING WITH ERROR HANDLING
# ============================================================================

def load_data():
    """Load all required CSV files from DATA_RAW_DIR."""
    data = {}
    missing = []
    
    for name, filename in FILES.items():
        filepath = DATA_RAW_DIR / filename
        if not filepath.exists():
            missing.append(filename)
        else:
            data[name] = pd.read_csv(filepath, sep=',', low_memory=False)
    
    if missing:
        raise FileNotFoundError(
            f"Missing required files: {missing}\n"
            f"Please download from Jeff Sackmann's GitHub repositories:\n"
            f"  https://github.com/JeffSackmann/tennis_atp\n"
            f"  https://github.com/JeffSackmann/tennis_MatchChartingProject\n"
            f"and place all CSV files in: {DATA_RAW_DIR}"
        )
    
    # Load ranking files (dynamic naming)
    rank_files = list(DATA_RAW_DIR.glob("atp_rankings_*.csv"))
    if rank_files:
        data['ranks'] = pd.concat([pd.read_csv(f, sep=',') for f in rank_files], axis=0)
    else:
        raise FileNotFoundError(f"No ranking files found in {DATA_RAW_DIR}")
    
    return data

# ============================================================================
# YEAR-END TOP 10 PLAYERS
# ============================================================================

def get_year_end_top10(df_ranks, df_players):
    """Extract players ranked Top 10 at year-end (December)."""
    df = df_ranks.copy()
    df['is_dec'] = df['ranking_date'].astype(str).str[4:6] == '12'
    df['is_top10'] = df['rank'] <= 10
    df = df[df['is_dec'] & df['is_top10']]
    df = df.rename(columns={'player': 'player_id'})
    df = df.merge(df_players[['player_id', 'name_first', 'name_last']], on='player_id', how='left')
    df['player_name'] = df['name_first'] + " " + df['name_last']
    return df['player_name'].unique().tolist()

# ============================================================================
# MATCH FILTERING
# ============================================================================

def filter_matches(df_matches, top10_players):
    """Filter matches by tournament, date, round, and player quality."""
    df = df_matches.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
    
    mask = (df['Tournament'].isin(TARGET_TOURNAMENTS) &
            (df['Date'] >= START_YEAR) &
            df['Round'].isin(TARGET_ROUNDS) &
            ~df['match_id'].isin(MISSING_STATS) &
            (df['Player 1'].isin(top10_players) | df['Player 2'].isin(top10_players)))
    
    return df[mask].copy()

# ============================================================================
# PLAYER QUALIFICATION (MIN MATCHES PER SURFACE)
# ============================================================================

def get_qualified_players(df_matches, top10_players):
    """Return players with minimum matches on all three surfaces."""
    p1 = pd.crosstab(df_matches['Player 1'], df_matches['Surface'])
    p2 = pd.crosstab(df_matches['Player 2'], df_matches['Surface'])
    player_surf = p1.add(p2, fill_value=0).reset_index().rename(columns={'index': 'Player'})
    
    mask = (player_surf['Player'].isin(top10_players) &
            (player_surf['Clay'] >= MIN_MATCHES_PER_SURFACE) &
            (player_surf['Hard'] >= MIN_MATCHES_PER_SURFACE) &
            (player_surf['Grass'] >= MIN_MATCHES_PER_SURFACE))
    
    return player_surf[mask]['Player'].unique().tolist()

# ============================================================================
# STRATIFIED SAMPLING
# ============================================================================

def stratified_sample(df_matches, qualified_players, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    """Sample exactly N matches per player per surface."""
    # Create player-match lookup from both columns
    p1 = df_matches[['match_id', 'Player 1', 'Surface']].rename(columns={'Player 1': 'Player'})
    p2 = df_matches[['match_id', 'Player 2', 'Surface']].rename(columns={'Player 2': 'Player'})
    all_matches = pd.concat([p1, p2], ignore_index=True)
    all_matches = all_matches[all_matches['Player'].isin(qualified_players)]
    
    # Sort for deterministic ordering (ensures reproducibility across environments)
    all_matches = all_matches.sort_values(['Player', 'Surface', 'match_id']).reset_index(drop=True)
    
    np.random.seed(seed)
    sampled = (all_matches.groupby(['Player', 'Surface'], group_keys=False)
               .apply(lambda x: x.sample(n=min(sample_size, len(x)), random_state=seed))
               .reset_index(drop=True))
    sampled.columns = [c.lower() for c in sampled.columns]
    
    return sampled

# ============================================================================
# METRIC CALCULATION
# ============================================================================

def calculate_entropy(wide, body, t):
    """Calculate Shannon entropy for serve placement variety."""
    total = wide + body + t
    if total == 0:
        return 0.0
    p_wide = wide / total
    p_body = body / total
    p_t = t / total
    return -(
        p_wide * np.log(p_wide + 1e-10) +
        p_body * np.log(p_body + 1e-10) +
        p_t * np.log(p_t + 1e-10)
    )


def compute_metrics(sampled_matches, data):
    """Calculate all playing style metrics per player per surface."""
    results = []
    
    # Pre-compute total shots for each player-surface (denominator for slice/drop)
    total_shots_list = []
    for surface in sampled_matches['surface'].unique():
        surf_matches = sampled_matches[sampled_matches['surface'] == surface]
        st = data['shot_types'].merge(surf_matches[['match_id', 'player']], on=['match_id', 'player'])
        st = st[st['row'] == CONTEXT['shot_types']]
        total = st.groupby('player')['shots'].sum().reset_index().rename(columns={'shots': 'total_shots'})
        total['surface'] = surface
        total_shots_list.append(total)
    total_shots_df = pd.concat(total_shots_list, ignore_index=True)
    
    for surface in sampled_matches['surface'].unique():
        print(f"  Processing {surface}...")
        surf_matches = sampled_matches[sampled_matches['surface'] == surface]
        match_ids = surf_matches['match_id'].tolist()
        players = surf_matches['player'].tolist()
        
        # Helper to filter and merge
        def get_stats(df, fields, row_ctx=None, set_ctx=None):
            filtered = df.merge(surf_matches[['match_id', 'player']], on=['match_id', 'player'])
            if row_ctx:
                filtered = filtered[filtered['row'] == row_ctx]
            if set_ctx:
                filtered = filtered[filtered['set'] == set_ctx]
            return filtered[fields]
        
        # 1. Overview (winner/error ratio, break points saved)
        ov = get_stats(data['overview'], FIELDS['overview'], set_ctx=CONTEXT['overview'])
        ov_agg = ov.groupby('player')[['bk_pts', 'bp_saved', 'winners', 'unforced']].sum().reset_index()
        ov_agg['winner_error_ratio'] = ov_agg['winners'] / ov_agg['unforced'].replace(0, np.nan)
        ov_agg['break_points_saved_pct'] = ov_agg['bp_saved'] / ov_agg['bk_pts'].replace(0, np.nan)
        ov_agg = ov_agg[['player', 'winner_error_ratio', 'break_points_saved_pct']]
        
        # 2. Serve metrics (ace rate, 1st serve %, 1st serve points won)
        sv = get_stats(data['serve'], FIELDS['serve'], row_ctx=CONTEXT['serve'])
        sv_agg = sv.groupby('player')[['pts', 'pts_won', 'first_in', 'aces']].sum().reset_index()
        sv_agg['ace_rate'] = sv_agg['aces'] / sv_agg['pts'].replace(0, np.nan)
        sv_agg['first_serve_pct'] = sv_agg['first_in'] / sv_agg['pts'].replace(0, np.nan)
        sv_agg['first_serve_pts_won'] = sv_agg['pts_won'] / sv_agg['pts'].replace(0, np.nan)
        sv_agg = sv_agg[['player', 'ace_rate', 'first_serve_pct', 'first_serve_pts_won']]
        
        # 3. Serve basics (unreturned rate, placement variety via entropy)
        sb = get_stats(data['serve_basics'], FIELDS['serve_basics'], row_ctx=CONTEXT['serve_basics'])
        sb_agg = sb.groupby('player')[['pts', 'unret', 'wide', 'body', 't']].sum().reset_index()
        sb_agg['unreturned_serve_rate'] = sb_agg['unret'] / sb_agg['pts'].replace(0, np.nan)
        sb_agg['serve_placement_variety'] = sb_agg.apply(
            lambda x: calculate_entropy(x['wide'], x['body'], x['t']), axis=1)
        sb_agg = sb_agg[['player', 'unreturned_serve_rate', 'serve_placement_variety']]
        
        # 4. Return points won %
        ret = get_stats(data['return'], FIELDS['return'], row_ctx=CONTEXT['return'])
        ret_agg = ret.groupby('player')[['pts', 'pts_won']].sum().reset_index()
        ret_agg['return_points_won_pct'] = ret_agg['pts_won'] / ret_agg['pts'].replace(0, np.nan)
        ret_agg = ret_agg[['player', 'return_points_won_pct']]
        
        # 5. Break point conversion %
        bp = get_stats(data['return'], FIELDS['break_points'], row_ctx=CONTEXT['break_points'])
        bp_agg = bp.groupby('player')[['pts', 'pts_won']].sum().reset_index()
        bp_agg['break_point_conversion_pct'] = bp_agg['pts_won'] / bp_agg['pts'].replace(0, np.nan)
        bp_agg = bp_agg[['player', 'break_point_conversion_pct']]
        
        # 6. Net points won %
        net = get_stats(data['net'], FIELDS['net'], row_ctx=CONTEXT['net'])
        net_agg = net.groupby('player')[['net_pts', 'pts_won']].sum().reset_index()
        net_agg['net_points_won_pct'] = net_agg['pts_won'] / net_agg['net_pts'].replace(0, np.nan)
        net_agg = net_agg[['player', 'net_points_won_pct']]
        
        # 7. Shot direction (crosscourt ratio)
        sd = get_stats(data['shot_dir'], FIELDS['shot_dir'], row_ctx=CONTEXT['shot_dir'])
        sd_agg = sd.groupby('player')[['crosscourt', 'down_the_line']].sum().reset_index()
        total_dir = sd_agg['crosscourt'] + sd_agg['down_the_line']
        sd_agg['crosscourt_ratio'] = sd_agg['crosscourt'] / total_dir.replace(0, np.nan)
        sd_agg = sd_agg[['player', 'crosscourt_ratio']]
        
        # 8. Slice rate
        sl = get_stats(data['shot_types'], FIELDS['shot_types'], row_ctx=CONTEXT['slice'])
        sl_agg = sl.groupby('player')['shots'].sum().reset_index()
        total_surf = total_shots_df[total_shots_df['surface'] == surface][['player', 'total_shots']]
        sl_agg = sl_agg.merge(total_surf, on='player', how='left')
        sl_agg['slice_rate'] = sl_agg['shots'] / sl_agg['total_shots'].replace(0, np.nan)
        sl_agg = sl_agg[['player', 'slice_rate']]
        
        # 9. Drop shots (frequency and effectiveness)
        dr = get_stats(data['shot_types'], FIELDS['shot_types'], row_ctx=CONTEXT['drop_shots'])
        dr_agg = dr.groupby('player')[['shots', 'winners', 'induced_forced']].sum().reset_index()
        dr_agg = dr_agg.merge(total_surf, on='player', how='left')
        dr_agg['drop_shot_frequency'] = dr_agg['shots'] / dr_agg['total_shots'].replace(0, np.nan)
        dr_agg['drop_shot_effectiveness'] = (dr_agg['winners'] + dr_agg['induced_forced']) / dr_agg['shots'].replace(0, np.nan)
        dr_agg = dr_agg[['player', 'drop_shot_frequency', 'drop_shot_effectiveness']]
        
        # Merge all metrics for this surface
        surface_result = (ov_agg.merge(sv_agg, on='player', how='outer')
                               .merge(sb_agg, on='player', how='outer')
                               .merge(ret_agg, on='player', how='outer')
                               .merge(bp_agg, on='player', how='outer')
                               .merge(net_agg, on='player', how='outer')
                               .merge(sd_agg, on='player', how='outer')
                               .merge(sl_agg, on='player', how='outer')
                               .merge(dr_agg, on='player', how='outer'))
        surface_result.insert(0, 'surface', surface)
        results.append(surface_result)
    
    return pd.concat(results, ignore_index=True).fillna(0)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 60)
    print("TENNIS PCA DATA PREPARATION")
    print("=" * 60)
    print(f"Raw data directory: {DATA_RAW_DIR}")
    print(f"Output directory: {DATA_DIR}")
    print("-" * 60)
    
    # Load data
    print("Loading data files...")
    data = load_data()
    print("  All files loaded")
    
    # Get year-end Top 10 players
    print("Identifying year-end Top 10 players...")
    top10 = get_year_end_top10(data['ranks'], data['players'])
    print(f"  Found {len(top10)} players")
    
    # Filter matches
    print("Filtering matches (2010-2024, Slams/Masters/Grass500, R32+)...")
    matches_filt = filter_matches(data['matches'], top10)
    print(f"  Filtered to {len(matches_filt)} matches")
    
    # Get qualified players (>=5 matches per surface)
    print(f"Filtering players with >= {MIN_MATCHES_PER_SURFACE} matches per surface...")
    qualified = get_qualified_players(matches_filt, top10)
    print(f"  Qualified: {len(qualified)} players")
    
    # Stratified sampling
    print(f"Sampling {SAMPLE_SIZE} matches per player per surface...")
    sampled = stratified_sample(matches_filt, qualified)
    print(f"  Sampled {len(sampled)} matches total")
    
    # Calculate metrics
    print("Calculating 14 playing style metrics...")
    df_metrics = compute_metrics(sampled, data)
    
    # Sort and save
    df_metrics = df_metrics.sort_values(['surface', 'player']).reset_index(drop=True)
    output_path_csv = DATA_DIR / OUTPUT_FILE_CSV
    output_path_json = DATA_DIR / OUTPUT_FILE_JSON
    df_metrics.to_csv(output_path_csv, index=False, encoding='utf-8', sep=';')
    df_metrics.to_json(output_path_json, orient='records')
    
    print("-" * 60)
    print(f"Saved {len(df_metrics)} rows to: {output_path_csv}")
    print(f"  Players: {df_metrics['player'].nunique()}")
    print(f"  Surfaces: {df_metrics['surface'].unique().tolist()}")
    print(f"  Metrics: {len([c for c in df_metrics.columns if c not in ['surface', 'player']])}")
    print("=" * 60)

if __name__ == "__main__":
    main()