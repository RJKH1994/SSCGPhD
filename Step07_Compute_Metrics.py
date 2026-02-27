#!/usr/bin/env python3

"""

Step07_Compute_Metrics.py

Authors: Ryan Howard (@RJKH1994) & Matt Bridge (@MWBRIDGE)
Adapted from: Guangze Zhang (@latilongitude)

- The purpose of this code is to compute team-level (and optionally opposition) tactical metrics from processed GPS tracking files (e.g., Zeroed_*, Normalized_Zeroed_*), and save new "Processed_*" metric outputs.
- The code is designed to work with:
    - Single-team match folders (one set of files per Output folder)
    - Two-team match folders where TEAMA_* and TEAMB_* exist as siblings under a common parent directory
- The pipeline for the workflow is:
    1) Discover all processable Output folders (Output and/or Output/Normalized) under MASTER_FOLDER.
    2) Run the single-team pipeline for each output folder:
        - Standardise player columns
        - Compute centroid, surface area, surface area, stretch index, interpersonal distances, and unit metrics
        - Compute directionality metrics (anisotropy/orientation/direction) relative to the team centroid
        - Save metrics as Processed_<input_file>.csv in the same folder
    3) (Optional) Run the opposition pipeline for TEAMA/TEAMB sibling pairs:
        - Pair opposite phases (IP↔OOP, AT↔DT)
        - Time-align both teams and compute within-team and between-team metrics
        - Save paired opposition outputs into the chosen output folder

Inputs:
- Output or Output/Normalized folders containing files ending in "filled.csv" with prefixes:
    - Zeroed_
    - Normalized_Zeroed_
- These input files are expected to be team tracking time series:
    - A Time column
    - Player coordinate columns ending in *_x and *_y

Outputs:
- Single-team:
    - Processed_<original_filename>.csv saved next to the input file
- Opposition (two-team):
    - Processed_Paired_OppPhases_<desc>.csv saved into the chosen output folder

Notes:
- Input files are assumed to be pre-cleaned and already sampled appropriately (e.g., 10 Hz) and optionally filled.
- The single-team pipeline drops the Half column if present (to avoid affecting dropna operations).
- The opposition pipeline pairs phases based on PHASE_OPPOSITES and aligns by Time (exact match unless tolerance is set).
- The script avoids reprocessing any file that already starts with "Processed_".

"""

import os
import re
import math
import itertools
import statistics as stat
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from shapely.geometry import Polygon


## CONFIGURATION

MASTER_FOLDER = '/Users/ryanhoward/Desktop/Data' # Change this to the master folder containing individual game folders

# Allow either of these output locations under each match/team folder:
#   - <root>/Output
#   - <root>/Output/Normalized
OUTPUT_CANDIDATES = [
    os.path.join('Output', 'Normalized'),
    'Output',
]

ASSUME_ATTACK_X_POS = True
RUN_OPPOSITION_METRICS = True

# 0 = exact match on Time only
# >0 = allow nearest neighbour merge within tolerance seconds
TIME_TOLERANCE_SEC = 0

# Phase pairing rules for opposition:
# - In possession aligns with opponent out of possession
# - Attacking transition aligns with opponent defensive transition
PHASE_OPPOSITES = {"IP": "OOP", "OOP": "IP", "AT": "DT", "DT": "AT"}


# FUNCTIONS #

## File filters (what counts as "processable" data)

# FUNCTION 1: Check whether a folder contains processable input tracking files

def has_processable_files(folder_path: str) -> bool:
    """
    Return True if folder contains at least one input file ending in 'filled.csv' that starts with Zeroed_ or Normalized_Zeroed_, and is not already Processed_*.
    """
    try:
        for f in os.listdir(folder_path):
            if not f.endswith("filled.csv"):
                continue
            if f.startswith("Processed_"):
                continue
            if f.startswith("Zeroed_") or f.startswith("Normalized_Zeroed_"):
                return True
    except Exception:
        return False
    return False


# FUNCTION 2: Label output folder type for pairing (Output vs Output/Normalized)

def folder_kind(path: str) -> str:
    """Return 'Normalized' if folder is Output/Normalized, otherwise 'Output'."""
    norm_rel = os.path.join("Output", "Normalized")
    if path.endswith(norm_rel) or (os.sep + norm_rel) in path:
        return "Normalized"
    return "Output"


# FUNCTION 3: Identify candidate output folders under a base folder

def get_candidate_outputs(base_folder: str) -> List[str]:
    """
    Given any folder (e.g., TEAMA_*), return any existing Output candidates under it (Output and/or Output/Normalized) that contain processable files.
    """
    outs = []
    for rel in OUTPUT_CANDIDATES:
        p = os.path.join(base_folder, rel)
        if os.path.isdir(p) and has_processable_files(p):
            outs.append(p)
    return outs


## Master discovery (supports single-team games and TEAMA/TEAMB siblings)

# FUNCTION 4: Discover all output folders and TEAMA/TEAMB sibling pairs for opposition

def discover_outputs_and_pairs(master_folder: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Returns:
    - single_outputs: folders under master that look like Output or Output/Normalized with processable files
    - paired_outputs: list of (teamA_output, teamB_output, label) for opposition processing

    Opposition pairing rules:
    - Look for sibling folders TEAMA_* and TEAMB_* under the same parent directory.
    - For each sibling pair, pair Output vs Output and/or Normalized vs Normalized if both exist.
    """
    if not os.path.isdir(master_folder):
        raise FileNotFoundError(f"MASTER_FOLDER does not exist: {master_folder}")

    single_outputs_set = set()
    paired_outputs: List[Tuple[str, str, str]] = []

    # Single-team discovery:
    # Any folder that is literally named Output or Normalized and contains processable files.
    # Note: this captures both ".../Output" and ".../Output/Normalized" during os.walk.
    for root, dirs, files in os.walk(master_folder):
        base = os.path.basename(root)
        if base == "Output" or base == "Normalized":
            if has_processable_files(root):
                single_outputs_set.add(root)

    # Pair discovery:
    # Look in each directory for TEAMA_* and TEAMB_* siblings, then match by the remainder of the folder name.
    for root, dirs, files in os.walk(master_folder):
        teamA_children = [d for d in dirs if d.upper().startswith("TEAMA_")]
        teamB_children = [d for d in dirs if d.upper().startswith("TEAMB_")]
        if not teamA_children or not teamB_children:
            continue

        a_map: Dict[str, str] = {}
        b_map: Dict[str, str] = {}

        for d in teamA_children:
            rest = d[len("TEAMA_"):]
            a_map[rest] = os.path.join(root, d)

        for d in teamB_children:
            rest = d[len("TEAMB_"):]
            b_map[rest] = os.path.join(root, d)

        common = sorted(set(a_map.keys()) & set(b_map.keys()))
        if not common:
            continue

        for rest in common:
            a_folder = a_map[rest]
            b_folder = b_map[rest]

            a_outs = {folder_kind(p): p for p in get_candidate_outputs(a_folder)}
            b_outs = {folder_kind(p): p for p in get_candidate_outputs(b_folder)}

            # Also add both team output folders to single processing set.
            for p in a_outs.values():
                single_outputs_set.add(p)
            for p in b_outs.values():
                single_outputs_set.add(p)

            # Pair by matching output "kinds"
            for kind in ["Normalized", "Output"]:
                if kind in a_outs and kind in b_outs:
                    label = f"{os.path.basename(root)}__{rest}__{kind}"
                    paired_outputs.append((a_outs[kind], b_outs[kind], label))

    single_outputs = sorted(single_outputs_set)
    return single_outputs, paired_outputs


## Common helpers

# FUNCTION 5: Detect player coordinate pairs (columns ending in _x / _y)

def detect_player_xy_pairs(columns):
    """
    Detect (x, y) coordinate column pairs from a list of columns.

    Returns:
    - List of tuples (x_col, y_col) in file order.
    """
    cols = list(columns)
    x_cols = [c for c in cols if str(c).endswith('_x')]
    pairs = []
    for xc in x_cols:
        yc = xc[:-2] + '_y'
        if yc in cols:
            pairs.append((xc, yc))
    pairs.sort(key=lambda p: cols.index(p[0]))
    return pairs


# SINGLE-TEAM PIPELINE #

## Standardisation and core geometry metrics

# FUNCTION 6: Rename players to a standard player_1...player_n convention (preserves original labels separately)

def rename_players(df):
    """
    Standardise player coordinate columns to player_1_x/player_1_y... for consistent downstream indexing.

    Returns:
    - df_renamed: DataFrame with standardised player column names
    - orig_position_labels: list of original position/player stems inferred from original *_x columns
    """
    first_cols = df.columns[:1].tolist()     # typically ['Time']
    other_cols = df.columns[1:]

    pairs = detect_player_xy_pairs(df.columns)
    orig_position_labels = [x.split('_')[0] for (x, y) in pairs]

    new_columns = first_cols.copy()
    player_counter = 1

    for col in other_cols:
        if str(col).endswith('_x'):
            new_columns.append(f'player_{player_counter}_x')
        elif str(col).endswith('_y'):
            new_columns.append(f'player_{player_counter}_y')
            player_counter += 1
        else:
            new_columns.append(col)

    if len(new_columns) != len(df.columns):
        raise ValueError("Renaming mismatch: new columns length differs from original.")

    df_renamed = df.copy()
    df_renamed.columns = new_columns

    # Drop rows with any NaNs (ensures convex hull and IPD operations are valid frame-by-frame)
    df_renamed = df_renamed.dropna().reset_index(drop=True)

    return df_renamed, orig_position_labels


# FUNCTION 7: Compute centroid / hull metrics for a single time row

def compute_metrics(row: pd.Series):
    """
    Compute:
    - Team convex hull centroid (cen_x, cen_y)
    - Convex hull area (surface_area)
    - Bounding box (minx, miny, maxx, maxy)

    If fewer than 3 valid player points exist, returns NaNs.
    """
    pairs = detect_player_xy_pairs(row.index)

    coords = []
    for xc, yc in pairs:
        x = row.get(xc, np.nan)
        y = row.get(yc, np.nan)
        if pd.notna(x) and pd.notna(y):
            coords.append((float(x), float(y)))

    if len(coords) < 3:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    hull = Polygon(coords).convex_hull
    cen = hull.centroid
    minx, miny, maxx, maxy = hull.bounds
    area = hull.area if hull.geom_type == "Polygon" else 0.0

    return float(cen.x), float(cen.y), float(area), float(minx), float(miny), float(maxx), float(maxy)


# FUNCTION 8: Build a time-series DataFrame of centroid and hull metrics for a whole match file

def build_cen_dataframe(team_df):
    """
    Build a per-frame metrics DataFrame using compute_metrics(row).

    Adds derived columns:
    - length, width (bounding box extents)
    - LpW (length-to-width ratio)
    """
    Cen = pd.DataFrame(columns=["cen_x", "cen_y", "surface_area", "minx", "miny", "maxx", "maxy"])
    for i in range(len(team_df)):
        try:
            Cen.loc[len(Cen)] = compute_metrics(team_df.loc[i])
        except Exception:
            Cen.loc[len(Cen)] = [np.nan] * 7

    Cen["length"] = Cen["maxx"] - Cen["minx"]
    Cen["width"] = Cen["maxy"] - Cen["miny"]
    Cen["LpW"] = Cen["length"] / Cen["width"]
    return Cen


# FUNCTION 9: Compute stretch index (SI) relative to team centroid

def calculate_stretch_index(team_df, Cen, playersnum):
    """
    Stretch index is defined here as the mean absolute displacement of players from the team centroid in X and Y.
    """
    x_positions = [1 + i * 2 for i in range(playersnum)]
    y_positions = [2 + i * 2 for i in range(playersnum)]
    Cen["SI_x"] = sum(abs(team_df.iloc[:, x_pos] - Cen["cen_x"]) for x_pos in x_positions) / playersnum
    Cen["SI_y"] = sum(abs(team_df.iloc[:, y_pos] - Cen["cen_y"]) for y_pos in y_positions) / playersnum
    return Cen


## Interpersonal distances (single-team)

# FUNCTION 10: Compute all pairwise IPDs for one row and append summary metrics

def calculate_interpersonal_distances_for_row(row, playersnum):
    """
    Returns:
    - list of all pairwise distances (upper triangle) in player index order
    - plus max_dist, min_dist, mean_dist appended at the end
    """
    coords = []
    for i in range(playersnum):
        x = row.iloc[1 + i * 2]
        y = row.iloc[2 + i * 2]
        coords.append((x, y))

    dists = [math.dist(coords[i], coords[j]) for i, j in itertools.combinations(range(playersnum), 2)]
    if not dists:
        return [np.nan, np.nan, np.nan]

    return dists + [max(dists), min(dists), stat.mean(dists)]


# FUNCTION 11: Add IPD columns to Cen DataFrame (including rename to original position labels)

def add_ipd_columns(team_df, Cen, playersnum, orig_position_labels):
    """
    Adds:
    - All player pair distances (renamed from player1_2 to POSITIONA_POSITIONB where possible)
    - Summary columns: max_dist, min_dist, mean_dist
    """
    player_pairs = list(itertools.combinations(range(1, playersnum + 1), 2))
    raw_pair_cols = [f"player{p1}_{p2}" for p1, p2 in player_pairs]
    summary_cols = ["max_dist", "min_dist", "mean_dist"]
    all_cols = raw_pair_cols + summary_cols

    block = pd.DataFrame(np.nan, index=Cen.index, columns=all_cols, dtype="float64")
    Cen = pd.concat([Cen.reset_index(drop=True), block.reset_index(drop=True)], axis=1)

    for i in range(len(Cen)):
        ipd_values = calculate_interpersonal_distances_for_row(team_df.iloc[i], playersnum)
        Cen.loc[i, all_cols] = ipd_values

    # Rename pair columns back to original labels (if mapping is possible)
    rename_map = {}
    for col in raw_pair_cols:
        parts = col.replace("player", "").split("_")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            p1_idx = int(parts[0]) - 1
            p2_idx = int(parts[1]) - 1
            if 0 <= p1_idx < len(orig_position_labels) and 0 <= p2_idx < len(orig_position_labels):
                rename_map[col] = f"{orig_position_labels[p1_idx]}_{orig_position_labels[p2_idx]}"

    if rename_map:
        Cen.rename(columns=rename_map, inplace=True)

    return Cen


## Directionality / anisotropy metrics relative to the centroid

# FUNCTION 12: Compute anisotropy/orientation/direction relative to centroid for each frame

def compute_directionality_from_raw_and_centroid(raw_df: pd.DataFrame,
                                                 cen_x: np.ndarray,
                                                 cen_y: np.ndarray,
                                                 assume_attack_x_pos: bool = True):
    """
    Computes:
    - SI_anisotropy: 1 - (minor_axis / major_axis) from covariance of player displacements about centroid
    - SI_orientation_deg: orientation of major axis in [0, 180)
    - SI_direction_deg_360: direction of major axis in [0, 360), optionally flipped to align with attacking direction

    Method:
    - For each frame, subtract centroid from each player position to get displacement vectors
    - Compute covariance matrix of these vectors
    - Eigen-decompose covariance: major eigenvector gives principal spread direction
    """
    pairs = detect_player_xy_pairs(raw_df.columns)
    if not pairs:
        raise ValueError("No player (x,y) columns detected in raw file.")

    X = raw_df[[xc for xc, _ in pairs]].to_numpy(dtype=float)
    Y = raw_df[[yc for _, yc in pairs]].to_numpy(dtype=float)

    n = min(len(cen_x), len(raw_df))
    out_anis = np.full(len(cen_x), np.nan, dtype=float)
    out_or = np.full(len(cen_x), np.nan, dtype=float)
    out_dir = np.full(len(cen_x), np.nan, dtype=float)

    for i in range(n):
        xi = X[i, :]
        yi = Y[i, :]
        m = np.isfinite(xi) & np.isfinite(yi)
        if not np.any(m):
            continue
        dx = xi[m] - cen_x[i]
        dy = yi[m] - cen_y[i]
        if dx.size < 2:
            continue

        disp = np.column_stack([dx, dy])
        Z = disp - disp.mean(axis=0)
        C = (Z.T @ Z) / Z.shape[0]

        vals, vecs = np.linalg.eigh(C)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        major = float(np.sqrt(max(vals[0], 0.0)))
        minor = float(np.sqrt(max(vals[1], 0.0)))
        anis = 1.0 - (minor / major) if major > 0 and minor > 0 else np.nan

        vx, vy = float(vecs[0, 0]), float(vecs[1, 0])
        orient = (np.degrees(np.arctan2(vy, vx)) + 180.0) % 180.0
        direct = (np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0

        # Optional convention: ensure direction points "towards attack" in +X
        if assume_attack_x_pos and vx < 0:
            direct = (direct + 180.0) % 360.0

        out_anis[i] = anis
        out_or[i] = orient
        out_dir[i] = direct

    return out_anis, out_or, out_dir


# FUNCTION 13: Insert new columns after a specified column (for consistent column ordering)

def insert_after(df: pd.DataFrame, after_col: str, new_cols: dict):
    """
    Insert columns into df after after_col (if present). Otherwise append at end.
    """
    items = list(new_cols.items())
    if after_col in df.columns:
        idx = df.columns.get_loc(after_col) + 1
        for k, v in items[::-1]:
            df.insert(idx, k, v)
    else:
        for k, v in items:
            df[k] = v
    return df


## Unit (DEF/MID/ATT) inference and unit-level metrics

# FUNCTION 14: Infer unit label from a player/position string

def infer_unit_from_label(label: str) -> Optional[str]:
    """
    Map a player label to a unit group:
    - DEF / MID / ATT

    Returns None if label cannot be classified.
    """
    s = label.lower()
    if any(k in s for k in ["gk", "cb", "lb", "rb", "lcb", "rcb", "lwb", "rwb", "def"]):
        return "DEF"
    if any(k in s for k in ["cm", "dm", "am", "rm", "lm", "mid"]):
        return "MID"
    if any(k in s for k in ["st", "cf", "rf", "lf", "rw", "lw", "att", "fw", "for"]):
        return "ATT"
    return None


# FUNCTION 15: Build a mapping of unit -> player indices based on original labels

def build_unit_index_map(labels: list[str]) -> dict[str, list[int]]:
    """Return dict of indices for each unit based on infer_unit_from_label."""
    idx = {"DEF": [], "MID": [], "ATT": []}
    for i, lab in enumerate(labels):
        unit = infer_unit_from_label(lab)
        if unit in idx:
            idx[unit].append(i)
    return idx


# FUNCTION 16: Compute unit centroid coordinates for specified player indices

def unit_centroid_coords(team_df: pd.DataFrame, player_indices: list[int]) -> tuple[pd.Series, pd.Series]:
    """
    Compute unit centroid as mean position of players in player_indices for each frame.

    Returns:
    - cen_x series
    - cen_y series
    """
    if not player_indices:
        n = len(team_df)
        return (
            pd.Series([np.nan] * n, index=team_df.index),
            pd.Series([np.nan] * n, index=team_df.index),
        )
    x_cols = [1 + i * 2 for i in player_indices]
    y_cols = [2 + i * 2 for i in player_indices]
    cen_x = team_df.iloc[:, x_cols].mean(axis=1)
    cen_y = team_df.iloc[:, y_cols].mean(axis=1)
    return cen_x, cen_y


# FUNCTION 17: Add unit-level metrics to the single-team metrics DataFrame

def add_single_team_unit_metrics(Cen: pd.DataFrame,
                                 team_df: pd.DataFrame,
                                 orig_position_labels: list[str]) -> pd.DataFrame:
    """
    Adds:
    - Unit intra mean IPD for DEF/MID/ATT (within-unit spacing)
    - Unit centroid coordinates (unit_cen_x, unit_cen_y)
    - Distances between unit centroids (pairwise) and mean distance to other units
    """
    units = ["DEF", "MID", "ATT"]
    unit_idx_map = build_unit_index_map(orig_position_labels)
    n_rows = len(team_df)

    # Intra-unit mean IPD
    for unit in units:
        indices = unit_idx_map.get(unit, [])
        colname = f"{unit}_intra_mean_IPD"
        if len(indices) < 2:
            Cen[colname] = np.nan
            continue

        x_idx = [1 + i * 2 for i in indices]
        y_idx = [2 + i * 2 for i in indices]

        values = []
        for r in range(n_rows):
            coords = [(team_df.iat[r, x_col], team_df.iat[r, y_col]) for x_col, y_col in zip(x_idx, y_idx)]
            dists = [math.dist(coords[i], coords[j]) for i, j in itertools.combinations(range(len(coords)), 2)]
            values.append(stat.mean(dists) if dists else np.nan)

        Cen[colname] = values

    # Unit centroids
    unit_cen = {}
    for unit in units:
        indices = unit_idx_map.get(unit, [])
        cx, cy = unit_centroid_coords(team_df, indices)
        Cen[f"{unit}_cen_x"] = cx
        Cen[f"{unit}_cen_y"] = cy
        unit_cen[unit] = (cx.to_numpy(), cy.to_numpy())

    # Pairwise unit centroid distances
    for i, u1 in enumerate(units):
        for u2 in units[i + 1:]:
            x1, y1 = unit_cen[u1]
            x2, y2 = unit_cen[u2]
            Cen[f"{u1}_to_{u2}_centroid_dist"] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Mean distance from each unit to the other units
    for unit in units:
        dist_cols = []
        for other in units:
            if other == unit:
                continue
            col1 = f"{unit}_to_{other}_centroid_dist"
            col2 = f"{other}_to_{unit}_centroid_dist"
            if col1 in Cen.columns:
                dist_cols.append(col1)
            if col2 in Cen.columns:
                dist_cols.append(col2)
        Cen[f"{unit}_centroid_mean_dist_to_units"] = Cen[dist_cols].mean(axis=1) if dist_cols else np.nan

    return Cen


## Single-team run functions

# FUNCTION 18: Process one single-team tracking file and save metrics

def process_single_file(folder_path, csv_file):
    """
    Read a single tracking file, compute metrics, and save Processed_<file> next to the input file.
    """
    file_path = os.path.join(folder_path, csv_file)
    print(f"\nProcessing single-team file: {folder_path} / {csv_file}")

    # Drop Half if present to avoid unwanted removal during dropna in rename_players
    df_raw = pd.read_csv(file_path).drop(columns=["Half"], errors="ignore")
    df, orig_position_labels = rename_players(df_raw)

    playersnum = sum(1 for c in df.columns if str(c).endswith("_x"))
    print(f"Detected {playersnum} players")

    Cen = build_cen_dataframe(df)
    Cen = calculate_stretch_index(df, Cen, playersnum)
    Cen = add_ipd_columns(df, Cen, playersnum, orig_position_labels)

    # Directionality metrics derived from raw coordinate structure and centroid time-series
    cen_x = Cen["cen_x"].to_numpy(dtype=float)
    cen_y = Cen["cen_y"].to_numpy(dtype=float)
    anis, orient, direct = compute_directionality_from_raw_and_centroid(
        df_raw, cen_x, cen_y, assume_attack_x_pos=ASSUME_ATTACK_X_POS
    )
    Cen = insert_after(Cen, "SI_y", {
        "SI_anisotropy": anis,
        "SI_orientation_deg": orient,
        "SI_direction_deg_360": direct,
    })

    Cen = add_single_team_unit_metrics(Cen, df, orig_position_labels)
    Cen["Time"] = df["Time"].values

    out_name = f"Processed_{csv_file}"
    out_path = os.path.join(folder_path, out_name)
    Cen.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


# FUNCTION 19: Run the single-team pipeline over all matching files in an output folder

def run_single_team_pipeline(output_folder: str):
    """
    Find and process every file in output_folder that:
    - ends in 'filled.csv'
    - starts with Zeroed_ or Normalized_Zeroed_
    - is not already Processed_*
    """
    if not os.path.isdir(output_folder):
        print(f"Single-team Output folder not found, skipping: {output_folder}")
        return

    csv_files = [
        f for f in os.listdir(output_folder)
        if f.endswith("filled.csv")
        and (f.startswith("Zeroed_") or f.startswith("Normalized_Zeroed_"))
        and not f.startswith("Processed_")
    ]

    if not csv_files:
        print(f"No matching files in {output_folder}")
        return

    print(f"\n=== Single-team processing ===\n{output_folder}")
    for csv_file in csv_files:
        process_single_file(output_folder, csv_file)


# OPPOSITION (TWO-TEAM) PIPELINE #

## File matching and standardisation

# FUNCTION 20: Map "tail filename" -> full path for zeroed files in an output folder

def find_zeroed_files(folder: str) -> dict:
    """
    Return a mapping:
      tail_name (after Zeroed_ / Normalized_Zeroed_) -> full file path

    This simplifies pairing files across teams based on consistent suffix naming.
    """
    files = {}
    for f in os.listdir(folder):
        if not f.endswith("filled.csv"):
            continue
        if f.startswith("Normalized_Zeroed_"):
            tail = f[len("Normalized_Zeroed_"):]
        elif f.startswith("Zeroed_") or f.startswith("zeroed_"):
            tail = f[len("Zeroed_"):] if f.startswith("Zeroed_") else f[len("zeroed_"):]
        else:
            continue
        files[tail] = os.path.join(folder, f)
    return files


# FUNCTION 21: Convert an input file tail into its opposite-phase tail using PHASE_OPPOSITES

def make_opposite_tail(tail: str) -> Optional[str]:
    """
    Detect the phase token in a tail name and swap it to the opposite phase.

    Examples:
    - IP -> OOP
    - OOP -> IP
    - AT -> DT
    - DT -> AT
    """
    m = re.search(r'(?P<pre>^|_)(?P<tok>IP|OOP|AT|DT)(?=_)', tail)
    if not m:
        return None
    tok = m.group('tok')
    opp = PHASE_OPPOSITES[tok]
    return tail[:m.start('tok')] + opp + tail[m.end('tok'):]


# FUNCTION 22: Standardise a team tracking DataFrame for opposition merging

def standardize_team_df(df: pd.DataFrame):
    """
    Standardise a team DataFrame to:
    - Ensure Time is numeric
    - Ensure coordinate columns exist as *_x/*_y pairs
    - Rename players to player_1_x/player_1_y... for consistent indexing
    - Drop rows with NaNs to ensure pairwise computations are valid

    Returns:
    - df standardised with columns: Time, player_1_x, player_1_y, ...
    - orig_positions: list of original player stems (for naming outputs)
    """
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if str(c).lower() == "half"], errors="ignore")

    if "Time" not in df.columns:
        raise ValueError("Expected a 'Time' column.")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")

    cols = df.columns.tolist()
    x_cols = [c for c in cols if str(c).endswith("_x")]
    y_cols = [c for c in cols if str(c).endswith("_y")]
    if not x_cols or not y_cols:
        raise ValueError("No coordinate columns found ending in '_x'/'_y'.")

    pairs = []
    orig_positions = []
    for xc in sorted(x_cols, key=cols.index):
        stem = xc[:-2]
        yc = stem + "_y"
        if yc not in y_cols:
            raise ValueError(f"Missing '_y' for '{xc}'")
        pairs.append((xc, yc))
        orig_positions.append(stem.rstrip("_"))

    data_cols = ["Time"]
    new_cols = ["Time"]
    p = 1
    for xc, yc in pairs:
        data_cols += [xc, yc]
        new_cols += [f"player_{p}_x", f"player_{p}_y"]
        p += 1

    df = df[data_cols].dropna(subset=["Time"]).dropna().reset_index(drop=True)
    df.columns = new_cols
    return df, orig_positions


## Per-team metrics (opposition) - reused versions of single-team outputs

# FUNCTION 23: Compute per-team centroid/hull metrics for opposition pipeline

def per_team_metrics(team_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(columns=["cen_x", "cen_y", "surface_area", "minx", "miny", "maxx", "maxy"])
    for i in range(len(team_df)):
        try:
            out.loc[len(out)] = compute_metrics(team_df.loc[i])
        except Exception:
            out.loc[len(out)] = [np.nan] * 7
    out["length"] = out["maxx"] - out["minx"]
    out["width"] = out["maxy"] - out["miny"]
    out["LpW"] = out["length"] / out["width"]
    return out


# FUNCTION 24: Stretch index for a standardised team DataFrame

def stretch_index(team_df: pd.DataFrame, centers_df: pd.DataFrame, playersnum: int) -> pd.DataFrame:
    x_idx = [1 + i * 2 for i in range(playersnum)]
    y_idx = [2 + i * 2 for i in range(playersnum)]
    centers_df["SI_x"] = sum((team_df.iloc[:, xi] - centers_df["cen_x"]).abs() for xi in x_idx) / playersnum
    centers_df["SI_y"] = sum((team_df.iloc[:, yi] - centers_df["cen_y"]).abs() for yi in y_idx) / playersnum
    return centers_df


## Within-team and between-team distance metrics (opposition)

# FUNCTION 25: Compute intra-team distances for one row and return full list plus summary

def intra_team_ipds_row(row: pd.Series, playersnum: int):
    coords = [(row.iloc[1 + i * 2], row.iloc[2 + i * 2]) for i in range(playersnum)]
    dists = [math.dist(coords[i], coords[j]) for i, j in itertools.combinations(range(playersnum), 2)]
    return dists, (max(dists), min(dists), stat.mean(dists))


# FUNCTION 26: Add intra-team pairwise distances to an output holder DataFrame

def add_intra_team_ipds(team_df: pd.DataFrame, holder_df: pd.DataFrame,
                        playersnum: int, orig_labels: list, prefix: str) -> pd.DataFrame:
    """
    Adds:
    - All within-team pair distances (named using original labels)
    - Summary: max_intra, min_intra, mean_intra
    """
    pairs = list(itertools.combinations(range(playersnum), 2))
    pair_cols = [f"{prefix}{orig_labels[i]}__{orig_labels[j]}" for i, j in pairs]
    summary_cols = [f"{prefix}max_intra", f"{prefix}min_intra", f"{prefix}mean_intra"]

    block = pd.DataFrame(np.nan, index=holder_df.index, columns=pair_cols + summary_cols, dtype="float64")
    holder_df = pd.concat([holder_df.reset_index(drop=True), block.reset_index(drop=True)], axis=1)

    for i in range(len(holder_df)):
        dlist, summary = intra_team_ipds_row(team_df.loc[i], playersnum)
        holder_df.loc[i, pair_cols] = dlist
        holder_df.loc[i, summary_cols] = summary

    return holder_df


# FUNCTION 27: Compute all cross-team distances for one row plus minimum distances per player

def cross_team_distances_row(rowA: pd.Series, rowB: pd.Series, nA: int, nB: int):
    """
    Returns:
    - full: nA x nB distance matrix (as nested lists)
    - A_min: per-player minimum distance from A to any B player
    - B_min: per-player minimum distance from B to any A player
    """
    A = [(rowA.iloc[1 + i * 2], rowA.iloc[2 + i * 2]) for i in range(nA)]
    B = [(rowB.iloc[1 + j * 2], rowB.iloc[2 + j * 2]) for j in range(nB)]
    full = [[math.dist(a, b) for b in B] for a in A]
    A_min = [min(r) for r in full]
    B_min = [min(col) for col in zip(*full)]
    return full, A_min, B_min


# FUNCTION 28: Add all cross-team distances and summary metrics to an output holder DataFrame

def add_cross_team_distances(holder_df: pd.DataFrame,
                             teamA_df: pd.DataFrame, teamB_df: pd.DataFrame,
                             labelsA: list, labelsB: list) -> pd.DataFrame:
    """
    Adds:
    - A_i to B_j distances for all player pairs
    - Minimum distance to opponent for each player
    - Summary: AB_min_dist, AB_max_dist, AB_mean_dist over all cross-team pairs
    """
    nA = len(labelsA)
    nB = len(labelsB)

    all_cols = [f"{labelsA[i]}__to__{labelsB[j]}" for i in range(nA) for j in range(nB)]
    a_min_cols = [f"{lbl}_min_to_opp" for lbl in labelsA]
    b_min_cols = [f"{lbl}_min_to_opp" for lbl in labelsB]

    block_cols = all_cols + a_min_cols + b_min_cols
    block = pd.DataFrame(np.nan, index=holder_df.index, columns=block_cols, dtype="float64")
    holder_df = pd.concat([holder_df.reset_index(drop=True), block.reset_index(drop=True)], axis=1)

    for r in range(len(holder_df)):
        full, Amins, Bmins = cross_team_distances_row(teamA_df.loc[r], teamB_df.loc[r], nA, nB)
        holder_df.loc[r, all_cols] = [full[i][j] for i in range(nA) for j in range(nB)]
        holder_df.loc[r, a_min_cols] = Amins
        holder_df.loc[r, b_min_cols] = Bmins

    holder_df["AB_min_dist"] = holder_df[all_cols].min(axis=1)
    holder_df["AB_max_dist"] = holder_df[all_cols].max(axis=1)
    holder_df["AB_mean_dist"] = holder_df[all_cols].mean(axis=1)
    return holder_df


## Time alignment and reconstruction

# FUNCTION 29: Rebuild a standard team view from a merged (suffix-based) DataFrame

def rebuild_team_view(merged_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    After merging A and B with suffixes, rebuild a standardised team DataFrame for one team:
    - Time, player_1_x, player_1_y, ...
    """
    px = sorted(
        [c for c in merged_df.columns if c.endswith(f"_x_{suffix}")],
        key=lambda s: int(re.search(r'player_(\d+)_x', s).group(1))
    )
    py = sorted(
        [c for c in merged_df.columns if c.endswith(f"_y_{suffix}")],
        key=lambda s: int(re.search(r'player_(\d+)_y', s).group(1))
    )
    cols = ["Time"] + [c for pair in zip(px, py) for c in pair]
    out = merged_df[cols].copy()

    new_cols = ["Time"]
    pnum = 1
    for _ in zip(px, py):
        new_cols += [f"player_{pnum}_x", f"player_{pnum}_y"]
        pnum += 1
    out.columns = new_cols
    return out


# FUNCTION 30: Merge A and B time series using merge_asof with an optional tolerance

def asof_merge_with_tolerance(A_suff: pd.DataFrame, B_suff: pd.DataFrame, tol_sec: float) -> pd.DataFrame:
    """
    Time-align two team DataFrames using nearest-neighbour matching on Time.

    - If tol_sec <= 0, only exact matches are retained (tolerance disabled).
    - Otherwise, nearest matches within tol_sec are used.
    """
    A_suff["Time"] = pd.to_numeric(A_suff["Time"], errors="coerce")
    B_suff["Time"] = pd.to_numeric(B_suff["Time"], errors="coerce")
    A_suff = A_suff.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    B_suff = B_suff.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    tolerance = None if tol_sec is None or tol_sec <= 0 else tol_sec
    merged = pd.merge_asof(A_suff, B_suff, on="Time", direction="nearest", tolerance=tolerance)
    return merged.dropna().reset_index(drop=True)


## Unit metrics (opposition)

# FUNCTION 31: Add unit intra mean distances (within unit) for opposition pipeline

def add_unit_intra_mean_dists(holder_df: pd.DataFrame,
                              team_df: pd.DataFrame,
                              unit_idx_map: dict[str, list[int]],
                              units: list[str],
                              prefix: str) -> pd.DataFrame:
    """
    Add within-unit mean distance per frame for each unit (DEF/MID/ATT).
    """
    n_rows = len(team_df)
    for unit in units:
        indices = unit_idx_map.get(unit, [])
        colname = f"{prefix}{unit}_intra_mean_dist"
        if len(indices) < 2:
            holder_df[colname] = np.nan
            continue

        x_idx = [1 + i * 2 for i in indices]
        y_idx = [2 + i * 2 for i in indices]

        values = []
        for r in range(n_rows):
            coords = [(team_df.iat[r, x_col], team_df.iat[r, y_col]) for x_col, y_col in zip(x_idx, y_idx)]
            dists = [math.dist(coords[i], coords[j]) for i, j in itertools.combinations(range(len(coords)), 2)]
            values.append(stat.mean(dists) if dists else np.nan)

        holder_df[colname] = values
    return holder_df


## Opposition run function

# FUNCTION 32: Run opposition pipeline between Team A and Team B output folders

def run_opposition_pipeline(teamA_output: str, teamB_output: str):
    """
    For a TEAMA output folder and TEAMB output folder:
    - Detect all (Normalized_)Zeroed_*filled.csv files
    - Pair opposite phases (IP↔OOP, AT↔DT) using filename tails
    - Align by Time and compute:
        - Per-team centroid metrics + stretch index
        - Centroid-to-centroid distance
        - Within-team IPDs (A and B)
        - Cross-team distances and AB summary
        - Unit centroid distances (A unit to B unit)
        - Unit intra mean distances (A and B)
    - Append a final average row (mean of numeric columns) as a summary
    - Save as Processed_Paired_OppPhases_<desc>.csv
    """
    filesA = find_zeroed_files(teamA_output)
    filesB = find_zeroed_files(teamB_output)
    if not filesA or not filesB:
        print("Opposition: one or both folders have no (Normalized_)Zeroed_*filled.csv files.")
        return

    # Output location for opposition files (currently set to teamB_output)
    opp_out_folder = teamB_output
    os.makedirs(opp_out_folder, exist_ok=True)

    # Build all opposite-phase file pairs
    pairs = []
    for tailA, pathA in filesA.items():
        opp_tail_for_B = make_opposite_tail(tailA)
        if opp_tail_for_B and opp_tail_for_B in filesB:
            pathB = filesB[opp_tail_for_B]
            base_desc = re.sub(r'\.csv$', '', f"A_{tailA}__vs__B_{opp_tail_for_B}")
            pairs.append((pathA, pathB, base_desc))

    for tailB, pathB in filesB.items():
        opp_tail_for_A = make_opposite_tail(tailB)
        if opp_tail_for_A and opp_tail_for_A in filesA:
            pathA = filesA[opp_tail_for_A]
            base_desc = re.sub(r'\.csv$', '', f"A_{opp_tail_for_A}__vs__B_{tailB}")
            if (pathA, pathB, base_desc) not in pairs:
                pairs.append((pathA, pathB, base_desc))

    if not pairs:
        print("No opposite-phase pairs found (IP↔OOP / AT↔DT).")
        return

    for pathA, pathB, desc in pairs:
        print(f"\n=== Opposition: {os.path.basename(pathA)} vs {os.path.basename(pathB)} ===")

        A_raw = pd.read_csv(pathA)
        B_raw = pd.read_csv(pathB)
        A_df, A_labels = standardize_team_df(A_raw)
        B_df, B_labels = standardize_team_df(B_raw)

        # Suffix to avoid collisions during merge
        A_suff = A_df.add_suffix("_A")
        A_suff.rename(columns={"Time_A": "Time"}, inplace=True)
        B_suff = B_df.add_suffix("_B")
        B_suff.rename(columns={"Time_B": "Time"}, inplace=True)

        merged = asof_merge_with_tolerance(A_suff, B_suff, TIME_TOLERANCE_SEC)
        if merged.empty:
            print(f"Warning: no overlapping Time rows for pair '{desc}'. Skipping.")
            continue

        # Rebuild time-aligned team views
        A = rebuild_team_view(merged, "A")
        B = rebuild_team_view(merged, "B")

        nA = len([c for c in A.columns if c.endswith('_x')])
        nB = len([c for c in B.columns if c.endswith('_x')])

        # Per-team centroid metrics and stretch index
        A_cent = per_team_metrics(A)
        B_cent = per_team_metrics(B)
        A_cent = stretch_index(A, A_cent, nA)
        B_cent = stretch_index(B, B_cent, nB)

        A_cent["Time"] = A["Time"].values
        B_cent["Time"] = B["Time"].values

        out = A_cent.merge(B_cent, on="Time", suffixes=("_A", "_B"))

        # Centroid-to-centroid distance
        out["centroid_dx"] = out["cen_x_A"] - out["cen_x_B"]
        out["centroid_dy"] = out["cen_y_A"] - out["cen_y_B"]
        out["centroid_dist_AB"] = (out["centroid_dx"]**2 + out["centroid_dy"]**2) ** 0.5

        # Within-team IPDs
        out = add_intra_team_ipds(A, out, nA, A_labels, prefix="A_")
        out = add_intra_team_ipds(B, out, nB, B_labels, prefix="B_")

        # Cross-team distances
        out = add_cross_team_distances(out, A, B, A_labels, B_labels)

        # Unit centroid distances
        units = ["DEF", "MID", "ATT"]
        unit_idx_A = build_unit_index_map(A_labels)
        unit_idx_B = build_unit_index_map(B_labels)

        A_unit_cen = {}
        B_unit_cen = {}
        for unit in units:
            Ax, Ay = unit_centroid_coords(A, unit_idx_A[unit])
            Bx, By = unit_centroid_coords(B, unit_idx_B[unit])
            A_unit_cen[unit] = (Ax.to_numpy(), Ay.to_numpy())
            B_unit_cen[unit] = (Bx.to_numpy(), By.to_numpy())

        for uA in units:
            Ax, Ay = A_unit_cen[uA]
            for uB in units:
                Bx, By = B_unit_cen[uB]
                out[f"A_{uA}_to_B_{uB}_centroid_dist"] = np.sqrt((Ax - Bx) ** 2 + (Ay - By) ** 2)

        # Unit intra mean distances
        out = add_unit_intra_mean_dists(out, A, unit_idx_A, units, prefix="A_")
        out = add_unit_intra_mean_dists(out, B, unit_idx_B, units, prefix="B_")

        # Append an average row as a compact summary
        col_means = out.mean(numeric_only=True)
        avg_row = {col: col_means.get(col, np.nan) for col in out.columns}
        avg_row["Time"] = np.nan
        out.loc[len(out)] = avg_row

        out_name = f"Processed_Paired_OppPhases_{desc}.csv"
        out_path = os.path.join(opp_out_folder, out_name)
        out.to_csv(out_path, index=False)
        print(f"Saved opposition file: {out_path}")


# MAIN WORKFLOW #

def main():
    """
    Run:
    - Single-team pipeline on every discovered Output folder
    - Optional opposition pipeline on every discovered TEAMA/TEAMB sibling pair
    """
    single_outputs, paired_outputs = discover_outputs_and_pairs(MASTER_FOLDER)

    if not single_outputs:
        print(f"No Output/Normalized folders with processable files found under MASTER_FOLDER:\n  {MASTER_FOLDER}")
        return

    print(f"Found {len(single_outputs)} output folder(s) for single-team processing (Output + Normalized).")
    for out_dir in single_outputs:
        run_single_team_pipeline(out_dir)

    if RUN_OPPOSITION_METRICS:
        if not paired_outputs:
            print("\nNo TEAMA_/TEAMB_ sibling pairs found anywhere under MASTER_FOLDER. Skipping opposition.")
            return

        print(f"\nFound {len(paired_outputs)} paired output set(s) for opposition processing.")
        for a_out, b_out, label in paired_outputs:
            print("\n" + "=" * 70)
            print(f"Pair: {label}")
            print(f" A: {a_out}")
            print(f" B: {b_out}")
            run_opposition_pipeline(a_out, b_out)


if __name__ == "__main__":
    main()
