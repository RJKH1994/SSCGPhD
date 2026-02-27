#!/usr/bin/env python3
"""
Step08_Voronoi.py

Authors: Ryan Howard (@RJKH1994) & Matt Bridge (@MWBRIDGE)

- Purpose:
    Compute frame-level Voronoi cell areas for two teams (TEAMA vs TEAMB) for paired phases:
      - IP vs OOP
      - AT vs DT
    Optionally generate both directions (A IP vs B OOP AND A OOP vs B IP, etc).

- Key assumptions:
    - Tracking files are already zeroed (and optionally normalized).
    - Coordinates are in pitch space: x in [0..L], y in [0..W] (after any transforms/clipping).
    - Team A and Team B files can be time-aligned by exact "Time" inner join.

- Outputs:
    - Per-frame Voronoi areas for each player (A_* and B_* columns)
    - For IP/OOP pairs: adds a bottom "total_phase" average row
    - For AT/DT pairs: transition segmentation summary + bottom "total_transition" row
"""

import os, re
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, box
from mplsoccer.pitch import Pitch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe


## CONFIGURATION 

# Master folder that contains TEAMA_* and TEAMB_* folders
MASTER_FOLDER = '/Users/ryanhoward/Desktop/Data'

# Prefer Output/Normalized when it exists, else Output
OUTPUT_CANDIDATES = [
    os.path.join('Output', 'Normalized'),
    'Output',
]

# Output folders (separate for Normalized vs Zeroed) created UNDER TEAM A output folder
VORONOI_DIR_NORMALIZED_NAME = 'VoronoiNormalized'
VORONOI_DIR_ZEROED_NAME     = 'Voronoi'

# Generate both phase directions (A IP vs B OOP AND A OOP vs B IP)
GENERATE_BOTH_DIRECTIONS = True

# Optional Plots (debug/visual inspection)
SHOW_PLOTS     = False
PLOT_MAX_ROWS  = 1

# Units scaling (keep 1.0 unless converting units e.g., cm/mm)
SCALE_X        = 1.0
SCALE_Y        = 1.0

# Manual transform (normally keep False)
ROTATE_90      = False  # swap x<->y
FLIP_X         = False  # mirror L↔R after rotate
FLIP_Y         = False  # mirror B↔T after rotate

# Optional rezero: if your coords sometimes have substantial negatives
ALLOW_REZERO_IF_NEEDED = False
EPS          = 1e-6
NOISE_CLIP   = 0.05
MATERIAL_TOL = 0.20

# Auto-expand pitch dims slightly if data marginally exceeds bounds (use cautiously)
AUTO_EXPAND_DIMS = False
EXPAND_SLACK     = 0.10
MAX_EXPAND       = 0.50

# Complementary phase pairings (script will build IP↔OOP and AT↔DT)
PHASE_PAIRS = {'IP': 'OOP', 'AT': 'DT'}

# Phase regex:
#  - captures the phase token that appears after "Zeroed_" or "Normalized_Zeroed_"
#  - supports optional "_TEAMA/_TEAMB" suffix in name
PHASE_RE = re.compile(
    r'^(?:Normalized_)?Zeroed_([^_]+)_GPS_Data_new_10Hz(?:_TEAM[AB])?_(?:filled)\.csv$',
    re.IGNORECASE
)

# Pitch Markings
MARKING_MODE = "ratio"   # "absolute" or "ratio"

ABS = dict(
    centre_radius = 9.15,
    penalty_depth = 16.5,
    penalty_width = 40.32,
    goal_depth    = 5.5,
    goal_width    = 18.32,
    penalty_spot  = 11.0,
    corner_radius = 1.0,
    goal_frame_w  = 7.32,
    goal_frame_d  = 2.0
)
REF_LENGTH = 105.0
REF_WIDTH  = 68.0
LINE_KW = dict(linewidth=1.3, color=(1,1,1,0.85))

STRICT_CLIP_REPORT_TOP = 5

# Transition segmentation: identifies new segment when Time jump exceeds this
TIME_JUMP_THRESHOLD = 0.11  # seconds

## MATCH DISCOVERY (TEAMA_/TEAMB_ pairing + Output/Normalized support)

# FUNCTION 1: Choose the preferred Output folder for a team root (Output/Normalized first)
def pick_output_folder(team_folder: str) -> str | None:
    """
    Returns the first existing candidate output folder under a team folder.
    Preference order is OUTPUT_CANDIDATES.
    """
    for rel in OUTPUT_CANDIDATES:
        p = os.path.join(team_folder, rel)
        if os.path.isdir(p):
            return p
    return None

# FUNCTION 2: Discover TEAMA_*/TEAMB_* sibling pairs under MASTER_FOLDER
def discover_pairs(master_folder: str):
    """
    Finds TEAMA_* and TEAMB_* folders at the top level of master_folder,
    pairs them by shared suffix after TEAMA_/TEAMB_.

    Returns list of tuples:
      (match_key, teamA_output_dir, teamB_output_dir, teamA_root, teamB_root)
    """
    teamA = {}
    teamB = {}

    for name in sorted(os.listdir(master_folder)):
        full = os.path.join(master_folder, name)
        if not os.path.isdir(full):
            continue

        out_dir = pick_output_folder(full)
        if not out_dir:
            continue

        up = name.upper()
        if up.startswith("TEAMA_"):
            key = name[len("TEAMA_"):]
            teamA[key] = (out_dir, full)
        elif up.startswith("TEAMB_"):
            key = name[len("TEAMB_"):]
            teamB[key] = (out_dir, full)

    pairs = []
    for key, (a_out, a_root) in teamA.items():
        if key in teamB:
            b_out, b_root = teamB[key]
            pairs.append((key, a_out, b_out, a_root, b_root))

    return pairs

# FUNCTION 3: Locate Pitch_Lengths(_Normalized).csv for a match
def find_pitch_csv(match_root: str, teamA_output: str, teamB_output: str) -> str:
    """
    Prefer Pitch_Lengths_Normalized.csv, else Pitch_Lengths.csv.

    Search order:
      1) match_root (common path of TEAMA/TEAMB)
      2) teamA_output
      3) teamB_output
    """
    candidates = [
        os.path.join(match_root, 'Pitch_Lengths_Normalized.csv'),
        os.path.join(match_root, 'Pitch_Lengths.csv'),
        os.path.join(teamA_output, 'Pitch_Lengths_Normalized.csv'),
        os.path.join(teamA_output, 'Pitch_Lengths.csv'),
        os.path.join(teamB_output, 'Pitch_Lengths_Normalized.csv'),
        os.path.join(teamB_output, 'Pitch_Lengths.csv'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No pitch CSV found for this match. Looked for Pitch_Lengths_Normalized.csv or Pitch_Lengths.csv in:\n"
        f"  - {match_root}\n  - {teamA_output}\n  - {teamB_output}"
    )

## OUTPUT MODE HELPERS

# FUNCTION 4: Detect whether a file is Normalized_Zeroed_*
def is_normalized_file(path: str) -> bool:
    """Used to choose VoronoiNormalized vs Voronoi output folder."""
    return os.path.basename(path).lower().startswith("normalized_zeroed_")

## PLAYER COORDINATE HELPERS

# FUNCTION 5: Find *_x/*_y column pairs (supports _X/_Y variants)
def find_xy_pairs(df):
    """
    Returns a sorted list of (x_col, y_col) pairs.
    A pair is formed when a column ends with _x/_X and a matching _y/_Y exists.
    """
    cols = list(df.columns)
    pairs = []
    for c in cols:
        if re.search(r'(_x|_X)$', str(c)):
            base = re.sub(r'(_x|_X)$', '', str(c))
            y1, y2 = base + '_y', base + '_Y'
            if y1 in cols:
                pairs.append((c, y1))
            elif y2 in cols:
                pairs.append((c, y2))
    return sorted(pairs)

# FUNCTION 6: Flatten all x and y data into 2D arrays (n_players x n_rows)
def flatten_xy(df, pairs):
    """
    Builds:
      X: shape (n_players, n_rows)
      Y: shape (n_players, n_rows)
    """
    X = np.vstack([pd.to_numeric(df[x], errors='coerce').to_numpy() for x, _ in pairs]) if pairs else np.empty((0,0))
    Y = np.vstack([pd.to_numeric(df[y], errors='coerce').to_numpy() for _, y in pairs]) if pairs else np.empty((0,0))
    return X, Y

# FUNCTION 7: Apply uniform scaling to coordinate columns
def scale_inplace(df, pairs, sx, sy):
    """
    Multiplies x columns by sx and y columns by sy in-place.
    """
    if sx != 1.0:
        for x, _ in pairs:
            df[x] = pd.to_numeric(df[x], errors='coerce') * sx
    if sy != 1.0:
        for _, y in pairs:
            df[y] = pd.to_numeric(df[y], errors='coerce') * sy

# FUNCTION 8: Apply rotate/flip transforms and return updated (L,W)
def apply_rotation_flip_inplace(df, pairs, L, W):
    """
    Applies:
      - ROTATE_90: swap x<->y (also swaps L<->W)
      - FLIP_X: mirror across vertical axis (x -> L - x)
      - FLIP_Y: mirror across horizontal axis (y -> W - y)

    Returns updated pitch dimensions after rotate (if any).
    """
    curL, curW = L, W
    if ROTATE_90:
        for x, y in pairs:
            tmp = df[x].copy()
            df[x] = df[y]
            df[y] = tmp
        curL, curW = W, L
    if FLIP_X:
        for x, _ in pairs:
            df[x] = curL - pd.to_numeric(df[x], errors='coerce')
    if FLIP_Y:
        for _, y in pairs:
            df[y] = curW - pd.to_numeric(df[y], errors='coerce')
    return curL, curW

# FUNCTION 9: Print warnings when coords exceed pitch bounds by > tol
def warn_bounds(tag, X, Y, L, W, tol=MATERIAL_TOL):
    """
    Non-fatal diagnostic: reports if max(x) exceeds L or max(y) exceeds W beyond tolerance.
    """
    if X.size == 0 or Y.size == 0:
        return
    over_x = float(np.nanmax(X) - L)
    over_y = float(np.nanmax(Y) - W)
    if over_x > tol:
        print(f"[BOUNDS WARNING] {tag}: x exceeds L by {over_x:.2f} m (> {tol} m).")
    if over_y > tol:
        print(f"[BOUNDS WARNING] {tag}: y exceeds W by {over_y:.2f} m (> {tol} m).")

# FUNCTION 10: Gentle clip around bounds to eliminate tiny float noise / micro negatives
def micro_clip_inplace(df, pairs, L, W):
    """
    Clips coords to [-EPS..L+NOISE_CLIP] and [-EPS..W+NOISE_CLIP]
    prior to strict final clipping. Helps reduce edge-case geometry issues.
    """
    for x, y in pairs:
        df[x] = np.clip(pd.to_numeric(df[x], errors='coerce'), -EPS, L + NOISE_CLIP)
        df[y] = np.clip(pd.to_numeric(df[y], errors='coerce'), -EPS, W + NOISE_CLIP)

# FUNCTION 11: Optional rezero when substantial negatives exist
def rezero_if_needed_inplace(df, pairs):
    """
    If min(x) or min(y) is less than -MATERIAL_TOL, translate the entire dataset so that minima become ~0.
    Useful if your "zeroing" occasionally fails or outputs a shifted coordinate system.
    """
    X, Y = flatten_xy(df, pairs)
    if X.size == 0 or Y.size == 0:
        return
    if np.nanmin(X) < -MATERIAL_TOL or np.nanmin(Y) < -MATERIAL_TOL:
        ox = float(np.nanmin(X))
        oy = float(np.nanmin(Y))
        print(f"[REZERO] substantial negatives → translating by (ox,oy)=({ox:.3f},{oy:.3f})")
        for x, y in pairs:
            df[x] = pd.to_numeric(df[x], errors='coerce') - ox
            df[y] = pd.to_numeric(df[y], errors='coerce') - oy

# FUNCTION 12: Optional auto expansion of pitch dims (small expansion only)
def adjust_dims_to_data(L, W, X, Y, slack=EXPAND_SLACK, max_expand=MAX_EXPAND):
    """
    If data slightly exceeds pitch bounds, expand L/W up to max_expand metres.
    Use cautiously — expanding the pitch changes Voronoi geometry.
    """
    if X.size == 0 or Y.size == 0:
        return L, W
    L_needed = float(np.nanmax(X) + slack)
    W_needed = float(np.nanmax(Y) + slack)
    L_new, W_new = L, W
    if L_needed > L and (L_needed - L) <= max_expand:
        L_new = L_needed
        print(f"[AUTO-EXPAND] pitch_length {L:.2f}→{L_new:.2f}")
    if W_needed > W and (W_needed - W) <= max_expand:
        W_new = W_needed
        print(f"[AUTO-EXPAND] pitch_width  {W:.2f}→{W_new:.2f}")
    return L_new, W_new

# FUNCTION 13: Enforce pitch invariant length>width (some scripts assume this)
def enforce_length_gt_width(L, W):
    """
    Many downstream assumptions and pitch drawing behave better if L>W.
    If not, swap.
    """
    if L <= W:
        print(f"[INVARIANT] Enforcing length>width → swapping {L:.2f}↔{W:.2f}")
        return W, L
    return L, W

# FUNCTION 14: Strict final clipping to [0..L] x [0..W] with reporting
def final_clip_inplace(df, pairs, L, W):
    """
    Strictly clips all coords to within [0..L] and [0..W].
    Reports number of clipped values and examples (first few).
    """
    nxc = nyc = 0
    examples = []
    for x, y in pairs:
        xv = pd.to_numeric(df[x], errors='coerce').to_numpy()
        yv = pd.to_numeric(df[y], errors='coerce').to_numpy()
        mask_x = (xv < 0.0) | (xv > L)
        mask_y = (yv < 0.0) | (yv > W)
        if mask_x.any():
            nxc += int(mask_x.sum())
            idxs = np.where(mask_x)[0][:STRICT_CLIP_REPORT_TOP]
            for i in idxs:
                examples.append((x, i, float(xv[i])))
        if mask_y.any():
            nyc += int(mask_y.sum())
            idxs = np.where(mask_y)[0][:STRICT_CLIP_REPORT_TOP]
            for i in idxs:
                examples.append((y, i, float(yv[i])))
        df[x] = np.clip(xv, 0.0, L)
        df[y] = np.clip(yv, 0.0, W)
    if nxc or nyc:
        print(f"[FINAL-CLIP] Clipped {nxc} x-values and {nyc} y-values to [0..{L}]×[0..{W}].")
        for c, i, v in examples[:STRICT_CLIP_REPORT_TOP]:
            print(f"  • {c}[row {i}] was {v:.3f}")
    return nxc, nyc

## PITCH MARKINGS (OPTIONAL VISUALS)

# FUNCTION 15: Compute scaled marking dimensions (ratio or absolute)
def markings_dims(L, W, mode="absolute"):
    """
    Returns a dict of pitch marking dimensions.
    If mode == "ratio", scales FIFA-like values to your (L,W) using REF_LENGTH/REF_WIDTH.
    """
    if mode == "ratio":
        sL, sW = L / REF_LENGTH, W / REF_WIDTH
        return dict(
            R  = ABS['centre_radius'] * (sL + sW) / 2.0,
            PD = ABS['penalty_depth'] * sL,
            PW = ABS['penalty_width'] * sW,
            GD = ABS['goal_depth'] * sL,
            GW = ABS['goal_width'] * sW,
            PS = ABS['penalty_spot'] * sL,
            CR = ABS['corner_radius'] * ((sL + sW) / 2.0),
            GFW= ABS['goal_frame_w'] * sW,
            GFD= ABS['goal_frame_d'] * sL
        )
    return dict(
        R  = ABS['centre_radius'],
        PD = ABS['penalty_depth'],
        PW = ABS['penalty_width'],
        GD = ABS['goal_depth'],
        GW = ABS['goal_width'],
        PS = ABS['penalty_spot'],
        CR = ABS['corner_radius'],
        GFW= ABS['goal_frame_w'],
        GFD= ABS['goal_frame_d']
    )

# FUNCTION 16: Draw pitch markings on an axes (independent of mplsoccer lines)
def draw_scaled_markings(ax, L, W, mode="absolute"):
    """
    Draws touchlines, halfway, centre circle, boxes, and goal frames using scaled markings_dims().
    Used only for SHOW_PLOTS debug plots.
    """
    d = markings_dims(L, W, mode)

    ax.plot([0, L], [0, 0], **LINE_KW)
    ax.plot([0, L], [W, W], **LINE_KW)
    ax.plot([0, 0], [0, W], **LINE_KW)
    ax.plot([L, L], [0, W], **LINE_KW)
    ax.plot([L/2, L/2], [0, W], **LINE_KW)

    ax.add_patch(patches.Circle((L/2, W/2), d['R'], fill=False, **LINE_KW))

    y0 = (W - d['PW']) / 2
    ax.add_patch(patches.Rectangle((0, y0), d['PD'], d['PW'], fill=False, **LINE_KW))
    ax.add_patch(patches.Rectangle((L - d['PD'], y0), d['PD'], d['PW'], fill=False, **LINE_KW))

    y0g = (W - d['GW']) / 2
    ax.add_patch(patches.Rectangle((0, y0g), d['GD'], d['GW'], fill=False, **LINE_KW))
    ax.add_patch(patches.Rectangle((L - d['GD'], y0g), d['GD'], d['GW'], fill=False, **LINE_KW))

    ax.plot(d['PS'], W/2, marker='o', markersize=2.5, **LINE_KW)
    ax.plot(L - d['PS'], W/2, marker='o', markersize=2.5, **LINE_KW)

    R  = d['R']
    PD = d['PD']
    PS = d['PS']
    dx = PD - PS
    theta0 = np.degrees(np.arccos(dx / R)) if R > abs(dx) else 0

    ax.add_patch(patches.Arc((PS, W/2), 2 * R, 2 * R, theta1=-theta0, theta2=theta0, fill=False, **LINE_KW))
    ax.add_patch(patches.Arc((L - PS, W/2), 2 * R, 2 * R, theta1=180 - theta0, theta2=180 + theta0, fill=False, **LINE_KW))

    dCR = d['CR']
    corner_arcs = [
        ((0, 0),        0,   90),
        ((0, W),        270, 360),
        ((L, 0),        90,  180),
        ((L, W),        180, 270),
    ]
    for (cx, cy), t1, t2 in corner_arcs:
        ax.add_patch(patches.Arc((cx, cy), 2 * dCR, 2 * dCR, theta1=t1, theta2=t2, fill=False, **LINE_KW))

    y1 = W/2 - d['GFW'] / 2
    ax.add_patch(patches.Rectangle((-d['GFD'], y1), d['GFD'], d['GFW'], fill=False, **LINE_KW))
    ax.add_patch(patches.Rectangle((L, y1), d['GFD'], d['GFW'], fill=False, **LINE_KW))

    margin_x = d['GFD'] * 1.2
    margin_y = d['CR'] * 1.2
    ax.set_xlim(-margin_x, L + margin_x)
    ax.set_ylim(-margin_y, W + margin_y)
    ax.set_aspect('equal', adjustable='box')

## FILE DISCOVERY

# FUNCTION 17: Extract phase token from filename (based on PHASE_RE)
def _phase_from_filename(fname: str):
    m = PHASE_RE.match(fname)
    return m.group(1) if m else None

# FUNCTION 18: Map phase -> filepath for a given output folder
def list_phase_files(dir_path: str):
    """
    Returns dict:
      { "IP": ".../Zeroed_IP_...csv", "OOP": "...", ... }

    Note: If multiple files match same phase, last one encountered will overwrite.
    """
    out = {}
    if not os.path.isdir(dir_path):
        return out
    for f in os.listdir(dir_path):
        if not f.lower().endswith(".csv"):
            continue
        ph = _phase_from_filename(f)
        if ph:
            out[ph] = os.path.join(dir_path, f)
    return out

## VORONOI CORE

# FUNCTION 19: Process one phase pair (e.g., IP vs OOP) for a match
def process_pair(phase_a, phase_b, path_a, path_b, L_init, W_init,
                 output_dir_normalized, output_dir_zeroed,
                 match_tag="TEAMA_TEAMB",
                 is_transition_pair=False, is_ip_oop_pair=False):
    """
    Reads tracking files for TEAM A phase_a and TEAM B phase_b, then:
      1) Finds coordinate pairs
      2) Applies scaling and optional transforms
      3) Clips/rezero as configured
      4) Aligns rows by inner-join on Time (if available)
      5) For each aligned frame:
          - uses mplsoccer Pitch.voronoi with tids (TEAM A = 1, TEAM B = 0)
          - clips polygons to pitch bounds
          - computes polygon area per player
      6) Writes a CSV with A_* and B_* columns

    If is_transition_pair (AT/DT):
      - identifies segments using TIME_JUMP_THRESHOLD
      - writes a segment summary file and appends total_transition row

    If is_ip_oop_pair (IP/OOP):
      - appends total_phase row (mean over frames)
    """
    pair_name = f"{phase_a}_vs_{phase_b}"

    using_normalized = is_normalized_file(path_a) and is_normalized_file(path_b)
    out_dir = output_dir_normalized if using_normalized else output_dir_zeroed
    out_prefix = "VoronoiNormalized" if using_normalized else "Voronoi"

    print(f"\n=== Processing pair: {pair_name} [{match_tag}] (transition={is_transition_pair}, ip_oop={is_ip_oop_pair}, normalized={using_normalized})")

    # Load both teams
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    # Detect player columns
    pairs_a = find_xy_pairs(df_a)
    pairs_b = find_xy_pairs(df_b)
    if not pairs_a or not pairs_b:
        print("  Skipping: one of the teams has no *_x/*_y columns.")
        return

    # Scale coords (unit conversions)
    scale_inplace(df_a, pairs_a, SCALE_X, SCALE_Y)
    scale_inplace(df_b, pairs_b, SCALE_X, SCALE_Y)

    # Pitch dimensions (from CSV) with same scaling applied
    L = L_init * SCALE_X
    W = W_init * SCALE_Y
    L, W = enforce_length_gt_width(L, W)

    # Apply identical rotate/flip to both teams (IMPORTANT: must be the same transform)
    L, W = apply_rotation_flip_inplace(df_a, pairs_a, L, W)
    L, W = apply_rotation_flip_inplace(df_b, pairs_b, L, W)

    # Optional re-zero if substantial negatives exist
    if ALLOW_REZERO_IF_NEEDED:
        rezero_if_needed_inplace(df_a, pairs_a)
        rezero_if_needed_inplace(df_b, pairs_b)

    # Light clip then strict clip (reduces Voronoi edge-case failures)
    micro_clip_inplace(df_a, pairs_a, L, W)
    micro_clip_inplace(df_b, pairs_b, L, W)

    Xa, Ya = flatten_xy(df_a, pairs_a)
    Xb, Yb = flatten_xy(df_b, pairs_b)

    warn_bounds("TEAM A post-zero", Xa, Ya, L, W)
    warn_bounds("TEAM B post-zero", Xb, Yb, L, W)

    # Optional pitch expansion (rarely needed; changes geometry)
    if AUTO_EXPAND_DIMS:
        Xcat = np.hstack([Xa, Xb]) if Xa.size and Xb.size else (Xa if Xa.size else Xb)
        Ycat = np.hstack([Ya, Yb]) if Ya.size and Yb.size else (Ya if Ya.size else Yb)
        L, W = adjust_dims_to_data(L, W, Xcat, Ycat)

    L, W = enforce_length_gt_width(L, W)

    final_clip_inplace(df_a, pairs_a, L, W)
    final_clip_inplace(df_b, pairs_b, L, W)

    # Align by Time (exact intersection). If Time missing, falls back to row truncation below.
    if 'Time' in df_a.columns and 'Time' in df_b.columns:
        t = df_a[['Time']].merge(df_b[['Time']], on='Time', how='inner')['Time']
        df_a = df_a[df_a['Time'].isin(t)].sort_values('Time').reset_index(drop=True)
        df_b = df_b[df_b['Time'].isin(t)].sort_values('Time').reset_index(drop=True)

    nrows = min(len(df_a), len(df_b))

    # Pitch setup for Voronoi call
    pitch = Pitch(
        pitch_type='custom',
        pitch_color='grass',
        line_color=(1,1,1,0),
        linewidth=0.0,
        pitch_length=L,
        pitch_width=W
    )
    pitch_box = box(0.0, 0.0, L, W)

    # Player stem names (strip trailing _x/_X)
    names_a = [re.sub(r'(_x|_X)$', '', x) for x, _ in pairs_a]
    names_b = [re.sub(r'(_x|_X)$', '', x) for x, _ in pairs_b]
    result_rows = []

    # Frame loop: compute Voronoi areas per player
    for idx in range(nrows):
        row_a, row_b = df_a.iloc[idx], df_b.iloc[idx]
        xs_all, ys_all, tids_all, names_all = [], [], [], []

        # TEAM A points → tid=1
        for base, (xcol, ycol) in zip(names_a, pairs_a):
            x, y = row_a[xcol], row_a[ycol]
            if pd.notna(x) and pd.notna(y):
                xs_all.append(float(x)); ys_all.append(float(y))
                tids_all.append(1); names_all.append(('A', base))

        # TEAM B points → tid=0
        for base, (xcol, ycol) in zip(names_b, pairs_b):
            x, y = row_b[xcol], row_b[ycol]
            if pd.notna(x) and pd.notna(y):
                xs_all.append(float(x)); ys_all.append(float(y))
                tids_all.append(0); names_all.append(('B', base))

        # If too few valid points, output NaNs for all players
        if sum(np.isfinite(xs_all)) < 2:
            row_out = {'Row_Index': idx}
            if 'Time' in df_a.columns:
                row_out['Time'] = df_a.at[idx, 'Time']
            for nm in names_a:
                row_out[f'A_{nm}'] = np.nan
            for nm in names_b:
                row_out[f'B_{nm}'] = np.nan
            result_rows.append(row_out)
            continue

        # Compute Voronoi polygons (mplsoccer returns raw polygons per team)
        polys_a, polys_b = pitch.voronoi(xs_all, ys_all, tids_all)

        # Helper: clip each polygon to pitch rectangle and return exterior coords
        def _clip_list(polys):
            out = []
            for coords in polys:
                try:
                    geom = Polygon(coords)
                    inter = geom.intersection(pitch_box)
                    if inter.is_empty or not inter.is_valid:
                        out.append([])
                    else:
                        # If MultiPolygon, keep largest piece
                        if inter.geom_type == 'MultiPolygon':
                            biggest = max(inter.geoms, key=lambda g: g.area)
                            out.append(list(biggest.exterior.coords))
                        else:
                            out.append(list(inter.exterior.coords))
                except:
                    out.append([])
            return out

        clipped_a = _clip_list(polys_a)
        clipped_b = _clip_list(polys_b)

        # Accumulate areas into dicts keyed by player stem name
        areasA = {nm: np.nan for nm in names_a}
        areasB = {nm: np.nan for nm in names_b}
        ia = ib = 0

        # Iterate in the same order we built xs_all/ys_all.
        # We map sequentially into clipped_a/clipped_b lists for each team.
        for (team_tag, nm), tid in zip(names_all, tids_all):
            try:
                if tid == 1:
                    poly = Polygon(clipped_a[ia]); ia += 1
                    areasA[nm] = poly.area if poly.is_valid and not poly.is_empty else np.nan
                else:
                    poly = Polygon(clipped_b[ib]); ib += 1
                    areasB[nm] = poly.area if poly.is_valid and not poly.is_empty else np.nan
            except:
                # Defensive increment to keep indexing consistent if polygon build fails
                if tid == 1:
                    ia += 1
                else:
                    ib += 1

        # Row output
        row_out = {'Row_Index': idx}
        if 'Time' in df_a.columns:
            row_out['Time'] = df_a.at[idx, 'Time']

        for nm in names_a:
            row_out[f'A_{nm}'] = areasA[nm]
        for nm in names_b:
            row_out[f'B_{nm}'] = areasB[nm]

        result_rows.append(row_out)

        # Optional plot for first few frames
        if SHOW_PLOTS and idx < PLOT_MAX_ROWS:
            fig, axd = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                                  endnote_height=0.04, title_space=0, endnote_space=0)
            ax = axd['pitch']
            draw_scaled_markings(ax, L, W, MARKING_MODE)

            if any(clipped_a):
                pitch.polygon([c for c in clipped_a if c], ax=ax, ec='white', lw=1.2, alpha=0.35, zorder=2)
            if any(clipped_b):
                pitch.polygon([c for c in clipped_b if c], ax=ax, ec='white', lw=1.2, alpha=0.35, zorder=2)

            xsA = [x for x, t in zip(xs_all, tids_all) if t == 1]
            ysA = [y for y, t in zip(ys_all, tids_all) if t == 1]
            xsB = [x for x, t in zip(xs_all, tids_all) if t == 0]
            ysB = [y for y, t in zip(ys_all, tids_all) if t == 0]

            pitch.scatter(xsA, ysA, ax=ax, s=100, edgecolors='black', zorder=5)
            pitch.scatter(xsB, ysB, ax=ax, s=100, edgecolors='black', marker='s', zorder=5)

            outline_fx = [pe.withStroke(linewidth=2.4, foreground="black", alpha=0.65)]
            for (team_tag, nm), x, y in zip(names_all, xs_all, ys_all):
                ax.text(x, y, f"{team_tag}_{nm}", ha='center', va='center',
                        fontsize=10, weight='bold', color='white', zorder=6, path_effects=outline_fx)

            fig.suptitle(f"{pair_name} — Row {idx}", fontsize=16)
            plt.show()

    # Build output DataFrame
    os.makedirs(out_dir, exist_ok=True)
    out_df = pd.DataFrame(result_rows)

    # Column ordering: Row_Index, Time (if present), then all players
    base_cols = ['Row_Index']
    if 'Time' in out_df.columns:
        base_cols.append('Time')
    player_cols = [f'A_{nm}' for nm in names_a] + [f'B_{nm}' for nm in names_b]
    cols = base_cols + player_cols
    out_df = out_df.reindex(columns=cols)

    # ---------- Transition segmentation logic (AT/DT only) ----------
    if is_transition_pair and 'Time' in out_df.columns and len(out_df) > 0:
        # Segment boundaries from time jumps (typical when concatenating multiple transitions)
        times = pd.to_numeric(out_df['Time'], errors='coerce')
        dt = times.diff().abs().fillna(0)
        segment_id = (dt > TIME_JUMP_THRESHOLD).cumsum()

        tmp = out_df.copy()
        tmp['segment_id'] = segment_id

        # Segment-level aggregates: mean areas, first/last (for delta), pct change, pct per second
        seg_means = tmp.groupby('segment_id')[player_cols].mean()
        seg_first = tmp.groupby('segment_id')[player_cols].first()
        seg_last  = tmp.groupby('segment_id')[player_cols].last()
        seg_delta = seg_last - seg_first

        seg_first_nonzero = seg_first.replace(0, np.nan)
        seg_pct = (seg_delta / seg_first_nonzero) * 100.0

        t_start = tmp.groupby('segment_id')['Time'].min()
        t_end   = tmp.groupby('segment_id')['Time'].max()
        duration = t_end - t_start

        duration_nonzero = duration.replace(0, np.nan)
        seg_pct_per_s = seg_pct.div(duration_nonzero, axis=0)

        seg_summary = seg_means.reset_index()
        seg_summary['t_start'] = t_start.values
        seg_summary['t_end']   = t_end.values
        seg_summary['duration'] = duration.values

        # Append delta/pct/pps columns
        delta_cols, pct_cols, pps_cols = [], [], []
        for col in player_cols:
            dcol  = f"{col}_delta"
            pcol  = f"{col}_pct"
            ppscol = f"{col}_pct_per_s"
            seg_summary[dcol]   = seg_delta[col].values
            seg_summary[pcol]   = seg_pct[col].values
            seg_summary[ppscol] = seg_pct_per_s[col].values
            delta_cols.append(dcol); pct_cols.append(pcol); pps_cols.append(ppscol)

        seg_cols = ['segment_id', 't_start', 't_end', 'duration'] + player_cols + delta_cols + pct_cols + pps_cols
        seg_summary = seg_summary[seg_cols]

        # Overall across segments
        overall_means   = seg_means.mean(axis=0)
        overall_deltas  = seg_delta.mean(axis=0)
        overall_pcts    = seg_pct.mean(axis=0)
        overall_pps     = seg_pct_per_s.mean(axis=0)

        overall_row = {col: np.nan for col in seg_cols}
        overall_row['segment_id'] = 'total_transition'
        for col in player_cols:
            overall_row[col] = overall_means[col]
        for dcol in delta_cols:
            base_col = dcol.replace('_delta', '')
            overall_row[dcol] = overall_deltas[base_col]
        for pcol in pct_cols:
            base_col = pcol.replace('_pct', '')
            overall_row[pcol] = overall_pcts[base_col]
        for ppscol in pps_cols:
            base_col = ppscol.replace('_pct_per_s', '')
            overall_row[ppscol] = overall_pps[base_col]

        seg_summary = pd.concat([seg_summary, pd.DataFrame([overall_row])], ignore_index=True)

        # Save segment-level file
        seg_out_path = os.path.join(out_dir, f"{out_prefix}_{pair_name}_transitions_{match_tag}.csv")
        seg_summary.to_csv(seg_out_path, index=False)
        print(f"  Saved segment-level Voronoi: {seg_out_path}")

        # Also append a compact total_transition row to the main frame-level file
        avg_row_main = {col: np.nan for col in out_df.columns}
        avg_row_main['Row_Index'] = 'total_transition'
        if 'Time' in out_df.columns:
            avg_row_main['Time'] = np.nan
        for col in player_cols:
            avg_row_main[col] = overall_means[col]
        out_df = pd.concat([out_df, pd.DataFrame([avg_row_main])], ignore_index=True)

    else:
        if is_transition_pair:
            print("  Warning: no Time column or empty frame-level data; skipping segmentation.")

        # IP/OOP: add overall mean row (total_phase)
        if is_ip_oop_pair and len(out_df) > 0:
            overall_means = out_df[player_cols].mean()
            avg_row_main = {col: np.nan for col in out_df.columns}
            avg_row_main['Row_Index'] = 'total_phase'
            if 'Time' in out_df.columns:
                avg_row_main['Time'] = np.nan
            for col in player_cols:
                avg_row_main[col] = overall_means[col]
            out_df = pd.concat([out_df, pd.DataFrame([avg_row_main])], ignore_index=True)
            print("  Added bottom 'total_phase' average row for IP/OOP pair.")

    # Save main frame-level file
    out_path = os.path.join(out_dir, f"{out_prefix}_{pair_name}_{match_tag}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

## MAIN

if __name__ == "__main__":
    # Discover TEAMA/TEAMB match pairs
    pairs = discover_pairs(MASTER_FOLDER)
    if not pairs:
        raise FileNotFoundError(
            f"No TEAMA_*/TEAMB_* pairs found under MASTER_FOLDER:\n  {MASTER_FOLDER}\n"
            f"Also ensure each team folder contains Output or Output/Output/Normalized."
        )

    print(f"Found {len(pairs)} match pair(s) to process.")

    # Iterate matches
    for match_key, TEAM_A_OUTPUT_DIR, TEAM_B_OUTPUT_DIR, TEAM_A_ROOT, TEAM_B_ROOT in pairs:
        print("\n" + "=" * 90)
        print(f"MATCH: {match_key}")
        print(f"  TEAM A Output: {TEAM_A_OUTPUT_DIR}")
        print(f"  TEAM B Output: {TEAM_B_OUTPUT_DIR}")

        # Determine match-level pitch CSV location
        match_root = os.path.commonpath([TEAM_A_ROOT, TEAM_B_ROOT])
        PITCH_CSV_PATH = find_pitch_csv(match_root, TEAM_A_OUTPUT_DIR, TEAM_B_OUTPUT_DIR)
        print(f"  Pitch CSV: {PITCH_CSV_PATH}")

        # Read pitch dims from CSV (expects length in [0,0], width in [0,1])
        pl = pd.read_csv(PITCH_CSV_PATH)
        L_csv = float(pl.iloc[0, 0])
        W_csv = float(pl.iloc[0, 1])
        if L_csv <= W_csv:
            print(f"[INVARIANT] CSV has length<=width ({L_csv}≤{W_csv}) → swapping.")
            L_csv, W_csv = W_csv, L_csv

        # Output dirs under TEAM A output
        OUTPUT_DIR_NORMALIZED = os.path.join(TEAM_A_OUTPUT_DIR, VORONOI_DIR_NORMALIZED_NAME)
        OUTPUT_DIR_ZEROED     = os.path.join(TEAM_A_OUTPUT_DIR, VORONOI_DIR_ZEROED_NAME)

        # Phase->file mapping for each team
        teamA_by_phase = list_phase_files(TEAM_A_OUTPUT_DIR)
        teamB_by_phase = list_phase_files(TEAM_B_OUTPUT_DIR)

        # Process each defined phase pairing
        for phase_a, phase_b in PHASE_PAIRS.items():
            phases_set = {phase_a, phase_b}
            is_trans   = phases_set == {'AT', 'DT'}
            is_ip_oop  = phases_set == {'IP', 'OOP'}

            if GENERATE_BOTH_DIRECTIONS:
                # Direction 1: TEAM A phase_a vs TEAM B phase_b
                if phase_a in teamA_by_phase and phase_b in teamB_by_phase:
                    print(f"[PAIR] Using TEAM A {phase_a} vs TEAM B {phase_b}")
                    process_pair(
                        phase_a, phase_b,
                        teamA_by_phase[phase_a], teamB_by_phase[phase_b],
                        L_csv, W_csv,
                        OUTPUT_DIR_NORMALIZED, OUTPUT_DIR_ZEROED,
                        match_tag=f"{match_key}",
                        is_transition_pair=is_trans,
                        is_ip_oop_pair=is_ip_oop
                    )
                else:
                    print(f"[PAIR] Missing files for TEAM A {phase_a} vs TEAM B {phase_b}")

                # Direction 2: TEAM A phase_b vs TEAM B phase_a
                if phase_b in teamA_by_phase and phase_a in teamB_by_phase:
                    print(f"[PAIR] Using TEAM A {phase_b} vs TEAM B {phase_a}")
                    process_pair(
                        phase_b, phase_a,
                        teamA_by_phase[phase_b], teamB_by_phase[phase_a],
                        L_csv, W_csv,
                        OUTPUT_DIR_NORMALIZED, OUTPUT_DIR_ZEROED,
                        match_tag=f"{match_key}",
                        is_transition_pair=is_trans,
                        is_ip_oop_pair=is_ip_oop
                    )
                else:
                    print(f"[PAIR] Missing files for TEAM A {phase_b} vs TEAM B {phase_a}")

            else:
                # Single-direction fallback:
                # Try A phase_a vs B phase_b else A phase_b vs B phase_a
                if phase_a in teamA_by_phase and phase_b in teamB_by_phase:
                    process_pair(
                        phase_a, phase_b,
                        teamA_by_phase[phase_a], teamB_by_phase[phase_b],
                        L_csv, W_csv,
                        OUTPUT_DIR_NORMALIZED, OUTPUT_DIR_ZEROED,
                        match_tag=f"{match_key}",
                        is_transition_pair=is_trans,
                        is_ip_oop_pair=is_ip_oop
                    )
                elif phase_b in teamA_by_phase and phase_a in teamB_by_phase:
                    process_pair(
                        phase_b, phase_a,
                        teamA_by_phase[phase_b], teamB_by_phase[phase_a],
                        L_csv, W_csv,
                        OUTPUT_DIR_NORMALIZED, OUTPUT_DIR_ZEROED,
                        match_tag=f"{match_key}",
                        is_transition_pair=is_trans,
                        is_ip_oop_pair=is_ip_oop
                    )
                else:
                    print(f"[PAIR] Skipping {phase_a}↔{phase_b}: missing files for both directions.")
