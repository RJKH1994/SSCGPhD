#!/usr/bin/env python3
"""

Step09_Shannon_Entropy.py

Authors: Ryan Howard (@RJKH1994) & Matt Bridge (@MWBRIDGE)

Purpose:
    Compute Shannon entropy of player movement distributions (2D histogram over pitch space) from team tracking CSVs.

Supports:
    - Both "Normalized_Zeroed_*_filled.csv" and "Zeroed_*_filled.csv"
    - Output folders in either:
        - Output/Normalized (preferred)
        - Output
    - Pitch dimension files found by searching upward from each CSV:
        - Pitch_Lengths_Normalized.csv OR Pitch_Lengths.csv

Behaviour by phase:
    - AT / DT (transition files):
        - Segment-level entropy per transition (segment_id inferred from Time jumps)
        - Writes: <stem>__entropy_transitions.csv (includes 'total_transition' aggregate)
        - Also writes whole-file summary: <stem>__entropy_summary.csv
    - IP / OOP (non-transition files):
        - Whole-file summary only: <stem>__entropy_summary.csv

Entropy levels computed:
    - players: per-player entropy
    - units_total: entropy from concatenated positions for each unit (Defence/Midfield/Attack)
    - units_avg: mean of per-player entropies within each unit
    - team: entropy of the team centroid trajectory

Notes:
    - Pitch_Length_Normalized.csv must be present in the normalized folder. This file is available for a 105m x 68m pitch on GitHub.
    - Entropy is computed from a 2D histogram with BIN_SIZE spacing (in metres by default).
    - Normalised entropy divides by max entropy for the bin grid (log(bins_x*bins_y)).
    - CLIP_TO_PITCH controls whether (x,y) are clipped to pitch bounds before histogramming.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import re
from scipy.stats import entropy
from math import ceil

## CONFIGURATION

MASTER_FOLDER = r'/Users/ryanhoward/Documents/University_of_Birmingham/Research Studies/Study 1/Study 1 Data' # Change this to the master folder containing individual game folders

LOG_BASE = 2
CLIP_TO_PITCH = False

# Find both Normalized_Zeroed_ and Zeroed_
DATA_PATTERNS = ("Normalized_Zeroed_", "Zeroed_")
DATA_SUFFIX = "_filled.csv"

# Pitch file can be either name
PITCH_FILES = ("Pitch_Lengths_Normalized.csv", "Pitch_Lengths.csv")

# Prefer Output/Normalized if present, else Output
OUTPUT_CANDIDATES = (
    ("Output", "Normalized"),
    ("Output",),
)

# 2D histogram bin size in metres (1.0 m default)
BIN_SIZE = 1.0

# Expected pitch columns in Pitch_Lengths*.csv
PITCH_LEN_CAND = ["pitch_length"]
PITCH_WID_CAND = ["pitch_width"]

PHASE_TAGS = ("AT", "DT", "IP", "OOP")

# For transitions (AT/DT): new segment when Time jump exceeds this
TIME_JUMP_THRESHOLD = 0.11  # seconds

## UNIT INFERENCE (POSITION -> Defence/Midfield/Attack)

def infer_unit(name: str) -> str:
    """
    Assign a player label to a unit group based on common substrings.
    Returns one of: Defence, Midfield, Attack, Unknown
    """
    n = name.upper()
    if any(tag in n for tag in ["CB", "CB2", "LB", "RB", "LWB", "RWB", "DEF"]):
        return "Defence"
    if any(tag in n for tag in ["DM", "DM2", "CM", "CM2", "AM", "AM2", "MID"]):
        return "Midfield"
    if any(tag in n for tag in ["CF", "CF2", "ST", "FW", "LW", "RW", "FOR"]):
        return "Attack"
    return "Unknown"

## OUTPUT FOLDER DISCOVERY (Output/Normalized OR Output)

def find_output_dirs_in_match(match_root: Path) -> list[Path]:
    """
    Returns output directories to scan within a match root.
    Preference is built into OUTPUT_CANDIDATES (Output/Normalized first).
    """
    out_dirs = []
    for rel in OUTPUT_CANDIDATES:
        p = match_root.joinpath(*rel)
        if p.is_dir():
            out_dirs.append(p)
    return out_dirs


def discover_match_roots(master: Path) -> list[Path]:
    """
    Treat each immediate subfolder of MASTER_FOLDER as a match root.
    """
    if not master.is_dir():
        raise FileNotFoundError(f"MASTER_FOLDER does not exist: {master}")

    roots = [p for p in master.iterdir() if p.is_dir()]
    return sorted(roots)

## FIND PITCH FILE (nearest up the folder tree)

def nearest_pitch_file(start_path: Path) -> Path | None:
    """
    Walk upward from start_path looking for Pitch_Lengths_Normalized.csv or Pitch_Lengths.csv.
    Returns the first match found, else None.
    """
    cur = start_path
    while True:
        for fname in PITCH_FILES:
            p = cur / fname
            if p.exists():
                return p
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


## READ PITCH DIMENSIONS

def read_pitch_dims(pitch_csv: Path) -> tuple[float, float]:
    """
    Read pitch_length and pitch_width from pitch_csv.

    Expects columns named in PITCH_LEN_CAND and PITCH_WID_CAND.
    Returns (L, W).
    """
    df = pd.read_csv(pitch_csv)

    len_col = next((c for c in PITCH_LEN_CAND if c in df.columns), None)
    wid_col = next((c for c in PITCH_WID_CAND if c in df.columns), None)
    if not (len_col and wid_col):
        raise ValueError(
            f"Could not find pitch_length/pitch_width in {pitch_csv}. "
            f"Looked for length in {PITCH_LEN_CAND} and width in {PITCH_WID_CAND}."
        )

    row = df[[len_col, wid_col]].dropna().head(1)
    if row.empty:
        raise ValueError(f"No valid pitch dimensions in {pitch_csv}")

    L, W = float(row.iloc[0, 0]), float(row.iloc[0, 1])
    if L <= 0 or W <= 0:
        raise ValueError(f"Non-positive pitch dimensions in {pitch_csv}: L={L}, W={W}")
    return L, W

## EXTRACT PLAYER x/y PAIRS FROM WIDE TRACKING DATA

def extract_players_wide(orig_columns: list[str]) -> dict[str, dict[str, str]]:
    """
    Parse wide-format tracking headers to find player coordinate pairs.

    Returns:
      {
        "LB": {"x": "LB_x", "y": "LB_y"},
        "CB": {"x": "CB_x", "y": "CB_y"},
        ...
      }

    Notes:
    - Normalises multiple underscores in column names.
    - Strips trailing phase tokens from base names (e.g., "_AT", "_DT", "_IP", "_OOP").
    - Skips obvious time-like columns.
    """
    norm_cols = [re.sub(r"_+", "_", c.strip()) for c in orig_columns]
    norm_to_orig = {n: o for n, o in zip(norm_cols, orig_columns)}

    def base_from(norm_col: str, axis: str) -> str:
        """
        Given a normalised column and axis ("x" or "y"), return a base player name.
        Removes trailing phase suffix like "_AT" and also handles "_AT_n" (or "_n_AT").
        """
        if not norm_col.lower().endswith(f"_{axis}"):
            return ""
        base = norm_col[:-(len(axis) + 1)].rstrip("_")

        tokens = [t for t in base.split("_") if t != ""]
        if not tokens:
            return ""

        phase_set = set(PHASE_TAGS)
        while tokens and (tokens[-1].upper() in phase_set or tokens[-1].lower() == "n"):
            tokens.pop()

        return "_".join(tokens).rstrip("_")

    pairs: dict[str, dict[str, str]] = {}
    for nc in norm_cols:
        low = nc.lower()
        if low in ("time", "t", "timestamp", "frame"):
            continue
        if low.endswith("_x"):
            key = base_from(nc, "x")
            if key:
                pairs.setdefault(key, {})["x"] = norm_to_orig[nc]
        elif low.endswith("_y"):
            key = base_from(nc, "y")
            if key:
                pairs.setdefault(key, {})["y"] = norm_to_orig[nc]

    # Keep only complete (x,y) pairs
    return {name: xy for name, xy in pairs.items() if "x" in xy and "y" in xy}

## SHANNON ENTROPY (2D HISTOGRAM)

def shannon_entropy_xy(
    x: np.ndarray,
    y: np.ndarray,
    L: float,
    W: float,
    bins_x: int,
    bins_y: int,
    base: float = LOG_BASE
) -> tuple[float, float]:
    """
    Compute Shannon entropy of 2D occupancy distribution for (x,y).

    Returns:
      (H_bits, H_norm)

    Where:
      - H_bits is Shannon entropy in the chosen LOG_BASE
      - H_norm is H_bits divided by max possible entropy for bins_x*bins_y
    """
    if x.size == 0:
        return (np.nan, np.nan)

    if CLIP_TO_PITCH:
        x = np.clip(x, 0, L)
        y = np.clip(y, 0, W)

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return (np.nan, np.nan)

    H, _, _ = np.histogram2d(
        x, y,
        bins=[bins_x, bins_y],
        range=[[0, L], [0, W]],
        density=False
    )

    counts = H.ravel().astype(float)
    total = counts.sum()
    if total == 0:
        return (np.nan, np.nan)

    p = counts[counts > 0] / total
    H_bits = float(entropy(p, base=base))

    maxH = np.log(bins_x * bins_y) / np.log(base) if (bins_x > 0 and bins_y > 0) else np.nan
    H_norm = H_bits / maxH if (maxH and np.isfinite(maxH) and maxH > 0) else np.nan
    return (H_bits, H_norm)

## TEAM CENTROID (from wide player columns)

def team_centroid_from_wide(df: pd.DataFrame, players: dict[str, dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute team centroid time series as mean of all player coordinates per frame.

    Returns:
      (cx, cy) arrays of length n_frames.
    """
    xs = [df[d["x"]].to_numpy(dtype=float) for d in players.values()]
    ys = [df[d["y"]].to_numpy(dtype=float) for d in players.values()]
    X = np.vstack(xs)
    Y = np.vstack(ys)
    cx = np.nanmean(X, axis=0)
    cy = np.nanmean(Y, axis=0)
    return cx, cy

## FILE PROCESSING

def is_transition_file(csv_path: Path) -> bool:
    """
    Detect transition phase (AT/DT) from filename tokens.
    """
    u = csv_path.name.upper()
    return ("_AT_" in u) or ("_DT_" in u)


def process_file(csv_path: Path):
    """
    Process one tracking CSV:
      - load pitch dims (nearest pitch file up the tree)
      - detect players
      - ensure Time exists and is numeric
      - if transition file: segment-level entropy + total_transition row
      - always: whole-file entropy summary
    """
    print(f"\nProcessing: {csv_path}")

    is_transition_phase = is_transition_file(csv_path)

    pitch_path = nearest_pitch_file(csv_path.parent)
    if pitch_path is None:
        raise FileNotFoundError(f"No pitch CSV found for {csv_path} (looked for {PITCH_FILES})")

    L, W = read_pitch_dims(pitch_path)

    bins_x = max(1, int(ceil(L / BIN_SIZE)))
    bins_y = max(1, int(ceil(W / BIN_SIZE)))
    print(f"  Pitch: L={L:.2f} m, W={W:.2f} m | bins_x={bins_x}, bins_y={bins_y}")

    df = pd.read_csv(csv_path)
    players = extract_players_wide(list(df.columns))
    if not players:
        raise ValueError("No '*_x'/'*_y' pairs found in header.")
    print(f"  Players detected: {len(players)}")

    if "Time" not in df.columns:
        raise ValueError("Expected a 'Time' column in the data.")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    # 1) SEGMENT-LEVEL ENTROPY (AT/DT only)
    if is_transition_phase:
        # Segment id increments when there is a large discontinuity in Time
        df["dt"] = df["Time"].diff().fillna(0)
        df["segment_id"] = (df["dt"].abs() > TIME_JUMP_THRESHOLD).cumsum().astype(int)

        seg_rows: list[dict] = []

        def add_seg_row(seg_id: int, t_start: float, t_end: float,
                        level: str, id_: str, H_bits: float, H_norm: float):
            """
            Append two rows for a segment metric:
              - entropy_bits
              - entropy_norm
            """
            seg_rows.append({
                "segment_id": seg_id,
                "t_start": t_start,
                "t_end": t_end,
                "level": level,
                "id": id_,
                "metric": "entropy_bits",
                "value": H_bits,
                "bins_x": bins_x,
                "bins_y": bins_y,
                "max_entropy_bits": float(np.log2(bins_x * bins_y))
            })
            seg_rows.append({
                "segment_id": seg_id,
                "t_start": t_start,
                "t_end": t_end,
                "level": level,
                "id": id_,
                "metric": "entropy_norm",
                "value": H_norm,
                "bins_x": bins_x,
                "bins_y": bins_y,
                "max_entropy_bits": float(np.log2(bins_x * bins_y))
            })

        # Compute per-segment entropy
        for seg_id, seg_df in df.groupby("segment_id"):
            t_start = float(seg_df["Time"].iloc[0])
            t_end = float(seg_df["Time"].iloc[-1])

            # Store per-player entropies for computing within-unit averages
            unit_memberships: dict[str, list[tuple[float, float]]] = {}

            # Players: per-player entropy
            for name, xy in players.items():
                x = seg_df[xy["x"]].to_numpy(dtype=float)
                y = seg_df[xy["y"]].to_numpy(dtype=float)
                H_bits, H_norm = shannon_entropy_xy(x, y, L, W, bins_x=bins_x, bins_y=bins_y)
                add_seg_row(seg_id, t_start, t_end, "players", name, H_bits, H_norm)

                unit = infer_unit(name)
                unit_memberships.setdefault(unit, []).append((H_bits, H_norm))

            # Units (total): concatenate positions from all players in a unit
            for unit in unit_memberships.keys():
                xs = np.concatenate([seg_df[players[n]["x"]].to_numpy(dtype=float) for n in players if infer_unit(n) == unit])
                ys = np.concatenate([seg_df[players[n]["y"]].to_numpy(dtype=float) for n in players if infer_unit(n) == unit])
                H_bits, H_norm = shannon_entropy_xy(xs, ys, L, W, bins_x=bins_x, bins_y=bins_y)
                add_seg_row(seg_id, t_start, t_end, "units_total", unit, H_bits, H_norm)

            # Units (avg): mean of per-player entropies in that unit
            for unit, vals in unit_memberships.items():
                if not vals:
                    continue
                H_bits_mean = np.nanmean([v[0] for v in vals])
                H_norm_mean = np.nanmean([v[1] for v in vals])
                add_seg_row(seg_id, t_start, t_end, "units_avg", unit, H_bits_mean, H_norm_mean)

            # Team: entropy of team centroid trajectory in this segment
            cx_seg, cy_seg = team_centroid_from_wide(seg_df, players)
            H_bits_c, H_norm_c = shannon_entropy_xy(cx_seg, cy_seg, L, W, bins_x=bins_x, bins_y=bins_y)
            add_seg_row(seg_id, t_start, t_end, "team", "centroid", H_bits_c, H_norm_c)

        seg_df_out = pd.DataFrame(seg_rows)
        if not seg_df_out.empty:
            # Allow segment_id to contain integers and "total_transition"
            seg_df_out["segment_id"] = seg_df_out["segment_id"].astype(object)

            # Compute "total_transition" by averaging segment values per (level,id,metric)
            avg = (
                seg_df_out
                .groupby(["level", "id", "metric"], as_index=False)
                .agg({
                    "value": "mean",
                    "bins_x": "first",
                    "bins_y": "first",
                    "max_entropy_bits": "first"
                })
            )
            avg["segment_id"] = "total_transition"
            avg["t_start"] = np.nan
            avg["t_end"] = np.nan

            # Column ordering
            cols = [
                "segment_id", "t_start", "t_end",
                "level", "id", "metric",
                "value", "bins_x", "bins_y", "max_entropy_bits"
            ]
            seg_df_out = seg_df_out[cols]
            avg = avg[cols]
            seg_df_out = pd.concat([seg_df_out, avg], ignore_index=True)

            # Sorting:
            #  - segments in numeric order, then total_transition at bottom
            #  - level order: players, units_total, units_avg, team
            def seg_order(val):
                return 10**9 if val == "total_transition" else int(val)

            sort_order = pd.CategoricalDtype(["players", "units_total", "units_avg", "team"], ordered=True)
            seg_df_out["level"] = seg_df_out["level"].astype(sort_order)
            seg_df_out["seg_sort"] = seg_df_out["segment_id"].apply(seg_order)

            seg_df_out = (
                seg_df_out
                .sort_values(["seg_sort", "level", "id", "metric"])
                .drop(columns=["seg_sort"])
                .reset_index(drop=True)
            )

            out_seg_csv = csv_path.with_name(csv_path.stem + "__entropy_transitions.csv")
            seg_df_out.to_csv(out_seg_csv, index=False)
            print(f"  Saved segment-level entropy (AT/DT only): {out_seg_csv.name}")
        else:
            print("  No segment-level entropy rows produced (check Time data).")
    else:
        print("  File is IP/OOP or non-transition; skipping segment-level entropy.")

    # 2) WHOLE-FILE SUMMARY (all phases)
    rows: list[dict] = []

    def add_row(level: str, id_: str, H_bits: float, H_norm: float):
        """
        Append two rows:
          - entropy_bits
          - entropy_norm
        """
        rows.append({
            "level": level,
            "id": id_,
            "metric": "entropy_bits",
            "value": H_bits,
            "bins_x": bins_x,
            "bins_y": bins_y,
            "max_entropy_bits": float(np.log2(bins_x * bins_y))
        })
        rows.append({
            "level": level,
            "id": id_,
            "metric": "entropy_norm",
            "value": H_norm,
            "bins_x": bins_x,
            "bins_y": bins_y,
            "max_entropy_bits": float(np.log2(bins_x * bins_y))
        })

    # Collect per-player entropies for unit averaging
    unit_memberships_full: dict[str, list[tuple[float, float]]] = {}

    # Players: per-player entropy across entire file
    for name, xy in players.items():
        x = df[xy["x"]].to_numpy(dtype=float)
        y = df[xy["y"]].to_numpy(dtype=float)
        H_bits, H_norm = shannon_entropy_xy(x, y, L, W, bins_x=bins_x, bins_y=bins_y)
        add_row("players", name, H_bits, H_norm)

        unit = infer_unit(name)
        unit_memberships_full.setdefault(unit, []).append((H_bits, H_norm))

    # Units (total): concatenate all unit player positions into one series
    for unit in unit_memberships_full.keys():
        xs = np.concatenate([df[players[n]["x"]].to_numpy(dtype=float) for n in players if infer_unit(n) == unit])
        ys = np.concatenate([df[players[n]["y"]].to_numpy(dtype=float) for n in players if infer_unit(n) == unit])
        H_bits, H_norm = shannon_entropy_xy(xs, ys, L, W, bins_x=bins_x, bins_y=bins_y)
        add_row("units_total", unit, H_bits, H_norm)

    # Units (avg): mean entropy of players in the unit
    for unit, vals in unit_memberships_full.items():
        if not vals:
            continue
        H_bits_mean = np.nanmean([v[0] for v in vals])
        H_norm_mean = np.nanmean([v[1] for v in vals])
        add_row("units_avg", unit, H_bits_mean, H_norm_mean)

    # Team: centroid entropy across entire file
    cx, cy = team_centroid_from_wide(df, players)
    H_bits_c, H_norm_c = shannon_entropy_xy(cx, cy, L, W, bins_x=bins_x, bins_y=bins_y)
    add_row("team", "centroid", H_bits_c, H_norm_c)

    # Sort and write summary
    out_df = pd.DataFrame(rows)
    sort_order_full = pd.CategoricalDtype(["players", "units_total", "units_avg", "team"], ordered=True)
    out_df["level"] = out_df["level"].astype(sort_order_full)
    out_df = out_df.sort_values(["level", "id", "metric"]).reset_index(drop=True)

    out_csv = csv_path.with_name(csv_path.stem + "__entropy_summary.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"  Saved whole-file summary: {out_csv.name}")


## MAIN

def main():
    """
    Iterate match roots under MASTER_FOLDER and process all matching tracking files.
    """
    master = Path(MASTER_FOLDER)
    match_roots = discover_match_roots(master)
    if not match_roots:
        print("No subfolders found in MASTER_FOLDER.")
        return

    total_files = 0
    processed = 0

    for match_root in match_roots:
        out_dirs = find_output_dirs_in_match(match_root)
        if not out_dirs:
            continue

        print("\n" + "=" * 80)
        print(f"MATCH ROOT: {match_root}")
        for od in out_dirs:
            print(f"  Scanning output dir: {od}")

            # Find matching input files (filled + correct prefix)
            files = [
                p for p in od.rglob("*.csv")
                if p.name.lower().endswith(DATA_SUFFIX.lower())
                and any(p.name.lower().startswith(pref.lower()) for pref in DATA_PATTERNS)
            ]

            if not files:
                print("   (no matching entropy input files here)")
                continue

            print(f"  Found {len(files)} file(s) in {od}")
            total_files += len(files)

            for f in sorted(files):
                try:
                    process_file(f)
                    processed += 1
                except Exception as e:
                    print(f"  ERROR on {f}: {e}")

    print("\n" + "=" * 80)
    print(f"Done. Processed {processed}/{total_files} files.")


if __name__ == "__main__":
    main()
