#!/usr/bin/env python3

"""

Step06_Halftime.py

Authors: Ryan Howard (@RJKH1994) & Matt Bridge (@MWBRIDGE)

- The purpose of this code is to add a 'Half' column (1st / 2nd half) to team tracking files.
- The code will iterate through folders under MASTER_FOLDER and identify relevant CSV files inside:
    - Output folders
    - Normalized folders
- Halftime is inferred from the Time column by detecting a large discontinuity (gap) in time. By default, halftime is defined as a gap greater than HALFTIME_GAP_SECONDS (330 seconds = 5.5 minutes).
- For matches where there is a break in play (any other stoppage than half time), the half time gap may need to be adjusted

- The pipeline for the workflow is to:
    1) Identify candidate CSV files that begin with 'Zeroed_' or 'Normalized_Zeroed_'.
    2) Load each file and confirm it contains a Time column.
    3) Convert Time to numeric and calculate the size of time gaps between consecutive samples.
    4) Assign Half = 1 until the first detected halftime gap, then Half = 2 from that point onwards.
    5) (Optional) If a halftime is detected, flip 2nd-half coordinates so that both halves are aligned to the same attacking direction.
    6) Write changes back into the same CSV file.

Inputs:
- Output/Zeroed_*.csv
- Normalized/Normalized_Zeroed_*.csv

Outputs:
- Overwrites the input files in-place by adding/updating:
    - Half (1 or 2)
    - Optionally updated *_x/*_y coordinates for Half == 2

Key assumptions / notes:
- Time is expected to be in seconds and monotonically increasing within each half.
- This approach assumes there is a single halftime break; if multiple large gaps are detected, a warning is printed.
- If no halftime gap is detected, the file is left as Half = 1 for all rows and a warning is printed.
- EXCLUDE_SUFFIX is used to avoid updating summary outputs that are not tracking time-series files.
- Coordinate flipping:
    - Output folders: uses pitch_length/pitch_width from Pitch_Lengths.csv (searched in current folder and parent folders).
    - Normalized folders: uses reference dimensions (L_REF/W_REF).
    - Flip is applied only when Half exists and Half == 2 rows exist.

"""

import os
import pandas as pd


## CONFIGURATION

MASTER_FOLDER = '/Users/ryanhoward/Documents/University_of_Birmingham/Research Studies/Study 1/Study 1 Data' # Change this to the master folder containing individual game folders
HALFTIME_GAP_SECONDS = 330  # 5.5 minutes

VALID_PREFIXES = ("Zeroed_", "Normalized_Zeroed_")
EXCLUDE_SUFFIX = ("__entropy_summary.csv", "__entropy_transitions.csv")
VALID_PARENT_FOLDERS = {"Output", "Normalized"}

# Optional: flip 2nd-half coordinates so both halves align to the same attacking direction
APPLY_ENDS_FLIP = True
FLIP_Y_AS_WELL = False  # False = flip X only (recommended); True = flip X and Y (180-degree rotation)

# Pitch dimensions for Normalized files
L_REF = 105.0
W_REF = 68.0

# Pitch meta for Output files (searched in folder and parents)
PITCH_META_FILENAME = "Pitch_Lengths.csv"
MAX_PARENT_LEVELS = 2


# FUNCTIONS #

## File/folder filters

# FUNCTION 1: Check whether the folder should be processed

def should_process_folder(folder_path: str) -> bool:
    """
    Restrict processing to Output or Normalized folders.

    This prevents scanning unrelated directories within MASTER_FOLDER.
    """
    return os.path.basename(folder_path) in VALID_PARENT_FOLDERS


# FUNCTION 2: Check whether the file matches the naming rules for processing

def should_process_file(filename: str) -> bool:
    """
    Only process:
    - CSV files that start with VALID_PREFIXES
    - Exclude files with EXCLUDE_SUFFIX (e.g., summary outputs)
    """
    return (
        filename.startswith(VALID_PREFIXES)
        and filename.lower().endswith(".csv")
        and not filename.endswith(EXCLUDE_SUFFIX)
    )


## Halftime detection

# FUNCTION 3: Identify the first index where a halftime gap occurs

def find_halftime_index(time_series: pd.Series, gap_seconds: float):
    """
    Detect the first halftime break index using time discontinuities.

    Logic:
    - Compute time differences between consecutive rows
    - Find where the difference exceeds gap_seconds
    - Return the first such index (or None if none found)
    """
    gaps = time_series.diff()
    idxs = gaps[gaps.gt(gap_seconds)].index
    return int(idxs[0]) if len(idxs) > 0 else None


## Pitch dimension helpers (for coordinate flip)

# FUNCTION 4: Find Pitch_Lengths.csv in current folder or up to N parent folders

def find_file_in_parents(start_folder: str, filename: str, max_levels: int = 2) -> str:
    """
    Search for filename in start_folder and up to max_levels parent folders.

    Returns:
    - Full path to the first match found

    Raises:
    - FileNotFoundError if no match is found
    """
    current = os.path.abspath(start_folder)

    for _ in range(max_levels + 1):
        candidate = os.path.join(current, filename)
        if os.path.isfile(candidate):
            return candidate

        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    raise FileNotFoundError(
        f"Could not find {filename} in {start_folder} or up to {max_levels} parent folder(s)."
    )


# FUNCTION 5: Read pitch_length/pitch_width from Pitch_Lengths.csv

def read_pitch_dims(pitch_csv_path: str):
    """
    Read pitch_length and pitch_width from Pitch_Lengths.csv.

    Returns:
    - (pitch_length, pitch_width) as floats

    Raises:
    - ValueError if required columns are missing or file is empty
    """
    df = pd.read_csv(pitch_csv_path)
    if df.empty:
        raise ValueError(f"{os.path.basename(pitch_csv_path)} is empty: {pitch_csv_path}")

    cols = {str(c).lower().strip(): c for c in df.columns}

    if "pitch_length" not in cols or "pitch_width" not in cols:
        raise ValueError(
            f"Expected columns ['pitch_length','pitch_width'] in {pitch_csv_path}. Found {list(df.columns)}"
        )

    L = float(df.iloc[0][cols["pitch_length"]])
    W = float(df.iloc[0][cols["pitch_width"]])
    return L, W


## Coordinate flip helpers

# FUNCTION 6: Detect coordinate column pairs (*_x and *_y)

def detect_xy_pairs(columns):
    """
    Detect (x, y) coordinate column pairs from a list of columns.

    Returns:
    - List of tuples (x_col, y_col) in file order.
    """
    cols = list(columns)
    x_cols = [c for c in cols if str(c).endswith("_x")]
    pairs = []
    for xc in x_cols:
        yc = xc[:-2] + "_y"
        if yc in cols:
            pairs.append((xc, yc))
    pairs.sort(key=lambda p: cols.index(p[0]))
    return pairs


# FUNCTION 7: Flip second-half coordinates to align ends

def flip_second_half_coords(df: pd.DataFrame, pitch_length: float, pitch_width: float, flip_y_as_well: bool):
    """
    Flip coordinates for rows where Half == 2.

    Default behaviour (flip X only):
    - x := pitch_length - x

    Optional full rotation (flip X and Y):
    - x := pitch_length - x
    - y := pitch_width  - y
    """
    if "Half" not in df.columns:
        return df

    pairs = detect_xy_pairs(df.columns)
    if not pairs:
        return df

    mask = pd.to_numeric(df["Half"], errors="coerce") == 2
    if mask.sum() == 0:
        return df

    for x_col, y_col in pairs:
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

        df.loc[mask, x_col] = pitch_length - df.loc[mask, x_col]
        if flip_y_as_well:
            df.loc[mask, y_col] = pitch_width - df.loc[mask, y_col]

    return df


## File processing

# FUNCTION 8: Add/overwrite the Half column in a single CSV (and optionally flip ends)

def process_csv(file_path: str, parent_folder_name: str):
    """
    Add a Half column to a single tracking CSV.

    Steps:
    - Load file
    - Confirm Time exists
    - Convert Time to numeric
    - Assign Half = 1 for all rows
    - Detect halftime gap and assign Half = 2 from that index onwards
    - If APPLY_ENDS_FLIP is enabled and Half==2 exists, flip second-half coordinates
    - Save updated file in-place
    """
    df = pd.read_csv(file_path)

    if "Time" not in df.columns:
        print(f"[SKIP] No Time column: {file_path}")
        return

    time = pd.to_numeric(df["Time"], errors="coerce")

    df["Half"] = 1

    halftime_index = find_halftime_index(time, HALFTIME_GAP_SECONDS)
    halftime_gaps = time.diff().gt(HALFTIME_GAP_SECONDS).sum()

    if halftime_gaps > 1:
        print(f"[WARN] Multiple halftime gaps ({halftime_gaps}) in {os.path.basename(file_path)}")
    elif halftime_gaps == 0:
        print(f"[WARN] No halftime found in {os.path.basename(file_path)}")

    if halftime_index is not None:
        df.loc[halftime_index:, "Half"] = 2

    if APPLY_ENDS_FLIP:
        # Only attempt to flip if we actually have a detected second half
        if (pd.to_numeric(df["Half"], errors="coerce") == 2).any():
            if parent_folder_name == "Normalized":
                # Normalized files are scaled to the reference dimensions
                df = flip_second_half_coords(df, L_REF, W_REF, FLIP_Y_AS_WELL)
            else:
                # Output files use the true pitch dimensions from Pitch_Lengths.csv
                try:
                    pitch_csv = find_file_in_parents(
                        start_folder=os.path.dirname(file_path),
                        filename=PITCH_META_FILENAME,
                        max_levels=MAX_PARENT_LEVELS
                    )
                    L, W = read_pitch_dims(pitch_csv)
                    df = flip_second_half_coords(df, L, W, FLIP_Y_AS_WELL)
                except Exception as e:
                    print(f"[WARN] Could not flip ends for {os.path.basename(file_path)} (missing pitch meta): {e}")

    df.to_csv(file_path, index=False)
    print(f"[OK] Updated: {file_path}")


# MAIN WORKFLOW #

def run_master(master_folder: str):
    processed = 0

    for root, _, files in os.walk(master_folder):
        if not should_process_folder(root):
            continue

        parent_folder_name = os.path.basename(root)

        for fname in sorted(files):
            if not should_process_file(fname):
                continue

            file_path = os.path.join(root, fname)
            print(f"\nProcessing: {file_path}")

            try:
                process_csv(file_path, parent_folder_name)
                processed += 1
            except Exception as e:
                print(f"[ERROR] Failed: {file_path}")
                print(f"        {type(e).__name__}: {e}")

    print(f"\n===================================")
    print(f"Completed. Total files processed: {processed}")
    print(f"Folders scanned: {', '.join(VALID_PARENT_FOLDERS)}")
    print(f"===================================")


if __name__ == "__main__":
    run_master(MASTER_FOLDER)
