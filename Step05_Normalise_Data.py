#!/usr/bin/env python3

"""

Step05_Normalize_Data.py

Authors: Ryan Howard (@RJKH1994) & Matt Bridge (@MWBRIDGE)

- The purpose of this code is to normalise (scale) zeroed team tracking data to a consistent reference pitch size, while also saving relative (0–1) coordinate versions of the same data.
- The code will iterate through folders under MASTER_FOLDER and identify Output folders containing Zeroed_* tracking files.
- Within each Output folder, the script expects Pitch_Lengths.csv to be available either in the same folder or within a limited number of parent folders (controlled by MAX_PARENT_LEVELS).
- The pipeline for the workflow is to:
    1) Identify candidate files matching the expected naming convention (Zeroed_..._GPS_Data_new_10Hz_filled.csv).
    2) Read pitch length/width from Pitch_Lengths.csv for that match.
    3) Convert absolute coordinates to relative coordinates (x/pitch_length, y/pitch_width).
    4) Scale relative coordinates to a reference pitch size (L_REF × W_REF).
    5) Save two outputs into a Normalized folder:
        - Normalized_* : coordinates scaled to reference pitch size in metres
        - Scaled_*     : coordinates stored as relative values (0–1) with *_relx and *_rely columns

Inputs (per Root_folder/game):
- Output/Pitch_Lengths.csv
    - Must contain columns: pitch_length, pitch_width
- Output/Zeroed_*_GPS_Data_new_10Hz_filled.csv
    - Player coordinate columns in *_x and *_y pairs

Outputs (per Root_folder/game):
- Output/Normalized/Normalized_Zeroed_*_GPS_Data_new_10Hz_filled.csv
- Output/Normalized/Scaled_Zeroed_*_GPS_Data_new_10Hz_filled.csv

Notes:
- Input files must already be zeroed so that pitch origin corresponds to (0, 0) prior to normalisation.
- The reference pitch size is defined by L_REF and W_REF and can be set to standard dimensions (e.g., 105 × 68 m).
- This normalisation is intended to allow comparison across matches/pitches of different measured sizes.

"""

import os
import re
import pandas as pd
from typing import List, Tuple


# CONFIGURATION #

MASTER_FOLDER = (
    '/Users/ryanhoward/Desktop/Data' # Change this to the master folder containing individual game folders
)

L_REF = 105.0
W_REF = 68.0

PITCH_FILENAME = "Pitch_Lengths.csv"
MAX_PARENT_LEVELS = 1                 # how many folders up to search for Pitch_Lengths.csv
ONLY_PROCESS_OUTPUT_FOLDERS = True    # True = only folders literally named "Output"


# FUNCTIONS #

## Find Pitch_Lengths.csv in parent folders

# FUNCTION 1: Search for Pitch_Lengths.csv in current and parent folders

def find_file_in_parents(start_folder: str, filename: str, max_levels: int = 8) -> str:
    """
    Search for a file by walking up the folder hierarchy.

    This is used to locate Pitch_Lengths.csv relative to an Output folder without hard-coding paths.

    Inputs:
    - start_folder: folder to begin searching from
    - filename: file to search for (e.g., Pitch_Lengths.csv)
    - max_levels: maximum number of parent folders to check

    Returns:
    - Full path to the detected file
    """
    current = os.path.abspath(start_folder)

    for _ in range(max_levels + 1):
        candidate = os.path.join(current, filename)
        if os.path.exists(candidate):
            return candidate

        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    raise FileNotFoundError(
        f"Could not find {filename} in {start_folder} or up to {max_levels} parent folder(s)."
    )


## Detect coordinate column pairs

# FUNCTION 2: Detect *_x / *_y coordinate pairs

def detect_xy_pairs(columns) -> List[Tuple[str, str]]:
    """
    Identify coordinate column pairs in the input data.

    Logic:
    - Find all columns ending in '_x'
    - For each, check that the matching '_y' column exists
    - Return pairs in the order they appear in the file
    """
    cols = list(columns)
    x_cols = [c for c in cols if c.endswith("_x")]
    pairs = []
    for xc in x_cols:
        yc = xc[:-2] + "_y"
        if yc in cols:
            pairs.append((xc, yc))
    pairs.sort(key=lambda p: cols.index(p[0]))
    return pairs


## Normalisation

# FUNCTION 3: Normalise coordinates to a reference pitch size and extract relative coordinates

def normalise_and_extract_scaled(
    df: pd.DataFrame,
    pitch_length: float,
    pitch_width: float,
    L_ref: float = 105.0,
    W_ref: float = 68.0
):
    """
    Convert zeroed pitch coordinates into:
    - Normalised coordinates scaled to a reference pitch (L_ref × W_ref)
    - Relative coordinates stored as 0–1 proportions of the measured pitch size

    Outputs:
    - df: the original DataFrame with *_x/*_y replaced by scaled coordinates in metres (reference pitch)
    - df_scaled: a new DataFrame containing *_relx and *_rely columns (relative values) and Time if present
    """
    df = df.copy()
    df_scaled = pd.DataFrame()

    xy_pairs = detect_xy_pairs(df.columns)
    if not xy_pairs:
        raise ValueError("No *_x/*_y coordinate pairs found")

    for x_col, y_col in xy_pairs:
        X_rel = df[x_col] / pitch_length
        Y_rel = df[y_col] / pitch_width

        base = x_col[:-2]
        df_scaled[f"{base}_relx"] = X_rel
        df_scaled[f"{base}_rely"] = Y_rel

        df[x_col] = X_rel * L_ref
        df[y_col] = Y_rel * W_ref

    if "Time" in df.columns:
        df_scaled["Time"] = df["Time"]

    return df, df_scaled


## Process one folder (Output folder)

# FUNCTION 4: Normalise every matching Zeroed_*.csv file in a folder

def batch_normalise_zeroed_files_in_folder(
    input_folder: str,
    L_ref: float = 105.0,
    W_ref: float = 68.0
) -> int:
    """
    Normalise every matching Zeroed_*.csv in input_folder.

    Notes:
    - Only processes files matching the expected naming pattern for phases (AT/IP/DT/OOP).
    - Writes outputs into input_folder/Normalized.
    - Returns the number of files processed.
    """
    pattern = re.compile(
        r"^Zeroed_.*(AT|IP|DT|OOP).*_GPS_Data_new_10Hz_filled\.csv$",
        re.IGNORECASE
    )

    candidates = [f for f in os.listdir(input_folder) if pattern.match(f)]
    if not candidates:
        return 0

    pitch_csv_path = find_file_in_parents(
        start_folder=input_folder,
        filename=PITCH_FILENAME,
        max_levels=MAX_PARENT_LEVELS
    )

    pitch_info = pd.read_csv(pitch_csv_path)
    pitch_length = float(pitch_info["pitch_length"].iloc[0])
    pitch_width  = float(pitch_info["pitch_width"].iloc[0])

    print(f"\n===================================================")
    print(f"Folder: {input_folder}")
    print(f"Pitch file: {pitch_csv_path}")
    print(f"Detected pitch: {pitch_length:.3f} x {pitch_width:.3f} m")

    normalized_folder = os.path.join(input_folder, "Normalized")
    os.makedirs(normalized_folder, exist_ok=True)

    processed = 0

    for fname in sorted(candidates):
        phase = pattern.match(fname).group(1).upper()
        print(f"\nProcessing: {fname}  (Phase: {phase})")

        in_path = os.path.join(input_folder, fname)
        df = pd.read_csv(in_path)

        df_norm, df_scaled = normalise_and_extract_scaled(
            df,
            pitch_length,
            pitch_width,
            L_ref=L_ref,
            W_ref=W_ref
        )

        norm_path = os.path.join(normalized_folder, f"Normalized_{fname}")
        scaled_path = os.path.join(normalized_folder, f"Scaled_{fname}")

        df_norm.to_csv(norm_path, index=False)
        df_scaled.to_csv(scaled_path, index=False)

        print(f"Saved normalised: {norm_path}")
        print(f"Saved scaled:     {scaled_path}")

        processed += 1

    return processed


## Find all folders to process under MASTER_FOLDER

# FUNCTION 5: Identify target folders containing matching Zeroed_* files

def find_target_folders(master_folder: str) -> List[str]:
    """
    Identify folders that contain matching Zeroed_* tracking files.

    If ONLY_PROCESS_OUTPUT_FOLDERS is True:
    - only folders named exactly 'Output' will be considered.

    Otherwise:
    - any folder containing matching files will be included.
    """
    pattern = re.compile(
        r"^Zeroed_.*(AT|IP|DT|OOP).*_GPS_Data_new_10Hz_filled\.csv$",
        re.IGNORECASE
    )

    targets: List[str] = []

    for root, dirs, files in os.walk(master_folder):
        folder_name = os.path.basename(root)

        if ONLY_PROCESS_OUTPUT_FOLDERS and folder_name != "Output":
            continue

        if any(pattern.match(f) for f in files):
            targets.append(root)

    targets = sorted(set(targets))
    return targets


# MAIN WORKFLOW #

def run_master(
    master_folder: str,
    L_ref: float = 105.0,
    W_ref: float = 68.0
):
    targets = find_target_folders(master_folder)

    print(f"\nMaster folder: {master_folder}")
    print(f"Found {len(targets)} folder(s) to process.")

    total_files = 0
    total_folders_with_files = 0
    skipped_pitch_missing = 0
    skipped_other_error = 0

    for folder in targets:
        try:
            n = batch_normalise_zeroed_files_in_folder(folder, L_ref=L_ref, W_ref=W_ref)
            if n > 0:
                total_folders_with_files += 1
                total_files += n
        except FileNotFoundError as e:
            skipped_pitch_missing += 1
            print(f"\n[SKIP] Pitch file not found for: {folder}")
            print(f"       {e}")
        except Exception as e:
            skipped_other_error += 1
            print(f"\n[ERROR] Failed folder: {folder}")
            print(f"        {type(e).__name__}: {e}")

    print(f"\n================== SUMMARY ==================")
    print(f"Folders found:                  {len(targets)}")
    print(f"Folders processed (had files):  {total_folders_with_files}")
    print(f"Total files processed:          {total_files}")
    print(f"Skipped (pitch missing):        {skipped_pitch_missing}")
    print(f"Errored (other):                {skipped_other_error}")
    print(f"============================================\n")


if __name__ == "__main__":
    run_master(
        master_folder=MASTER_FOLDER,
        L_ref=L_REF,
        W_ref=W_REF
    )
