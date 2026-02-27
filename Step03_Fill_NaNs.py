#!/usr/bin/env python3

"""

Step03_Fill_NaNs.py

Authors: Ryan Howard (@RJKH1994) & Matt Bridge (@MWBRIDGE)

- The purpose of this code is to fill missing (NaN) player coordinate values in processed GPS tracking files using a time-based adjacent-point approach.
- The code will iterate through every folder inside MASTER_FOLDER. Each subfolder is treated as an individual Root_folder (i.e., an individual game).
- Within each root_folder, there should be an Output folder containing phase-level team tracking files created by the pitch conversion / merging script (e.g., AT_GPS_Data_new_10Hz.csv, DT_GPS_Data_new_10Hz.csv, etc.).
- The pipeline for the workflow in each root_folder is to:
    1) Read each *_10Hz.csv file in Output.
    2) For each player coordinate column, identify NaN values and attempt to fill them using values above/below in time.
    3) Save a new file with suffix _filled.csv (original file is not overwritten).

Fill logic:
- If both adjacent samples exist and are exactly 0.1s either side (EXPECTED_DT), fill using the average of above and below.
- If there is a time gap on one side (i.e., > EXPECTED_DT), prefer the value from the other side (carry forward/backward).
- If only one adjacent sample exists at EXPECTED_DT, fill using that value.
- If adjacent samples do not meet the expected timing criteria, leave as NaN.

Inputs (per Root_folder/game):
- Output/*_10Hz.csv
    - First column must be Time (seconds)
    - Remaining columns are player coordinates (e.g., <PLAYER>_x, <PLAYER>_y)

Outputs (per Root_folder/game):
- Output/*_10Hz_filled.csv
    - Same structure as input, with NaNs filled where possible

Key assumptions / notes:
- EXPECTED_DT is 0.1 seconds (10 Hz). If sampling differs, update EXPECTED_DT accordingly.
- This method performs a simple local interpolation/imputation and is intended to address isolated dropouts rather than long missing segments. If dropouts are frequent or extended, a more robust approach may be required.
- Files are written with "_filled.csv" suffix so that original outputs remain unchanged for reproducibility.

"""

import os
import pandas as pd


# CONFIGURATION #

MASTER_FOLDER = '/Users/ryanhoward/Desktop/Data' # Change this to the master folder containing individual game folders
OUTPUT_FOLDER_NAME = 'Output'
TIME_COL_INDEX = 0          # assumes Time is first column
EXPECTED_DT = 0.1           # seconds (10 Hz)


# FUNCTIONS #

# FUNCTION 1: Fill NaNs in a single file using time-based adjacent-point averaging

def fill_na_with_time_based_average(file_path: str):
    """
    Fill NaN values in a phase-level team tracking file using adjacent time points.

    Method:
    - For each NaN at time t:
        - Check the sample immediately above (t_above) and below (t_below)
        - Use EXPECTED_DT (default 0.1s) to determine whether points are truly adjacent in time
        - If both sides are valid and adjacent, fill with the mean
        - If one side has a time gap (> EXPECTED_DT), use the other side value
        - If only one side exists and is adjacent, copy that value
    """
    df = pd.read_csv(file_path)

    time_col = df.columns[TIME_COL_INDEX]

    # Iterate through each player coordinate column (skip Time)
    for col in df.columns[1:]:
        # Row-by-row fill to apply time-aware logic rather than simple forward/back fill
        for i in range(len(df)):
            if pd.isna(df.at[i, col]):
                t = df.at[i, time_col]

                # Above (previous row)
                if i > 0:
                    t_above = df.at[i - 1, time_col]
                    v_above = df.at[i - 1, col]
                else:
                    t_above = v_above = None

                # Below (next row)
                if i < len(df) - 1:
                    t_below = df.at[i + 1, time_col]
                    v_below = df.at[i + 1, col]
                else:
                    t_below = v_below = None

                # Fill logic:
                # - Use adjacency checks to avoid filling across gaps in time
                if not pd.isna(v_above) and not pd.isna(v_below):
                    if round(abs(t - t_above), 1) == EXPECTED_DT and round(abs(t - t_below), 1) == EXPECTED_DT:
                        df.at[i, col] = (v_above + v_below) / 2
                    elif round(abs(t - t_above), 1) > EXPECTED_DT:
                        df.at[i, col] = v_below
                    elif round(abs(t - t_below), 1) > EXPECTED_DT:
                        df.at[i, col] = v_above

                elif not pd.isna(v_above) and round(abs(t - t_above), 1) == EXPECTED_DT:
                    df.at[i, col] = v_above

                elif not pd.isna(v_below) and round(abs(t - t_below), 1) == EXPECTED_DT:
                    df.at[i, col] = v_below

    out_path = file_path.replace('.csv', '_filled.csv')
    df.to_csv(out_path, index=False)
    print(f"    Filled: {os.path.basename(out_path)}")


# FUNCTION 2: Process one Root_folder (game folder)

def process_match_folder(match_root: str):
    """
    Process a single Root_folder by filling NaNs in all phase-level *_10Hz.csv files found in Output/.

    Notes:
    - Only processes files ending in '10hz.csv'
    - Skips files that are already '_filled.csv'
    """
    output_dir = os.path.join(match_root, OUTPUT_FOLDER_NAME)

    if not os.path.isdir(output_dir):
        return

    print(f"\nProcessing Output folder: {output_dir}")

    for file in os.listdir(output_dir):
        if file.lower().endswith('10hz.csv') and not file.lower().endswith('_filled.csv'):
            file_path = os.path.join(output_dir, file)
            try:
                fill_na_with_time_based_average(file_path)
            except Exception as e:
                print(f"    Error processing {file}: {e}")


# MAIN WORKFLOW #

def main():
    if not os.path.isdir(MASTER_FOLDER):
        raise FileNotFoundError(f"MASTER_FOLDER not found: {MASTER_FOLDER}")

    match_folders = [
        os.path.join(MASTER_FOLDER, d)
        for d in os.listdir(MASTER_FOLDER)
        if os.path.isdir(os.path.join(MASTER_FOLDER, d))
    ]

    print(f"Found {len(match_folders)} match folders.")

    for match_root in match_folders:
        process_match_folder(match_root)

    print("\nAll folders processed.")


if __name__ == "__main__":
    main()
