#!/usr/bin/env python3

"""

Step01_Sonra_GPS_Preprocess.py

Authors: Ryan Howard (@RJKH1994) & Matt Bridge (@MWBRIDGE)

- The purpose of this code is to extract necessary columns from individual player raw GPS data files obtained from SONRA (Statsports).
- The code will iterate through every folder inside MASTER_FOLDER. Each subfolder is treated as an individual Root_folder. The root_folder will be an individual game.
- The Root_folder (individual game) name should be *TEAM*_*DATE*_*FormatSize*_*Result*_*Pitch*.
    - Example individual game folder name: AstonVilla_020522_11A_L_Astro11B.
    - Example two team folder name: TEAMA_Study2_3Medium_X_3Medium.
- Individual player files should be name *Position*_*Phase*. For example CB_IP. Ensure if there are more than one player playing in the same named position, the names differ. For example CB2_IP.
- Within each root_folder, there should be a folder named GPS_Data, containing individual phase folders (IP - In Possession; OOP - Out of Possession; AT - Attacking Transition; DT - Defensive Transition).
- The pipeline for the workflow in each root_folder is to
    1) Convert any .xlsx files in GPS_Data to .csv (in GPS_Data folder)
    2) Create GPS_Data_new folder and write processed *_new.csv files into each retained phase folder.
    3) Create Match_data.xlsx in GPS_Data_new with split start/end times per individual player CSV.

Inputs (per Root_folder/game):
- GPS_Data/                (contains phase subfolders and player files; .xlsx and/or .csv)
    - AT/
    - DT/
    - IP/
    - OOP/

Outputs (per Root_folder/game):
- GPS_Data_new/            (mirrors the original GPS_Data folder structure)
    - <phase>/
        - *_new.csv        (processed player files)
        - Match_data.xlsx  (start/end time summary for each *_new.csv in that folder)

Notes:
- Raw SONRA exports include a 'Time' column formatted like MM:SS or MM:SS.s (any fractional seconds accepted).
- Raw SONRA exports include 'Lat' and 'Lon' columns (these are renamed to 'latitude' and 'longitude').
- Downsampling is applied by retaining every 10th row starting from row 2 (index 1). This is intentional to reduce file size / standardise sampling rate (e.g., if the source is ~100 Hz, this approximates ~10 Hz).
- This script does not overwrite raw files; it creates a parallel GPS_Data_new folder for processed outputs.
"""

import os
import pandas as pd

# CONFIGURATION #

MASTER_FOLDER = '/Users/ryanhoward/Desktop/Data' # Change this to the master folder containing individual game folders
GPS_FOLDER_NAME = 'GPS_Data' # the folder containing the GPS data must be named GPS_Data
NEW_SUFFIX = '_new'  # output folder suffix -> GPS_Data_new

# Columns to drop from raw exports (We only drop those that exist in each file to avoid errors)
COLUMNS_TO_DROP = [
    'Player Display Name', 'Hacc', 'Hdop', 'Quality of Signal', 'No. of Satellites',
    'Instantaneous Acceleration Impulse', 'Gyro Yro X', 'Gyro Y', 'Gyro Z'
]


# FUNCTIONS #

# FUNCTION 1: Convert .xlsx files to .csv in root_folder

def convert_xlsx_to_csv_in_gps(root_folder: str):
    """
    Convert any .xlsx files under <root_folder>/GPS_Data (including subfolders) into .csv files.

    Rationale:
    - SONRA exports may contain Excel files. Converting to CSV standardises downstream processing.
    - Conversion is done "in-place" (CSV saved alongside the original XLSX) and does not delete the XLSX.
    """
    gps_root = os.path.join(root_folder, GPS_FOLDER_NAME)
    if not os.path.isdir(gps_root):
        # If the folder is missing, the root_folder is not in the expected structure.
        return

    for subdir, _, files in os.walk(gps_root):
        for file in files:
            if file.lower().endswith('.xlsx'):
                filepath = os.path.join(subdir, file)
                try:
                    df = pd.read_excel(filepath)
                    csv_filename = filepath[:-5] + '.csv'
                    df.to_csv(csv_filename, index=False)
                    print(f"Converted xlsx->csv: {csv_filename}")
                except Exception as e:
                    # Fail-safe: continue processing other files rather than stopping the entire run
                    print(f"Error processing xlsx conversion {filepath}: {e}")

# FUNCTION 2: Convert time column into seconds from 00:00

def _normalize_time_to_seconds(series: pd.Series) -> pd.Series:
    """
    Convert SONRA 'Time' values into numeric seconds (float), rounded to 0.1s.

    Expected input formats (examples):
    - "01:23"      (MM:SS)
    - "01:23.4"    (MM:SS.s)
    - "01:23.45"   (MM:SS.ss)

    Method:
    - Coerces all values to strings, ensures a fractional component exists (adds '.0' if needed),
      parses with pandas datetime, then converts to total seconds.

    Notes:
    - Any unparsable values become NaT -> NaN seconds due to errors='coerce'.
    - Rounding to 0.1s aligns with typical 10 Hz processing outputs.
    """
    # Stringify safely (handles mixed types, avoids crashes on non-string entries)
    s = series.astype(str).str.strip()

    # Ensure a fractional part exists so '%M:%S.%f' parsing works consistently
    s = s.apply(lambda x: (x + ".0") if ("." not in x and ":" in x) else x)

    # Parse as datetime using minutes:seconds.microseconds
    t = pd.to_datetime(s, format='%M:%S.%f', errors='coerce')

    # Convert to seconds from 00:00
    secs = (t.dt.hour * 3600) + (t.dt.minute * 60) + t.dt.second + (t.dt.microsecond / 1e6)
    return secs.round(1)

# FUNCTION 3: Iterate through all files in gps subfolders. Write _new files into GPS_Data_new folder, retaining phase subfolders.

def process_files_in_gps_subfolders(root_folder: str, new_directory_suffix: str = NEW_SUFFIX):
    """
    Process all CSV files under <root_folder>/GPS_Data and write cleaned/downsampled CSVs into <root_folder>/GPS_Data_new (mirroring the folder structure).

    Processing steps per file:
    1) Drop non-essential raw export columns (if present)
    2) Rename 'Lat'/'Lon' -> 'latitude'/'longitude'
    3) Downsample rows by selecting every 10th row starting from index 1
    4) Convert 'Time' to seconds (float), rounded to 0.1s

    Why mirror folder structure?
    - Keeps phase grouping identical to the raw data structure (AT/DT/IP/OOP etc.)
    - Makes later steps (match windowing, merging, analysis) consistent across games/phases
    """
    gps_path = os.path.join(root_folder, GPS_FOLDER_NAME)
    if not os.path.isdir(gps_path):
        print(f"Skipping (no {GPS_FOLDER_NAME}): {root_folder}")
        return

    # Output directory (parallel to GPS_Data)
    new_gps_root = os.path.join(root_folder, GPS_FOLDER_NAME + new_directory_suffix)
    os.makedirs(new_gps_root, exist_ok=True)

    for gps_root, _, gps_files in os.walk(gps_path):
        for file in gps_files:
            if not file.lower().endswith('.csv'):
                continue

            file_path = os.path.join(gps_root, file)

            # Mirror the folder structure into GPS_Data_new
            rel_dir = os.path.relpath(gps_root, gps_path)  # subpath under GPS_Data
            out_dir = os.path.join(new_gps_root, rel_dir)
            os.makedirs(out_dir, exist_ok=True)

            base_name = os.path.splitext(file)[0]

            # Avoid doubling '_new' if a file is already named *_new.csv
            out_file = f"{base_name}_new.csv" if not base_name.lower().endswith("_new") else f"{base_name}.csv"
            out_path = os.path.join(out_dir, out_file)

            try:
                data = pd.read_csv(file_path)

                # Drop unwanted columns (defensive: only drops those that exist)
                data = data.drop(columns=[c for c in COLUMNS_TO_DROP if c in data.columns], errors='ignore')

                # Standardise coordinate column names for downstream scripts
                data.rename(columns={'Lat': 'latitude', 'Lon': 'longitude'}, inplace=True)

                # Downsample:
                # Keep the header and every 10th row starting from the 2nd row (index 1).
                # This reduces file size and standardises sampling frequency for analysis.
                result_df = data.iloc[1::10].copy()

                if 'Time' not in result_df.columns:
                    raise ValueError("Missing 'Time' column.")

                # Normalize Time -> numeric seconds
                result_df['Time'] = _normalize_time_to_seconds(result_df['Time'])

                # Save processed file
                result_df.to_csv(out_path, index=False)
                print(f"Processed and saved: {out_path}")

            except Exception as e:
                # Fail-safe: do not halt the whole pipeline for one bad file
                print(f"Error processing file column removal {file_path}: {e}")

# FUNCTION 4: Build Match_data.xlsx in each folder under GPS_Data_new.

def extract_time_from_csv(folder_path: str):
    """
    For each directory under folder_path:
    - Create Match_data.xlsx containing the first and last Time value for each CSV in that directory.

    Rationale:
    - Later processing scripts often require aligning players to a common overlap window. Match_data.xlsx provides split start/end times per player file, per phase folder.
    - The file is created within each folder that contains processed CSVs (e.g., each phase folder).
    """
    for root, _, files in os.walk(folder_path):
        combined_data = {'Filename': [], 'Split Start Time': [], 'Split End Time': []}

        for file in files:
            if not file.lower().endswith('.csv'):
                continue

            file_path = os.path.join(root, file)
            try:
                df = pd.read_csv(file_path)
                if df.empty or 'Time' not in df.columns:
                    continue

                # Start/end of the processed time series (rounded to 0.1s)
                first_time = round(float(df.loc[df.index[0], 'Time']), 1)
                last_time = round(float(df.loc[df.index[-1], 'Time']), 1)

                combined_data['Filename'].append(file)
                combined_data['Split Start Time'].append(first_time)
                combined_data['Split End Time'].append(last_time)

            except Exception as e:
                print(f"Error processing file in time extract {file_path}: {e}")

        # Only write Match_data.xlsx if at least one CSV was present in this folder
        if combined_data['Filename']:
            combined_df = pd.DataFrame(combined_data)
            out_xlsx = os.path.join(root, 'Match_data.xlsx')
            combined_df.to_excel(out_xlsx, index=False)
            print(f"Combined data saved to: {out_xlsx}")

# FUNCTION 5: Identify root_folders in master_folder. It will only be counted as a root_folder if it contains a GPS_Data folder.

def iter_root_folders(master_folder: str) -> list[str]:
    """
    Treat each immediate subfolder of master_folder as a Root_folder (game folder) IF it contains a GPS_Data directory.

    Notes:
    - This function only checks one level down (immediate underneath MASTER_FOLDER). If your game folders are nested deeper, this logic would need extending.
    """
    roots = []
    if not os.path.isdir(master_folder):
        raise FileNotFoundError(f"MASTER_FOLDER does not exist: {master_folder}")

    for name in sorted(os.listdir(master_folder)):
        p = os.path.join(master_folder, name)
        if not os.path.isdir(p):
            continue
        if os.path.isdir(os.path.join(p, GPS_FOLDER_NAME)):
            roots.append(p)

    return roots


# MAIN WORKFLOW

def main():
    """
    Run the preprocessing pipeline across all game folders in MASTER_FOLDER.

    Workflow per Root_folder:
    1) Convert any .xlsx to .csv in GPS_Data
    2) Process .csv -> GPS_Data_new (drop columns, rename, downsample, normalise Time)
    3) Create Match_data.xlsx in each processed folder for alignment/quality checks
    """
    root_folders = iter_root_folders(MASTER_FOLDER)
    if not root_folders:
        print(f"No subfolders with '{GPS_FOLDER_NAME}' found inside:\n  {MASTER_FOLDER}")
        return

    print(f"Found {len(root_folders)} root folder(s) to process.")
    for root_folder in root_folders:
        print("\n" + "=" * 70)
        print(f"ROOT: {root_folder}")

        # 1) Convert xlsx -> csv in GPS_Data (in-place)
        convert_xlsx_to_csv_in_gps(root_folder)

        # 2) Process csv -> GPS_Data_new (mirrors folder structure)
        process_files_in_gps_subfolders(root_folder, new_directory_suffix=NEW_SUFFIX)

        # 3) Build Match_data.xlsx summaries for each folder under GPS_Data_new
        processed_folder = os.path.join(root_folder, GPS_FOLDER_NAME + NEW_SUFFIX)
        extract_time_from_csv(processed_folder)

    print("\nAll folders processed.")


if __name__ == "__main__":
    main()
