#!/usr/bin/env python3

"""

Step02_Pitch_Normalisation.py

Authors: Ryan Howard (@RJKH1994) & Matt Bridge (@MWBRIDGE)

- The purpose of this code is to convert individual player GPS latitude/longitude data into a consistent, metre-based pitch coordinate system (X/Y) for analysis.
- The code will iterate through every folder inside MASTER_FOLDER. Each subfolder is treated as an individual Root_folder (i.e., an individual game).
- Within each root_folder, there should be:
    - Pitch_coordinates.xlsx containing four pitch corner coordinates (latitude/longitude).
    - A GPS_Data_new folder containing individual phase folders (IP - In Possession; OOP - Out of Possession; AT - Attacking Transition; DT - Defensive Transition).
    - Match_data.xlsx inside each phase folder (created by A_Remove_Columns.py script).
    
- The pipeline for the workflow in each root_folder is to:
    1) Convert pitch corner latitude/longitude into metre-based coordinates.
    2) Estimate pitch length/width and compute a rotation so the pitch long-axis aligns with the X-axis.
    3) Optionally swap X/Y axes if the long-axis remains aligned with Y after rotation.
    4) Identify a common overlap window across player files using Match_data.xlsx.
    5) Trim, convert, rotate, and merge all player files into a single team-level tracking file.
    6) Save pitch metadata and phase-level team tracking files into an Output folder.

Inputs (per Root_folder/game):
- Pitch_coordinates.xlsx
    - Must contain columns: ['latitude', 'longitude']
    - Must contain exactly 4 rows (pitch corners; any order)
- GPS_Data_new/<PHASE>/
    - Player CSVs (processed outputs from the extraction script)
        - Must contain Time, latitude, longitude (column names) or these as the first 3 columns
    - Match_data.xlsx
        - Must contain columns: Filename, Split Start Time, Split End Time

Outputs (per Root_folder/game):
- Output/Pitch_Lengths.csv
    - pitch_length, pitch_width (in metres)
    - Opitch_x, Opitch_y (minimum rotated X/Y; used as pitch origin offsets)
    - SWITCH_XY (0/1; whether axes were swapped post-rotation)
- Output/<PHASE>_GPS_Data_new_10Hz.csv
    - Team-level tracking file with columns: Time, *PLAYER*_x, *PLAYER*_y, ...
    - Missing samples are written as "NA"

Notes:
- Player identity is inferred from the first 7 characters of each filename (playername = file[0:7]). Ensure this is unique/consistent.
- Start/end indices are located using exact Time matches when possible, with a small microsecond tolerance fallback to handle float rounding.
- If pitch dimensions appear implausible, sanity-check Pitch_coordinates.xlsx values and the lat/lon -> metres conversion.

"""

import os
import math
import numpy as np
import pandas as pd


# CONFIGURATION #

MASTER_FOLDER = '/Users/ryanhoward/Desktop/Data' # Change this to the master folder containing individual game folders
folders = ["AT", "DT", "OOP", "IP"]
MATCH = 'GPS_Data_new'   # folder name under each Root_folder that contains phase folders


# FUNCTIONS #

# FUNCTION 1: Convert latitude/longitude to metre-based coordinates (UTM-like)

def latlon_to_utm_m(lat: float, lon: float) -> tuple[float, float]:
    """
    Convert latitude/longitude (degrees) into planar metre coordinates.

    Notes:
    - This is a custom UTM-like implementation to avoid external dependencies.
    - Local transverse-Mercator style projection used to obtain consistent metre-based coordinates for small spatial extents (e.g. a football pitch).
    - Absolute UTM zone correctness is not required; the projection is used solely for relative geometry prior to pitch rotation and normalisation.
    - The output is scaled to metres (Earth radius specified in km).
    """
    a = 6378.137
    e = 0.0818192
    k0 = 0.9996
    E0 = 500
    N0 = 0

    Zonenum = math.floor(lon / 6) + 31
    lamda0 = (Zonenum - 1) * 6 - 180 + 3
    lamda0 = lamda0 * math.pi / 180

    phi = lat * math.pi / 180
    lamda = lon * math.pi / 180

    v = 1 / math.sqrt(1 - e ** 2 * math.sin(phi) ** 2)
    A = (lamda - lamda0) * math.cos(phi)
    T = math.tan(phi) ** 2
    C = e ** 2 * math.cos(phi) ** 2 / (1 - e ** 2)

    s = (1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256) * phi - \
        (3 * e ** 2 / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * math.sin(2 * phi) + \
        (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * math.sin(4 * phi) - \
        35 * e ** 6 / 3072 * math.sin(6 * phi)

    UTME = E0 + k0 * a * v * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2) * A ** 5 / 120)
    UTMN = N0 + k0 * a * (s + v * math.tan(phi) *
                          (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24 +
                           (61 - 58 * T + T ** 2) * A ** 6 / 720))

    UTME *= 1000
    UTMN *= 1000
    return UTME, UTMN


# FUNCTION 2: Convert pitch corner coordinates into metre-based XY array

def pitch_latlon_to_xy(df_pitch: pd.DataFrame) -> np.ndarray:
    """
    Convert pitch corner latitude/longitude coordinates into metre-based XY coordinates.

    Assumptions:
    - Pitch_coordinates.xlsx contains exactly four rows (pitch corners).
    - Columns are named 'latitude' and 'longitude'.
    """
    if "latitude" not in df_pitch.columns or "longitude" not in df_pitch.columns:
        raise ValueError(
            f"Pitch_coordinates.xlsx must have columns ['latitude','longitude']. "
            f"Found {list(df_pitch.columns)}"
        )

    pts = []
    for lat, lon in zip(df_pitch["latitude"].to_list(),
                        df_pitch["longitude"].to_list()):
        x, y = latlon_to_utm_m(float(lat), float(lon))
        pts.append([x, y])

    pts = np.asarray(pts, dtype=float)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected 4 pitch corners, got shape {pts.shape}")

    return pts


## Corner ordering + pitch axes

# FUNCTION 3: Order pitch corners counter-clockwise

def order_corners_ccw(pts: np.ndarray) -> np.ndarray:
    """
    Order pitch corners counter-clockwise around the centroid.

    This ensures a consistent perimeter ordering regardless of input order.
    """
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


# FUNCTION 4: Compute edge lengths of the pitch perimeter

def edge_lengths(loop: np.ndarray) -> np.ndarray:
    """Compute Euclidean lengths of the four edges of the pitch."""
    nxt = np.roll(loop, -1, axis=0)
    return np.sqrt(((nxt - loop) ** 2).sum(axis=1))


# FUNCTION 5: Estimate pitch geometry (length and width)

def compute_pitch_geometry(loop: np.ndarray):
    """
    Estimate pitch length and width from ordered corner coordinates.

    Method:
    - The two longest edges are treated as pitch length.
    - The two shortest edges are treated as pitch width.
    """
    edges = edge_lengths(loop)
    long2 = np.sort(edges)[-2:]
    short2 = np.sort(edges)[:2]

    pitch_length = float(np.mean(long2))
    pitch_width = float(np.mean(short2))
    return pitch_length, pitch_width, edges


# FUNCTION 6: Select pitch origin and long-edge direction

def pick_origin_and_long_edge(loop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Select a consistent origin corner (p0) and a connected corner (p1) defining the pitch long-axis direction.
    """
    i0 = int(np.argmin(loop[:, 0] + loop[:, 1]))
    p0 = loop[i0]

    p_prev = loop[(i0 - 1) % 4]
    p_next = loop[(i0 + 1) % 4]

    p1 = p_next if np.linalg.norm(p_next - p0) >= np.linalg.norm(p_prev - p0) else p_prev
    return p0, p1


# FUNCTION 7: Build rotation matrix to align pitch length with X-axis

def rotation_matrix_to_align_length_with_x(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """
    Compute a 2x2 rotation matrix that aligns the pitch long-axis with the X-axis.
    """
    v = p1 - p0
    angle = math.atan2(float(v[1]), float(v[0]))
    c = math.cos(-angle)
    s = math.sin(-angle)

    return np.array([[c, -s],
                     [s,  c]], dtype=float)


# FUNCTION 8: Apply rotation to row-vector coordinates

def apply_rotation_rowvec(pts: np.ndarray, RM: np.ndarray) -> np.ndarray:
    """Apply a 2D rotation matrix to row-vector coordinates."""
    return pts @ RM.T


## Match info

# FUNCTION 9: Read Match_data.xlsx

def read_match_data(match_xlsx: str) -> pd.DataFrame:
    """
    Read Match_data.xlsx containing split start/end times for each player file.

    This file is used to determine a common overlap window across players.
    """
    return pd.read_excel(
        match_xlsx,
        sheet_name="Sheet1",
        usecols=['Filename', 'Split Start Time', 'Split End Time']
    )


## Player processing

# FUNCTION 10: Build list of player CSV files

def build_player_file_list(folder_dir: str) -> list[str]:
    """
    List all player CSV files within a phase folder.
    """
    files = [f for f in os.listdir(folder_dir) if f.lower().endswith(".csv")]
    return [f for f in files if f != ".DS_Store"]


# FUNCTION 11: Read individual player CSV

def read_player_latlon_csv(path: str) -> pd.DataFrame:
    """
    Read a processed player CSV containing Time, latitude, and longitude.

    Note:
    - Assumes these are the first three columns in the file.
    """
    return pd.read_csv(path, usecols=[0, 1, 2])


# FUNCTION 12: Find index bounds for overlap window

def find_indices_for_window(position: pd.DataFrame,
                            StartTS: float,
                            EndTS: float) -> tuple[int, int]:
    """
    Identify start and end row indices corresponding to a given time window.

    Strategy:
    - Attempt exact matches first.
    - If not found, search within a small microsecond tolerance to handle float rounding.
    """
    # Start
    if len(position.loc[position['Time'] == StartTS]) >= 1:
        StartIndex = position.loc[position['Time'] == StartTS].index[0]
    else:
        StartIndex = None
        for i in range(1, 10):
            if len(position.loc[
                (position['Time'] * 1_000_000).astype(int)
                == int(StartTS * 1_000_000) + i
            ]) >= 1:
                StartIndex = position.loc[
                    (position['Time'] * 1_000_000).astype(int)
                    == int(StartTS * 1_000_000) + i
                ].index[0]
                break
        if StartIndex is None:
            raise ValueError(f"StartTS {StartTS} not found.")

    # End
    if len(position.loc[position['Time'] == EndTS]) >= 1:
        EndIndex = position.loc[position['Time'] == EndTS].index[-1]
    else:
        EndIndex = None
        for i in range(1, 10):
            if len(position.loc[
                (position['Time'] * 1_000_000).astype(int)
                == int(EndTS * 1_000_000) - i
            ]) >= 1:
                EndIndex = position.loc[
                    (position['Time'] * 1_000_000).astype(int)
                    == int(EndTS * 1_000_000) - i
                ].index[-1]
                break
        if EndIndex is None:
            raise ValueError(f"EndTS {EndTS} not found.")

    return StartIndex, EndIndex


# FUNCTION 13: Convert and rotate individual player tracking

def convert_and_rotate_player(position: pd.DataFrame,
                              RM: np.ndarray,
                              SWITCH_XY: bool) -> pd.DataFrame:
    """
    Convert player latitude/longitude tracking into rotated metre-based X/Y coordinates.
    """
    lats = position["latitude"].to_numpy(dtype=float)
    lons = position["longitude"].to_numpy(dtype=float)

    X = np.empty_like(lats, dtype=float)
    Y = np.empty_like(lats, dtype=float)

    for i in range(len(lats)):
        x, y = latlon_to_utm_m(float(lats[i]), float(lons[i]))
        X[i] = x
        Y[i] = y

    xy = np.column_stack([X, Y])
    xy_rot = xy @ RM.T

    if SWITCH_XY:
        xy_rot = xy_rot[:, [1, 0]]

    return pd.DataFrame({
        "Time": position["Time"].to_numpy(dtype=float),
        "X": xy_rot[:, 0],
        "Y": xy_rot[:, 1],
    })


# FUNCTION 14: Build team-level tracking file

def TeamTracking(file_dir: str,
                 StartTS: float,
                 EndTS: float,
                 RM: np.ndarray,
                 SWITCH_XY: bool) -> pd.DataFrame:
    """
    Merge all player files within a phase folder into a single team-level tracking DataFrame.
    """
    file_list = build_player_file_list(file_dir)
    TeamPosition = pd.DataFrame()

    for file in file_list:
        path = os.path.join(file_dir, file)
        position = read_player_latlon_csv(path)

        StartIndex, EndIndex = find_indices_for_window(position, StartTS, EndTS)
        position = position.iloc[StartIndex:EndIndex + 1, :]

        if len(position["Time"].unique()) != len(position):
            position = (
                position.sort_values("Time")
                        .drop_duplicates(subset="Time", keep="first")
                        .reset_index(drop=True)
            )

        rot = convert_and_rotate_player(position, RM, SWITCH_XY)

        playername = file[0:7]
        rot.columns = ["Time", f"{playername}_x", f"{playername}_y"]

        if TeamPosition.empty:
            TeamPosition = rot
        else:
            TeamPosition = pd.merge(
                TeamPosition, rot,
                on="Time", how="outer",
                validate="one_to_one"
            )

    TeamPosition.sort_values("Time", inplace=True)
    TeamPosition.reset_index(drop=True, inplace=True)
    return TeamPosition


## Find root folders

# FUNCTION 15: Identify valid Root_folders

def iter_root_folders(master_folder: str) -> list[str]:
    """
    Treat each immediate subfolder of MASTER_FOLDER as a Root_folder if it contains:
      - Pitch_coordinates.xlsx
      - GPS_Data_new
    """
    roots = []
    for name in sorted(os.listdir(master_folder)):
        p = os.path.join(master_folder, name)
        if not os.path.isdir(p):
            continue

        has_pitch = os.path.isfile(os.path.join(p, "Pitch_coordinates.xlsx"))
        has_gps_new = os.path.isdir(os.path.join(p, MATCH))

        if has_pitch and has_gps_new:
            roots.append(p)

    return roots


## Run one root folder

# FUNCTION 16: Process a single Root_folder

def process_root_folder(Root_folder: str):
    """
    Run the full pitch-normalisation and team-tracking pipeline for a single game folder.
    """
    print("\n" + "=" * 70)
    print(f"Processing Root_folder: {Root_folder}")

    pitch_file = os.path.join(Root_folder, "Pitch_coordinates.xlsx")
    df_pitch = pd.read_excel(pitch_file)

    pts = pitch_latlon_to_xy(df_pitch)
    loop = order_corners_ccw(pts)

    pitch_length, pitch_width, _ = compute_pitch_geometry(loop)

    p0, p1 = pick_origin_and_long_edge(loop)
    RM = rotation_matrix_to_align_length_with_x(p0, p1)

    loop_rot = apply_rotation_rowvec(loop, RM)

    x_range = float(loop_rot[:, 0].max() - loop_rot[:, 0].min())
    y_range = float(loop_rot[:, 1].max() - loop_rot[:, 1].min())
    SWITCH_XY = x_range < y_range

    if SWITCH_XY:
        loop_rot = loop_rot[:, [1, 0]]
        x_range, y_range = y_range, x_range

    Opitch_x = float(loop_rot[:, 0].min())
    Opitch_y = float(loop_rot[:, 1].min())

    Pit = pd.DataFrame({
        "pitch_length": [pitch_length],
        "pitch_width": [pitch_width],
        "Opitch_x": [Opitch_x],
        "Opitch_y": [Opitch_y],
        "SWITCH_XY": [int(SWITCH_XY)]
    })

    output_dir = os.path.join(Root_folder, "Output")
    os.makedirs(output_dir, exist_ok=True)

    pitch_output_file = os.path.join(output_dir, "Pitch_Lengths.csv")
    Pit.to_csv(pitch_output_file, index=False)

    print("Pitch meta saved:")
    print(Pit)

    for folder in folders:
        phase_dir = os.path.join(Root_folder, MATCH, folder)
        match_xlsx = os.path.join(phase_dir, "Match_data.xlsx")

        if not os.path.isdir(phase_dir):
            print(f"  Skipping phase (missing folder): {phase_dir}")
            continue
        if not os.path.isfile(match_xlsx):
            print(f"  Skipping phase (missing Match_data.xlsx): {match_xlsx}")
            continue

        match_info = read_match_data(match_xlsx)

        playernum = 41
        StartTS = match_info.loc[0:playernum - 1, 'Split Start Time'].max()
        EndTS = match_info.loc[0:playernum - 1, 'Split End Time'].min()

        TeamRot = TeamTracking(phase_dir, StartTS, EndTS, RM, SWITCH_XY)

        out_path = os.path.join(output_dir, f"{folder}_{MATCH}_10Hz.csv")
        TeamRot.to_csv(out_path, index=False, na_rep="NA")
        print(f"Saved: {out_path}")


# MAIN WORKFLOW #

def main():
    if not os.path.isdir(MASTER_FOLDER):
        raise FileNotFoundError(f"MASTER_FOLDER does not exist: {MASTER_FOLDER}")

    roots = iter_root_folders(MASTER_FOLDER)
    if not roots:
        print(
            f"No valid Root_folders found inside:\n  {MASTER_FOLDER}\n"
            f"Expected each to contain Pitch_coordinates.xlsx and {MATCH}/"
        )
        return

    print(f"Found {len(roots)} Root_folder(s).")
    for rf in roots:
        try:
            process_root_folder(rf)
        except Exception as e:
            print(f"ERROR processing {rf}: {e}")

    print("\nAll folders processed.")


if __name__ == "__main__":
    main()
