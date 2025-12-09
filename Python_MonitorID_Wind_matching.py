# ==================================================
# Purpose of this script
# --------------------------------------------------
# This script links air-quality monitors to the nearest
# wind-measurement stations using great-circle distances.
#
# why we do this:
#   - In a separate step of the project, we have already matched
#     firms to air-quality monitors based on geographic distance.
#   - For water emissions, "upstream" and "downstream" are relatively
#     well-defined along the river network. For air emissions, however,
#     the direction of transport is much less obvious because wind
#     varies over time and across locations.
#   - To capture this dimension, we match each air-quality monitor to
#     its nearest wind station and use the prevailing wind direction
#     (e.g., the most frequent direction over a year) at that station
#     as a proxy for how pollution is likely transported around the monitor.
#   - These wind-based measures will be used in robustness checks.
#
# Output:
#   1. monitor_List_Opendate_Wind_data.csv
#        - full monitor list matched to nearest wind station
#   2. monitor_List(china)_Opendate_Wind_data.csv
#        - another documented monitor list matched to nearest wind station
# ==================================================

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

# -------------------- Paths --------------------
THIS_FILE = Path(__file__).resolve()
# /Users/tangheng/Dropbox/Green TFP China/RA_Heng_work/Code/this_script.py
PROJECT_ROOT = THIS_FILE.parents[2]
# /Users/tangheng/Dropbox/Green TFP China

DATA_DIR          = PROJECT_ROOT / "Data"
RA_DIR            = PROJECT_ROOT / "RA_Heng_work"
MONITOR_DIR       = RA_DIR / "Data" / "Monitor"
MONITOR_WORK_DIR  = MONITOR_DIR / "Workdata"
WIND_DATA_DIR     = DATA_DIR / "Workdata" / "Workdata for wind process"

china_monitor_file    = MONITOR_WORK_DIR / "monitor_List(china)_Opendate_data.csv"
monitor_file          = MONITOR_DIR      / "monitor_station_list_withOpendate.csv"
station_file          = WIND_DATA_DIR    / "processed_monitor_wind_data.csv"
station_filtered_file = MONITOR_WORK_DIR / "processed_monitor_wind_data_filtered.csv"
output_file           = MONITOR_DIR      / "monitor_List_Opendate_Wind_data.csv"
output2_file          = MONITOR_DIR      / "monitor_List(china)_Opendate_Wind_data.csv"

EARTH_RADIUS_KM = 6371.0

# -------------------- Step 1: filter station rows --------------------
station_all = pd.read_csv(station_file)

# Keep the row with the maximum days_count for each station
idx = station_all.groupby('STATION', as_index=False)['days_count'].idxmax()
station = station_all.loc[idx].reset_index(drop=True)

# Drop rows without coordinates
station = station.dropna(subset=['LATITUDE', 'LONGITUDE'])

station.to_csv(station_filtered_file, index=False)
print(f'Saved filtered station file: {station_filtered_file}, total {len(station)} rows')

# -------------------- Nearest-station function --------------------
def find_nearest_station(monitor_df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each monitor in `monitor_df`, find the nearest station in `station_df`
    using haversine distance via BallTree, then merge station attributes.
    Returns a merged DataFrame with a 'distance_monitor_station' column (km).
    """
    # Drop monitors without coordinates
    md = monitor_df.dropna(subset=['latitude', 'longitude']).copy()

    # Convert to radians
    mon_rad = np.radians(md[['latitude', 'longitude']].to_numpy(copy=False))
    stn_rad = np.radians(station_df[['LATITUDE', 'LONGITUDE']].to_numpy(copy=False))

    # Build tree (metric=haversine expects radians)
    tree = BallTree(stn_rad, metric='haversine')
    dists, inds = tree.query(mon_rad, k=1)
    d_km = (dists.flatten() * EARTH_RADIUS_KM).round(2)

    # Attach nearest station id + distance
    md['nearest_station'] = station_df.iloc[inds.flatten()]['STATION'].values
    md['distance_monitor_station'] = d_km

    # Merge station columns (keep one row per monitor)
    merged = md.merge(
        station_df,
        left_on='nearest_station',
        right_on='STATION',
        how='left',
        suffixes=('_mon', '_stn')
    )

    # Drop unneeded columns if present
    drop_cols = ['LATITUDE', 'LONGITUDE', 'referencestation', 'nearest_station', 'consistency_check']
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    return merged

# -------------------- Helper: run match + filter + save --------------------
def match_and_save(monitor_path: str, station_df: pd.DataFrame, out_path: str, openyear_cutoff: int = 2016) -> None:
    monitor_df = pd.read_csv(monitor_path)
    matched = find_nearest_station(monitor_df, station_df)

    # Keep observations with openyear < cutoff (if column exists)
    if 'openyear' in matched.columns:
        matched = matched[matched['openyear'] < openyear_cutoff].reset_index(drop=True)

    matched.to_csv(out_path, index=False)
    print(f'Matched results saved: {out_path} (rows: {len(matched)})')

# -------------------- Step 2 & 3: apply to both monitor lists --------------------
match_and_save(monitor_file, station, output_file, openyear_cutoff=2016)
match_and_save(china_monitor_file, station, output2_file, openyear_cutoff=2016)
