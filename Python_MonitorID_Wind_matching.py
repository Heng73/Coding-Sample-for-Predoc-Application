import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

# -------------------- Paths --------------------
china_monitor_file     = '/Users/tangheng/Dropbox/Green TFP China/RA_Heng_work/Data/Monitor/Workdata/monitor_List(china)_Opendate_data.csv'
monitor_file           = '/Users/tangheng/Dropbox/Green TFP China/RA_Heng_work/Data/Monitor/monitor_station_list_withOpendate.csv'
station_file           = '/Users/tangheng/Dropbox/Green TFP China/Data/Workdata/Workdata for wind process/processed_monitor_wind_data.csv'
station_filtered_file  = '/Users/tangheng/Dropbox/Green TFP China/RA_Heng_work/Data/Monitor/Workdata/processed_monitor_wind_data_filtered.csv'
output_file            = '/Users/tangheng/Dropbox/Green TFP China/RA_Heng_work/Data/Monitor/monitor_List_Opendate_Wind_data.csv'
output2_file           = '/Users/tangheng/Dropbox/Green TFP China/RA_Heng_work/Data/Monitor/monitor_List(china)_Opendate_Wind_data.csv'

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
