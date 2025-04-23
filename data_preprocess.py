import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='arguments for MARBLE dataset preprocessing')
    parser.add_argument("--marble_dataset_path", type=str, default="./dataset/MARBLE/dataset", help="Location of the MARBLE dataset")
    parser.add_argument('--sampling_rate', type=int, default="100", help="Sampling rate for the data. Is used to downsample to the required rate")

    args = parser.parse_args()

    return args

# List of all possible environmental sensors (17 total)
ALL_ENV_SENSORS = [
    "R1", "R2", "R5", "R6", "R7",
    "E1", "E2",
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"
]

def generate_sensor_summary(row):
    context_map = {
        "R1": "using pantry", "R2": "using cutlery drawer", "R5": "using pots drawer",
        "R6": "using medicines cabinet", "R7": "using fridge",
        "E1": "using stove plug", "E2": "using television plug",
        "P1": "on dining room chair", "P2": "on office chair", "P3": "on living room couch",
        "P4": "on dining room chair", "P5": "on dining room chair", "P6": "on dining room chair",
        "P7": "on living room couch", "P8": "on living room couch", "P9": "on living room couch"
    }

    summary_parts = []

    for sensor_id in ALL_ENV_SENSORS:
        col = f'env_{sensor_id}'
        if col not in row or row[col] == 'N/A' or row[col] == "":
            continue
        context = context_map.get(sensor_id)
        if not context:
            continue
        value = row[col]
        summary_parts.append(f"Motion sensor {context} with value {value}.")

    return ' '.join(summary_parts)


def process_all_instances(args):
    dataset_path = Path(args.marble_dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    all_dataframes = []

    scenario_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    for scenario_dir in tqdm(scenario_dirs, desc="Processing scenarios"):
        if not scenario_dir.is_dir():
            continue
        scenario_id = scenario_dir.name

        for instance_dir in scenario_dir.iterdir():
            if not instance_dir.is_dir():
                continue
            instance_id = instance_dir.name
            env_path = instance_dir / "environmental.csv"
            env_df = pd.read_csv(env_path) if env_path.exists() else pd.DataFrame()

            for subject_dir in instance_dir.iterdir():
                if not subject_dir.is_dir():
                    continue
                subject_id = subject_dir.name

                # Get min/max timestamp across all available files
                ts_min, ts_max = float("inf"), 0

                def update_bounds(file_path):
                    nonlocal ts_min, ts_max
                    if file_path.exists() and file_path.stat().st_size > 0:
                        try:
                            df = pd.read_csv(file_path)
                            if df.empty or df.shape[1] == 0:
                                return  # skip empty or header-less CSV
                            if 'ts' in df.columns:
                                ts_min = min(ts_min, df['ts'].min())
                                ts_max = max(ts_max, df['ts'].max())
                            elif 'ts_start' in df.columns:
                                ts_min = min(ts_min, df['ts_start'].min())
                                ts_max = max(ts_max, df['ts_end'].max())
                        except pd.errors.EmptyDataError:
                            print(f"Skipped empty file: {file_path}")
                            return  # skip completely empty files
                        
                # Update time bounds
                files = ['accelerometer.csv', 'magnetometer.csv', 'gyroscope.csv', 'barometer.csv',
                         'locations.csv', 'labels.csv']
                for file in files:
                    update_bounds(subject_dir / file)
                if not np.isfinite(ts_min) or not np.isfinite(ts_max):
                    continue

                # Create master timeline
                timeline = pd.DataFrame({'ts': np.arange(ts_min, ts_max + 1, args.sampling_rate)})

                # Load continuous sensor data
                for sensor in ['accelerometer', 'magnetometer', 'gyroscope', 'barometer']:
                    f = subject_dir / f"{sensor}.csv"
                    if f.exists():
                        df = pd.read_csv(f).dropna()
                        df['ts'] = df['ts'].astype(np.int64)
                        df = df.groupby('ts').mean().reset_index()
                        df_interp = df.set_index('ts').reindex(timeline['ts'], method='nearest').reset_index()
                        df_interp = df_interp.drop(columns='ts').add_prefix(f'{sensor}_')
                        timeline = pd.concat([timeline, df_interp], axis=1)

                # Load discrete interval-based data
                for file in ['labels.csv', 'locations.csv']:
                    f = subject_dir / file
                    col = file.replace('.csv', '')
                    timeline[col] = 'UNKNOWN'
                    if f.exists():
                        df = pd.read_csv(f)
                        if 'ts_start' in df.columns:
                            for _, row in df.iterrows():
                                timeline.loc[
                                    (timeline['ts'] >= row['ts_start']) & (timeline['ts'] <= row['ts_end']),
                                    col
                                ] = row.iloc[2]
                        else:
                            for _, row in df.iterrows():
                                idx = timeline['ts'].sub(row['ts']).abs().idxmin()
                                timeline.at[idx, col] = f"{row.iloc[0]}_{row.iloc[1]}"

                # Identify all sensors used in this instance's environmental.csv
                scenario_env_sensors = set(env_df['sensor_id'].unique()) if not env_df.empty else set()

                # Set OFF or N/A appropriately
                for sensor_id in ALL_ENV_SENSORS:
                    col_name = f'env_{sensor_id}'
                    if sensor_id in scenario_env_sensors:
                        timeline[col_name] = 'OFF'
                    else:
                        timeline[col_name] = 'N/A'

                # Filter environmental data by this subject
                subject_env = env_df[env_df['subject_id'].astype(str) == str(subject_id)]

                if not subject_env.empty:
                    subject_env['ts'] = subject_env['ts'].astype(np.int64)

                    for sensor_id in subject_env['sensor_id'].unique():
                        sensor_df = subject_env[subject_env['sensor_id'] == sensor_id].sort_values('ts').reset_index(drop=True)
                        col_name = f'env_{sensor_id}'

                        i = 0
                        while i < len(sensor_df):
                            row = sensor_df.iloc[i]
                            if row['sensor_status'] == 'ON':
                                ts_on = row['ts']
                                # Look for corresponding OFF event
                                ts_off = timeline['ts'].max()
                                for j in range(i+1, len(sensor_df)):
                                    if sensor_df.iloc[j]['sensor_status'] == 'OFF':
                                        ts_off = sensor_df.iloc[j]['ts']
                                        break
                                # Set ON in the timeline between ts_on and ts_off
                                timeline.loc[(timeline['ts'] >= ts_on) & (timeline['ts'] <= ts_off), col_name] = 'ON'
                                i = j + 1  # Continue after OFF
                            else:
                                i += 1

                # Add metadata
                timeline['scenario_id'] = scenario_id
                timeline['instance_id'] = instance_id
                timeline['subject_id'] = subject_id
                timeline['sensor_status_summary'] = timeline.apply(generate_sensor_summary, axis=1)
                all_dataframes.append(timeline)

    # Combine all into one DataFrame
    if all_dataframes:
        result_df = pd.concat(all_dataframes, ignore_index=True)
    else:
        result_df = pd.DataFrame()

    # Remove rows with 'TRANSITION' or 'UNKNOWN' label
    result_df = result_df[result_df['labels'] != "TRANSITION"]
    result_df = result_df[result_df['labels'] != "UNKNOWN"]
    return result_df




if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    df = process_all_instances(args)

    output_dir = Path("all_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "synced_marble_data.csv"
    df.to_csv(file_path, index=False)

    print(f"✅ File successfully saved to: {file_path.resolve()}")
