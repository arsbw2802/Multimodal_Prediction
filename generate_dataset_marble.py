"""
Use this script to generate a Marble dataset files for further processing. 
This script is a modified version of the marble_to_casas_adapter.py, 
    and is used to convert the marble dataset layout to match that of this dataset:
    [Blunck, H., Bhattacharya, S., Prentow, T., Kjrgaard, M., & Dey, A. (2015). Heterogeneity Activity Recognition [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5689X. ]

This is so that marble data, mainly IMU info, can be used for Mehga's supervised HAR model.

This script as is, generates 2 files (with no extension, but formatted just like a CSV):
    - acc_data: IMU sensor data.
    - labels: activity labels that correspond to timestamps to label the sensor data.
"""

# Meta-data.
__author__ = 'Abdullah Altaweel'
__copyright__ = ''
__credits__ = []
__license__ = ''
__version__ = '1.0'
__maintainer__ = ''
__email__ = 'aaltaweel3@gatech.edu'
__status__ = ''

# Dependencies.
from datetime import datetime, time, timedelta
import string
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


parser = argparse.ArgumentParser(description='This script is for converting the MARBLE dataset into a CASAS-like format so the TDOST scripts work almost out of the box.')


subdirs_map = []

def map_folder_structure(folder_path) -> dict:
  """
  Recursively maps the folder structure, returning a dictionary where keys are folder names and values are lists of subdirectory names.

  Args:
    folder_path: The path to the root folder.

  Returns:
    A dictionary representing the folder structure.
  """

  folder_structure = {}

  for root, dirs, files in os.walk(folder_path):
    folder_name = os.path.basename(root)
    folder_structure[folder_name] = dirs

  return folder_structure

def get_single_inhabitant(scenarios:list) -> list:
    res = []
    for scenario in scenarios:
        scenario:str
        if "1" in scenario:
            res.append(scenario)
    return res

def get_subject(path_to_curr_instance:str) -> str:
    contents = os.listdir(path_to_curr_instance)
    for item in contents:
        if "subject" in item:
            return item

def inject_synthetic_events(df_events:pd.DataFrame, df_locations:pd.DataFrame, synth_freq=100,):
    """Injects synthetic motion sensor events to df_events based on the locations provided in df_locations, for every `freq` seconds.

    Args:
        df_events (pd.DataFrame): dataframe of event triggers
        path_to_locations_csv (str): dataframe of location labels
        freq (int, optional): the frequency of synthetic event injections, in seconds. Defaults to 100.
    """
    noise_std = synth_freq/2.0

    def location_to_int(location: str) -> int:
        alphabet = string.ascii_uppercase
        # Convert each letter to its position in the alphabet and concatenate
        return int(''.join(str(alphabet.index(char.upper()) + 1) for char in location if char.isalpha()))
    
    # Prepare arrays to store synthetic data
    sensor_ids = []
    sensor_statuses = []
    timestamps = []
    subject_ids = []

    # Extract subject_id from the events DataFrame (assuming it's the same for all rows)
    subject_id = df_events['subject_id'].iloc[0]

    sensor_id_to_location_mapping = {}  # used to create further mappings for later scenarios
    for _, row in df_locations.iterrows():
        # Calculate the start and end timestamps for the location interval
        ts_start = row['ts_start']
        ts_end = row['ts_end']
        location: str = row['location']

        location_int = location_to_int(location)

        # Generate synthetic timestamps by adding noise to the average frequency
        synthetic_ts = []
        current_ts = ts_start
        while current_ts < ts_end:
            # Add random noise to the interval using a normal distribution
            noisy_interval = max(1, np.random.normal(synth_freq, noise_std))
            # Convert noisy_interval to microseconds and add it to the current timestamp
            current_ts += pd.Timedelta(microseconds=int(noisy_interval * 1_000_000))  # Convert seconds to microseconds
            if current_ts < ts_end:
                synthetic_ts.append(current_ts)

        # Generate "ON" and "OFF" events using broadcasting
        num_synthetic_events = len(synthetic_ts)
        sensor_ids.extend([f"M{location_int}"] * num_synthetic_events * 2)  # ON and OFF events
        sensor_id_to_location_mapping[f"M{location_int}"] = location
        sensor_statuses.extend(["ON", "OFF"] * num_synthetic_events)
        timestamps.extend(np.repeat(synthetic_ts, 2) + np.tile([pd.Timedelta(microseconds=0), pd.Timedelta(milliseconds=500)], num_synthetic_events))
        subject_ids.extend([subject_id] * num_synthetic_events * 2)

    # Create a DataFrame from the generated synthetic events
    df_synthetic = pd.DataFrame({
        'sensor_id': sensor_ids,
        'sensor_status': sensor_statuses,
        'ts': timestamps,
        'subject_id': subject_ids
    })

    # Concatenate the synthetic events with the original event data
    df_combined = pd.concat([df_events, df_synthetic], ignore_index=True)

    # Sort by timestamp (ts) to maintain chronological order
    df_combined = df_combined.sort_values(by='ts').reset_index(drop=True)

    fancy_print_mappings(sensor_id_to_location_mapping)
    # print(f'finished synthing a df')
    
    return df_combined

def fancy_print_mappings(sensor_mapping: dict):
    # print("sensor_mapping = {")
    for sensor_id, location in sensor_mapping.items():
        # Convert sensor_id to a tuple format, even if it's a single value
        sensor_tuple = f'("{sensor_id}",)'
        print(f"    {sensor_tuple}: \"in {location.lower()}\",")
    # print("}")


def read_scenario_csvs(scenario: str, file_structure: dict, synth_freq=0, convert_sensor_ids=True, inject_IMU=True) -> list[pd.DataFrame]:
    """Reads the environmental sensor file for a specified scenario and converts it to the CSV format.
    Args:
        scenario (str): the scenario to read the env sensors for
        file_structure (dict): the map of the MARBLE dataset file structure
    Returns:
        list[pd.DataFrame]: the converted CSVs in pd.DataFrame format.
    """  
    instances = get_scenario_instances(scenario, file_structure)
    
    res, lbs = [], []
    for instance in instances:
        dfs, subject = load_instance_files(scenario, instance)
        df_events, df_labels, df_locations = dfs[:3] # env sensors
        df_acc, df_baro, df_gyro, df_mag = dfs[3:]   # IMU sensors


        if inject_IMU:
            df_events = pd.DataFrame() # NOTE: THIS LINE CLEARS ALL ENV SENSORS
            print("Injecting wearable IMU data to output dataset(s)")
            df_events = inject_IMU_readings(df_events, [df_acc]) #! for testing
            # df_events = inject_IMU_readings(df_events, [df_acc, df_gyro, df_mag], to_text=False)
            # df_events = inject_IMU_readings(df_events, [df_acc, , df_gyro, df_mag, df_baro]) #TODO: make this work on 1D data (df_baro)
        
        process_time_start_end(df_labels)
        process_time_start_end(df_locations)
        process_files([df_events, df_acc])

        df_labels_filtered = df_labels
        if args.drop_transitions:
            df_labels_filtered = filter_labels(df_labels) 
        
        if synth_freq > 0:
            df_events = inject_synthetic_events(df_events, df_locations, synth_freq)
        
        df_events_annotated = df_events
        df_events_annotated['user'] = subject
        lbs.append(df_labels_filtered)
        res.append(df_events_annotated)
    return res, lbs


def get_scenario_instances(scenario: str, file_structure: dict) -> list[str]:
    """Retrieve instances for the given scenario."""
    instances = file_structure.get(scenario)
    if not instances:
        raise FileNotFoundError(f"Scenario {scenario} not found in the file structure.")
    return instances

def load_instance_files(scenario: str, instance: str) -> tuple:
    """Load file paths for a given instance and return the corresponding dataframes."""
    paths = get_file_paths(scenario, instance)
    subject = paths[-1]
    paths = paths[:-1]
    dfs = [pd.read_csv(path) for path in paths]
    return dfs, subject


def process_files(dfs: list[pd.DataFrame]) -> None:
    """Process datetime information for a list of dataframes."""
    for df in dfs:
        process_time(df)

def filter_labels(df_labels: pd.DataFrame) -> pd.DataFrame:
    """Filter out transitions from the labels dataframe."""
    return drop_transitions(df_labels)

def inject_IMU_readings(df_events:pd.DataFrame, df_imu:list[pd.DataFrame], to_text=False):
    df:pd.DataFrame = pd.concat([df_events, *df_imu], ignore_index=False)
    df = df.sort_values(by='ts', ascending=True)
    return df

def save(file_name:str, df_events_annotated:pd.DataFrame):
    converted_dataset_path = os.path.join(args.save_path, file_name) 
    df_events_annotated.to_csv(converted_dataset_path, header=True, index=False) 
    print("Saved ", converted_dataset_path, "\n")

def drop_transitions(df_labels):
    # remove TRANSITION labels
    df_labels_filtered = df_labels[df_labels['act'] != 'TRANSITION']
    df_labels_filtered=df_labels_filtered.reset_index(drop=True)
    return df_labels_filtered

def process_time(df):
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')

def process_time_start_end(df):
    df['ts_start'] = pd.to_datetime(df['ts_start'], unit='ms')
    df['ts_end'] = pd.to_datetime(df['ts_end'], unit='ms')

def get_file_paths(scenario, instance):

    path_to_curr_instance   = os.path.join(args.dataset_folder_dir, scenario, instance)
    path_to_env_csv         = os.path.join(path_to_curr_instance, "environmental.csv")
    subject                 = get_subject(path_to_curr_instance)
    path_to_locations_csv   = os.path.join(path_to_curr_instance, subject ,"locations.csv")
    path_to_labels_csv      = os.path.join(path_to_curr_instance, subject, "labels.csv")
    
    # IMUs
    path_to_acc_csv         = os.path.join(path_to_curr_instance, subject, "accelerometer.csv")
    path_to_baro_csv         = os.path.join(path_to_curr_instance, subject, "barometer.csv")
    path_to_gyro_csv         = os.path.join(path_to_curr_instance, subject, "gyroscope.csv")
    path_to_mag_csv         = os.path.join(path_to_curr_instance, subject, "magnetometer.csv")

    return path_to_env_csv,path_to_labels_csv, path_to_locations_csv, path_to_acc_csv,  path_to_baro_csv, path_to_gyro_csv, path_to_mag_csv, subject


def label_cols(df_events:pd.DataFrame, df_labels:pd.DataFrame):
    df_events['activity'] = ''

    event_index = 0

    while event_index < len(df_events):
        event_row   = df_events.iloc[event_index]  # Access the row using iloc
        event_time  = event_row['ts']

        
        for i, label_row in df_labels.iterrows():
            label_start_time = label_row ['ts_start']
            label_end_time   = label_row ['ts_end']
            label_name       = label_row ['act'].capitalize()
            
        
            if label_start_time <= event_time <= label_end_time:
                df_events.at[event_index, 'activity'] = f"{label_name}"
            
        event_index += 1
    # print(f'finished labelling a df')

def combine_dfs(dfs_list)-> pd.DataFrame:
    # Combine all DataFrames in the list without re-indexing
    combined_df = pd.concat(dfs_list, ignore_index=False)
    
    return combined_df

def clear_old_files(folder_path):
    """
    Deletes all files from the directory but keeps the folders.
    """
    print("Clearing from existing files (if they exist) and saving new files")

    # Check if the folder exists
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Check if it is a file
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    else:
        print(f"Directory {folder_path} does not exist.")

def main(info_to_view=None, synth_freq=0, in_dataset_folder_dir=None, use_imu=False):
    """Runs this file. 
    Args:
        info_to_view (int, optional): which instance to view. Defaults to None.
        save_to_disk (bool, optional): Determines if the data is saved to disk. Defaults to False.
    Returns:
        env_sensor[`info_to_view`], `labelled[info_to_view]`: a tuple of the env_sensor and labels DataFrames, for debugging.
    """    
    dataset_folder_dir = in_dataset_folder_dir

    struct = map_folder_structure(dataset_folder_dir)
    scenarios = struct.get("dataset")
    single_inhabitant_scenarios = get_single_inhabitant(scenarios)

    # Now we can run on the scenario corresponding to info_to_view
    env_sensors, labelled = read_scenario_csvs(
        scenario=single_inhabitant_scenarios[info_to_view],  # Run on the current scenario
        file_structure=struct, 
        synth_freq=synth_freq, 
        inject_IMU=use_imu
        )

    return env_sensors, labelled

if __name__ == '__main__':
    parser.add_argument('--dataset_folder_dir', type=str,  help="path to the ./MARBLE/dataset",                                             default="C:/Users/LENOVO/Desktop/TDOST_AICaring/dataset/MARBLE/dataset")
    parser.add_argument('--save_to_disk',       type=bool, help="saves the marble dataset to the disk in CASAS format (no file extension)", default=True)
    parser.add_argument('--synth_freq',         type=int,  help="freqency of synthetic event triggers, in seconds; 0: no synthetic events injected (less = more synthetic events)",default=0)
    parser.add_argument('--test_mode',          type=bool, help="processes only one instance, used for dev purposes",                       default=False)
    parser.add_argument('--combine_dataset',    type=bool, help="combines the dataset to one file",                                         default=False)
    parser.add_argument('--use_imu',            type=bool, help="ONLY uses wearable IMU data in the dataset",                               default=False)
    parser.add_argument('--save_path',          type=str, help="Where to save files",                                                       default="None")
    parser.add_argument('--drop_transitions',   type=bool, help="removes the TRANSITIONS labels from the saved label file",                 default=False)

    args = parser.parse_args()

    if not os.path.exists(args.dataset_folder_dir):
        raise FileNotFoundError(f"path {args.dataset_folder_dir} does not exist.")
    
    if not os.path.exists(args.save_path):
        raise FileNotFoundError(f"path {args.save_path} does not exist.")
    
    sensors = []
    labels = []

    if args.test_mode:
        print("NOT Processing entire MARBLE dataset; in test mode:")

        output = main(
            info_to_view=           0, 
            synth_freq=             args.synth_freq,
            in_dataset_folder_dir=  args.dataset_folder_dir,
            use_imu= args.use_imu
            )
                
        for df in output[0]:
            sensors.append(df)

        for df in output[1]:
            labels.append(df)
    
    else:
        print("Processing entire MARBLE dataset:")
        output = []
        # Get scenarios for parallel processing
        struct = map_folder_structure(args.dataset_folder_dir)
        scenarios = struct.get("dataset")
        single_inhabitant_scenarios = get_single_inhabitant(scenarios)

        # Use ThreadPoolExecutor to run each scenario in parallel
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(main, 
                                info_to_view=i,  
                                synth_freq=args.synth_freq, 
                                in_dataset_folder_dir=args.dataset_folder_dir,
                                use_imu= args.use_imu
                                )

                for i in range(len(single_inhabitant_scenarios))
            ]
            
            # Optionally collect and print the results
            for future in as_completed(futures):
                output = future.result()
                
                for df in output[0]:
                    sensors.append(df)

                for df in output[1]:
                    labels.append(df)

    if args.combine_dataset:
        sensors = [combine_dfs(sensors)]
        labels = [combine_dfs(labels)]
    
    if args.save_to_disk:
        # clear_old_files(args.save_path)

        for file in sensors:
            filename = f"acc_data"
            save(filename, file)

        for file in labels:
            filename = f"labels"
            save(filename, file)