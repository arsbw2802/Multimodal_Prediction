import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


# Root of dataset
DATASET_DIRECTORY = Path("dataset")

def create_subject_dataframe(
    instance_dir: str, subject_id: int, sensor_data: pd.DataFrame
) -> pd.DataFrame:
    base_path = f"{instance_dir}/subject-{subject_id}"

    # Add accelerometer to dataframe
    accelerometer_data = pd.read_csv(f"{base_path}/accelerometer.csv")
    accelerometer_data.columns = [
        f"accel_{j}" if j != "ts" else j for j in accelerometer_data.columns.values
    ]
    accelerometer_data = accelerometer_data.pivot(index="ts", columns=[])
    accelerometer_data = accelerometer_data[
        ~accelerometer_data.index.duplicated(keep="first")
    ]
    sensor_data = sensor_data.merge(accelerometer_data, how="outer", on="ts")

    # Add barometer to dataframe
    barometer_data = pd.read_csv(f"{base_path}/barometer.csv")
    barometer_data = barometer_data.pivot(index="ts", columns=[])
    barometer_data = barometer_data[~barometer_data.index.duplicated(keep="first")]
    sensor_data = sensor_data.merge(barometer_data, how="outer", on="ts")

    # Add gyroscope to dataframe
    gyroscope_data = pd.read_csv(f"{base_path}/gyroscope.csv")
    gyroscope_data.columns = [
        f"gyro_{j}" if j != "ts" else j for j in gyroscope_data.columns.values
    ]
    gyroscope_data = gyroscope_data.pivot(index="ts", columns=[])
    gyroscope_data = gyroscope_data[~gyroscope_data.index.duplicated(keep="first")]
    sensor_data = sensor_data.merge(gyroscope_data, how="outer", on="ts")

    # Add magnetometer to dataframe
    magnetometer_data = pd.read_csv(f"{base_path}/magnetometer.csv")
    magnetometer_data.columns = [
        f"mag_{j}" if j != "ts" else j for j in magnetometer_data.columns.values
    ]
    magnetometer_data = magnetometer_data.pivot(index="ts", columns=[])
    magnetometer_data = magnetometer_data[
        ~magnetometer_data.index.duplicated(keep="first")
    ]
    sensor_data = sensor_data.merge(magnetometer_data, how="outer", on="ts")

    # Add locations to dataframe
    location_data = pd.read_csv(f"{base_path}/locations.csv")
    location_data = location_data.melt(
        id_vars="location", value_vars=["ts_start", "ts_end"], value_name="ts"
    ).drop("variable", axis=1)
    sensor_data = sensor_data.merge(location_data, how="outer", on="ts")

    # Add labels to dataframe
    labels_data = pd.read_csv(f"{base_path}/labels.csv")
    labels_data = labels_data.melt(
        id_vars="act", value_vars=["ts_start", "ts_end"], value_name="ts"
    ).drop("variable", axis=1)
    sensor_data = sensor_data.merge(labels_data, how="outer", on="ts")

    # Add smartphone data to dataframe
    try:
        smartphone_data = pd.read_csv(f"{base_path}/smartphone.csv")
        smartphone_data = smartphone_data.pivot(index="ts", columns="event_id")
        smartphone_data.columns = [j for (_, j) in smartphone_data.columns.values]
        sensor_data = sensor_data.merge(smartphone_data, how="outer", on="ts")
    except pd.errors.EmptyDataError as e:
        ...

    # ffill works as events should keep on occurring until state changes
    sensor_data.ffill(inplace=True)

    # For state objects, fill in "OFF" as default state
    fill_dict = {
        col: "OFF"
        for col in sensor_data
        if col
        not in [
            "accel_x",
            "accel_y",
            "accel_z",
            "value",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "mag_x",
            "mag_y",
            "mag_z",
            "location",
            "act",
            "ts",
        ]
    }

    sensor_data.fillna(fill_dict, inplace=True)

    # Finally, do a bfill to take care of numerical values
    sensor_data.bfill(inplace=True)

    return sensor_data


def create_instance_dataframe(scenario_dir: str, instance_id: str) -> pd.DataFrame:
    base_dir = f"{scenario_dir}/{instance_id}"

    # Create sensor_data
    sensor_data = pd.read_csv(f"{base_dir}/environmental.csv")
    sensor_ids = sensor_data["sensor_id"].unique()

    dfs = []

    # Create and join frames based on subject id
    subject_ids = sensor_data["subject_id"].unique()
    for id in subject_ids:
        subject_frame = sensor_data[sensor_data["subject_id"] == id].drop(
            "subject_id", axis=1
        )
        subject_frame = subject_frame.pivot(index="ts", columns="sensor_id")
        subject_frame.columns = [j for (_, j) in subject_frame.columns.values]

        subject_frame = create_subject_dataframe(base_dir, int(id), subject_frame)
        subject_frame.insert(1, "subject_id", id)
        subject_frame.set_index(["ts", "subject_id"])

        # Reinsert all columns not present in this dataframe
        for sensor_id in sensor_ids:
            if sensor_id not in subject_frame.columns:
                subject_frame[sensor_id] = np.nan

        dfs.append(subject_frame)

    # Concat instance dataframes
    instance_df = pd.concat(dfs)
    instance_df.insert(0, "instance", instance_id)

    return instance_df


def create_scenario_dataframe(root_dir: str, scenario_id: str) -> pd.DataFrame:
    base_path = Path(f"{root_dir}/{scenario_id}")
    instance_ids = [d.name for d in base_path.iterdir() if d.is_dir()]

    dfs: list[pd.DataFrame] = []
    columns = set()
    for id in instance_ids:
        # Create instance dataframe
        instance_dataframe = create_instance_dataframe(base_path, id)

        # Store column names
        columns.update(instance_dataframe.columns)

        dfs.append(instance_dataframe)

    # Ensure all dataframes have all columns
    for column in columns:
        for df in dfs:
            if column not in df.columns:
                df[column] = np.nan

    # Concat scenario dataframes
    scenario_df = pd.concat(dfs)
    scenario_df.insert(0, "scenario", scenario_id)

    return scenario_df


def combine_datasets(dataset_root: Path):
    scenarios = [d.name for d in dataset_root.iterdir() if d.is_dir()]

    dfs: list[pd.DataFrame] = []
    columns = set()
    for id in scenarios:
        # Create scenarios dataframe
        scenarios_dataframe = create_scenario_dataframe(dataset_root, id)

        # Store column names
        columns.update(scenarios_dataframe.columns)

        dfs.append(scenarios_dataframe)

    # Ensure all dataframes have all columns
    for column in columns:
        for df in dfs:
            if column not in df.columns:
                df[column] = np.nan

    # Concat dataset dataframes
    dataset = pd.concat(dfs)
    return dataset

if __name__ == '__main__':
    # Root of dataset
    DATASET_DIRECTORY = Path("dataset")
    df = combine_datasets(DATASET_DIRECTORY)

    output_dir = Path("all_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "synced_marble_data.csv"
    df.to_csv(file_path, index=False)

    print(f"✅ File successfully saved to: {file_path.resolve()}")
