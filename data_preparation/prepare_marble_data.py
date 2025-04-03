import numpy as np
import argparse
import os
import pandas as pd
import pickle
import copy
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import date
from joblib import dump
from pathlib import Path

np.random.seed(42)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters for preparing MARBLE")
    parser.add_argument("--dataset_loc", type=str,
                        default="",
                        help="Location of the MARBLE dataset")
    parser.add_argument("--sampling_rate", type=int, default=50,
                        help="Sampling rate for the data. Is used to downsample to the required rate")
    parser.add_argument("--original_sampling_rate", type=int, default=100,
                        help="Original sampling rate for the dataset")
    parser.add_argument("--perform_naturalization", type=str, default="True",
                        help="To perform mean-variance normalization on the data")
    parser.add_argument("--num_sensor_channels", type=int, default=3,
                        help="Number of sensor channels to be used in the data preparation")
    parser.add_argument("--n_fold_validation", type=int, default=5,
                        help="To extract data with n-folds instead of random 20% test set. Default is 0, which creates the normal 80-20 split.")
    parser.add_argument("--null_class", type=str, default="False",
                        help="To move all the transitionary classes to NULL")

    args = parser.parse_args()

    return args

def map_activity_to_id():
    activity_list = ["ANSWERING_PHONE", "CLEARING_TABLE", "COOKING", "EATING", "ENTERING_HOME", "LEAVING_HOME", 
    "MAKING_PHONE_CALL", "PREPARING_COLD_MEAL", "SETTING_UP_TABLE", "TAKING_MEDICINES", "USING_PC", "WASHING_DISHES", "WATCHING_TV", "TRANSITION", "UNKNOWN"]

    activity_id = {1: "ANSWERING_PHONE", 2: "CLEARING_TABLE", 3: "COOKING", 4: "EATING",
                    5: "ENTERING_HOME", 6: "LEAVING_HOME", 7: "MAKING_PHONE_CALL", 8: "PREPARING_COLD_MEAL", 9: "SETTING_UP_TABLE",
                    10: "TAKING_MEDICINES", 11: "USING_PC", 12: "WASHING_DISHES", 13: "WATCHING_TV", 14: "TRANSITION", 15: "UNKNOWN"}
    
    chosen = {
        1: "ANSWERING_PHONE", 
        2: "CLEARING_TABLE", 
        3: "COOKING", 
        4: "EATING",
        5: "ENTERING_HOME", 
        6: "LEAVING_HOME", 
        7: "MAKING_PHONE_CALL", 
        8: "PREPARING_COLD_MEAL", 
        9: "SETTING_UP_TABLE",
        10: "TAKING_MEDICINES", 
        11: "USING_PC", 
        12: "WASHING_DISHES", 
        13: "WATCHING_TV"
    }

    chosen_activity_list = chosen.values()

    le = LabelEncoder()
    new_labels = le.fit_transform(list(chosen.keys()))

    activity_text_id = {}
    for k in new_labels:
        activity_text_id[chosen[le.inverse_transform([k])[0]]] = k

    return chosen, chosen_activity_list, le, activity_text_id


def perform_train_val_test_split(unique_subj, test_size=0.2, val_size=0.2):
    # Doing the train-test split
    train_val_subj, test_subj = train_test_split(unique_subj, test_size=test_size, random_state=42)
    print(f"The train and validation subjects are: {train_val_subj}")
    print(f"The test subjects are: {test_subj}")

    # Splitting further into train and validation subjects
    train_subj, val_subj = train_test_split(train_val_subj, test_size=val_size, random_state=42)

    subjects = {"train": train_subj, "val": val_subj, "test": test_subj}

    folder = os.path.join("all_data", date.today().strftime("%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "marble_subjects.pkl"), "wb") as f:
        pickle.dump(subjects, f, pickle.HIGHEST_PROTOCOL)

    return subjects


# Function to get directory paths to subjects in all instances in MARBLE dataset
def get_third_level_subdirectories(directory):
    directory = Path(directory)
    third_level_subdirs = []

    first_level_dirs = [f for f in directory.iterdir() if f.is_dir()]

    second_level_dirs = [sub for first_level in first_level_dirs for sub in first_level.iterdir() if sub.is_dir()]

    for second_level in second_level_dirs:
        third_level_subdirs.extend([sub for sub in second_level.iterdir() if sub.is_dir()])

    return third_level_subdirs

def convert_to_id():
    activity_to_id = {
        "ANSWERING_PHONE": 1,
        "CLEARING_TABLE": 2, 
        "COOKING": 3, 
        "EATING": 4,
        "ENTERING_HOME": 5, 
        "LEAVING_HOME": 6, 
        "MAKING_PHONE_CALL": 7, 
        "PREPARING_COLD_MEAL": 8, 
        "SETTING_UP_TABLE": 9,
        "TAKING_MEDICINES": 10, 
        "USING_PC": 11, 
        "WASHING_DISHES": 12, 
        "WATCHING_TV": 13,
        "TRANSITION": 14,
        "UNKNOWN": 15
    }
    return activity_to_id

def get_data(args):
    path_marble = args.dataset_loc
    path_marble_dataset = os.path.join(path_marble, "dataset")
    subdirectories = get_third_level_subdirectories(path_marble_dataset)
    
    # Placeholders to concatenate later
    sensor_all = np.empty((0, 3))
    target_all = np.empty((0,))
    subject_all = np.empty((0,))
    target_col = 1

    activity_to_id = convert_to_id()
    
    for prot in tqdm(subdirectories):
        df_accel = pd.read_csv(os.path.join(prot, "accelerometer.csv"))
        df_labels = pd.read_csv(os.path.join(prot, "labels.csv"))
        print('load from ...', prot, df_accel.shape)

        # foward fill and backward fill missing values
        df_accel = df_accel.ffill().bfill()

        # downsample
        interval = int(args.original_sampling_rate / args.sampling_rate)
        print(f"The interval is: {interval}")
        idx_ds = np.arange(0, df_accel.shape[0], interval)
        df_accel = df_accel.iloc[idx_ds]

        # Convert raw data timestamps to NumPy array
        accel_ts = np.array(df_accel['ts'])

        # Convert label data to NumPy arrays
        label_ts_start = np.array(df_labels['ts_start'])
        label_ts_end = np.array(df_labels['ts_end'])
        label_act = np.array(df_labels['act'])
        print("target classes ...", np.unique(label_act))

        # Initialize an array for the labels, initially set to 'UNKNOWN'
        target = np.full(accel_ts.shape, 'UNKNOWN', dtype=object)

        # For each raw timestamp, find the corresponding label
        for i, ts in enumerate(accel_ts):
            # Find the index of the closest ts_start that is less than or equal to the current timestamp
            idx = np.searchsorted(label_ts_start, ts, side='right') - 1
            if idx >= 0 and label_ts_start[idx] <= ts <= label_ts_end[idx]:
                target[i] = label_act[idx]

        # Convert string to numerical value
        target = np.vectorize(activity_to_id.get)(target)

        sensor = df_accel[['x', 'y', 'z']].values
    
        # Get subject
        basename = os.path.splitext(os.path.basename(prot))[0]
        assert basename[:-2] == "subject-2"
        sID = int(basename[-2:])
        subject = np.ones((target.shape[0],)) * sID

        # Sanity Check
        assert sensor.shape[0] == target.shape[0] and target.shape[0] == subject.shape[0]

        # Concatenate
        sensor_all = np.concatenate((sensor_all, sensor), axis=0)
        target_all = np.concatenate((target_all, target), axis=0)
        subject_all = np.concatenate((subject_all, subject), axis=0)

    print("real {} {} {}".format(
        sensor_all.shape, target_all.shape, subject_all.shape))
    print("labels | real {}".format(np.unique(target_all)))
    print("subject | real {}".format(np.unique(subject_all)))

    # Putting it back into a dataframe
    df_cols = {"user": subject_all, "label": target_all}
    locs = ["wrist"]
    sensor_names = ["acc"]
    axes = ["x", "y", "z"]

    # Looping over all sensor locations
    count = 0
    sensor_col_names = []
    for loc in locs:
        for name in sensor_names:
            for axis in axes:
                c = f"{loc}_{name}_{axis}"
                df_cols[c] = sensor_all[:, count]
                sensor_col_names.append(c)
                count += 1
    
    df = pd.DataFrame(df_cols)

    # Final size check
    assert df.shape[1] == 5, (f"All columns were not copied. Expected 5, got {df.shape[1]}")
    print(f"Done collecting! Shape is: {df.shape}")

    # Removing some classes
    activity_id, activity_list, le, _ = map_activity_to_id()

    df = df[df.label.isin(activity_id.keys())]
    print(f"After removing, the shape is: {df.shape}")
    print(f"The activites are: {np.unique(df['label'])}")

    # Encoding labels from 0 to N-1
    encoded = le.fit_transform(df["label"].values)
    df["gt"] = encoded
    print(f"After label encoding, the shape is: {df.shape}")
    print(f"The activities are: {np.unique(df['gt'])}")

    # Add text descriptions
    df["text_labels"] = df["label"].map(activity_id)

    return df, sensor_col_names


# Function to label data based on timestamp and label time ranges
def label_data(row, labels_df):
    for _, label_row in labels_df.iterrows():
        if label_row["ts_start"] <= row["ts"] <= label_row["ts_end"]:
            return label_row["act"]
    return "UNKNOWN"


def get_data_from_split(df, args, split, n_fold=0):
    # Partition by train, val, and test splits
    train_data = df[df["user"].isin(split["train"])]
    val_data = df[df["user"].isin(split["val"])]
    test_data = df[df["user"].isin(split["test"])]
    print(f"The shapes of the splits are: {train_data.shape}, {val_data.shape}, and {test_data.shape}")
    print(f"The unique classes in train are: {np.unique(train_data['label'])}")
    print(f"The unique classes in val are: {np.unique(val_data['label'])}")
    print(f"The unique classes in test are: {np.unique(test_data['label'])}")

    # Choosing only wrist accelerometry
    sensors = ["wrist_acc_x", "wrist_acc_y", "wrist_acc_z"]

    # Text to class ID mapping
    _, _, _, activity_text_id = map_activity_to_id()

    processed = {"train": {"data": train_data[sensors].values,
                           "labels": train_data["gt"].values,
                           "text_labels": train_data["text_labels"].values},
                "val": {"data": val_data[sensors].values,
                        "labels": val_data["gt"].values,
                        "text_labels": val_data["text_labels"].values},
                "test": {"data": test_data[sensors].values,
                         "labels": test_data["gt"].values,
                         "text_labels": test_data["text_labels"].values},
                "fold": split,
                "activity_text_id": activity_text_id}
    
    # Sanity check on the sizes
    for phase in ["train", "val", "test"]:
        assert processed[phase]["data"].shape[0] == len(processed[phase]["labels"])

    for phase in ["train", "val", "test"]:
        print(f"The phase is: {phase}. The data shape is: {processed[phase]['data'].shape}, {processed[phase]['labels'].shape}")

    # Before normalization
    print("Means before normalization")
    print(np.mean(processed["train"]["data"], axis=0))

    # Creating logs by date
    folder = os.path.join("all_data", date.today().strftime("%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    os.makedirs(os.path.join(folder, "unnormalized"), exist_ok=True)
    args.n_fold = n_fold
    if args.n_fold_validation != 0:
        save_name = f"marble_sr_{args.sampling_rate}_fold_{args.n_fold}"
    else:
        save_name = f"marble_sr_{args.sampling_rate}"
    
    # If null class, add _null to the end
    if args.null_class == "True":
        save_name += "null"
    
    # Saving the joblib file
    save_name_copy = copy.deepcopy(save_name)
    save_name += ".joblib"
    name = os.path.join(folder, "unnormalized", save_name)
    with open(name, "wb") as f:
        dump(processed, f)

    # Performing normalization
    scaler = StandardScaler()
    scaler.fit(processed["train"]["data"])
    for phase in ["train", "val", "test"]:
        processed[phase]["data"] = scaler.transform(processed[phase]["data"])

    # After normalization
    print("Means after normalization")
    print(np.mean(processed["train"]["data"], axis=0))

    # Saving into a joblib file
    name = os.path.join(folder, save_name)
    with open(name, "wb") as f:
        dump(processed, f)

    # Saving the scaler
    name = os.path.join(folder, save_name_copy + "_scaler.joblib")
    with open(name, "wb") as f:
        dump(scaler, f)
    
    print("Saved into a joblib file!")

    return


def prepare_data(args):
    # Getting all the available data
    df, sensors = get_data(args=args)

    # Getting the unique subject IDs for splitting
    unique_subj = np.unique(df["user"].values)
    print(f"The unique subjects are {unique_subj}")

    # Performing the train-val_split
    if args.n_fold_validation == 0:
        split = perform_train_val_test_split(unique_subj)
        get_data_from_split(df, args, split, 0)
    else:
        n_fold_validation = 5
        num_test_subj = int(np.floor((1.0 / n_fold_validation) * len(unique_subj)))
        print(f"The number of validation and test subjects: {num_test_subj}")

        sanity = {"train": [], "val": [], "test": []}

        for i in tqdm(range(n_fold_validation)):
            # First fold is the same as random 80:20 split
            if i == 0:
                split = perform_train_val_test_split(unique_subj)
                train_subj = split["train"]
                val_subj = split["val"]
                test_subj = split["test"]
            else:
                remaining_test = list(set(unique_subj) - set(sanity["test"]))

                if i != n_fold_validation - 1:
                    test_subj = remaining_test[:num_test_subj]
                else:
                    test_subj = remaining_test

                # Remaining participants for train+val
                train_val = list(set(unique_subj) - set(test_subj))

                # Splitting the 80:20
                train_subj, val_subj = train_test_split(train_val, test_size=0.2, random_state=42)

            # Sanity check to make sure all subjects are in test/val only once
            sanity["train"].extend(train_subj)
            sanity["val"].extend(val_subj)
            sanity["test"].extend(test_subj)

            subjects = {"train": train_subj, "val": val_subj, "test": test_subj}
            print("subjects:")
            print(i, subjects)

            # Saving the split data
            get_data_from_split(df, args, split=subjects, n_fold=i)

        # For test split, each participant should be there only once
        print(sanity)
        assert len(sanity["test"]) == 12
        assert sanity["test"].sort() == list(unique_subj).sort()

        v, c = np.unique(sanity["test"], return_counts=True)
        assert np.sum(c == 1) == 12

    return


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    
    prepare_data(args)
    print("Data preparation complete!")


