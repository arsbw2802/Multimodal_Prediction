"""
This script creates folds of multi-dimensional (usually x,y,z) IMU data and saves them as `.joblib` files

To use this script, ensure you have a readings file, and a corresponding labels file.

The script below is simlar to the `main.py` script used to prepare the Blunck et.al dataset to be ingested by Mehga's supervised model.

NOTE: Due to the number of unique subjects in the MARBLE dataset, setting `--n_fold_validation=5` may result in errors.
"""

import argparse
import pickle
from datetime import date
import string

import numpy as np
import os
import pandas as pd
import random
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import load
import copy

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters for '
                                                 'preparing HHAR watch')
    parser.add_argument('--dataset_loc', type=str,
                        default='/coc/pcba1/mthukral3/gt/datasets/hhar/Activity recognition exp/Watch_accelerometer.csv',
                        help='Location of the raw sensory data')
    parser.add_argument('--sampling_rate', type=int, default=50,
                        help='Sampling rate for the data. Is used to '
                             'downsample to the required rate')
    parser.add_argument('--num_sensor_channels', type=int, default=3,
                        help='Number of sensor channels for used in the data '
                             'preparation')
    parser.add_argument('--n_fold_validation', type=int, default=5,
                        help='To extract data with n-folds instead of random '
                             '20% test set. Default is 0, which creates the '
                             'normal 80-20 split.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--perform_normalization', type=str, default='True',
                        help='To perform mean-variance normalization on the '
                             'data')

    args = parser.parse_args()

    return args

def assign_letters(strings):
    letters = list(string.ascii_lowercase)
    assigned = {}

    # Add extended letters if more than 26 strings are provided
    while len(letters) < len(strings):
        letters += [f"{first}{second}" for first in string.ascii_uppercase for second in string.ascii_uppercase]
    
    # Map each string to a unique letter
    for i, s in enumerate(strings):
        assigned[s] = letters[i]

    return assigned

def map_activity_to_id():
    # List of activities being studied. Note that we *dont* use lying down class
    activity_list = [
        "Eating",
        "Taking_medicines",
        "Watching_tv",
        "Cooking",
        "Making_phone_call",
        "Washing_dishes",
        "Entering_home",
        "Answering_phone",
        "Leaving_home",
        "Using_pc",
        "Setting_up_table",
        "Transition",
        "Clearing_table",
        "Preparing_cold_meal",
        ]

    activity_id = {
        'Eating': 0,
        'Taking_medicines': 1,
        'Watching_tv': 2,
        'Cooking': 3,
        'Making_phone_call': 4,
        'Washing_dishes': 5,
        'Entering_home': 6,
        'Answering_phone': 7,
        'Leaving_home': 8,
        'Using_pc': 9,
        'Setting_up_table': 10,
        'Transition': 11,
        'Clearing_table': 12,
        'Preparing_cold_meal': 13
    }
    
    

    return activity_id, activity_list

def label_data(data:pd.DataFrame, labels:pd.DataFrame)-> pd.DataFrame:
    """
    Given a data datafame of x,y,z,ts cols, and a labels dataframe that has ts_start,ts_end,act which indicates windows of activity labels, 
    return a dataframe that contains x,y,z,ts,act, based on whether data[row][ts] is within a labels[act]'s start and end ts 

    """
    def drop_transitions(df_labels):
        # remove TRANSITION labels
        df_labels_filtered = df_labels[df_labels['act'] != 'TRANSITION']
        df_labels_filtered=df_labels_filtered.reset_index(drop=True)
        return df_labels_filtered
    
    # labels = drop_transitions(labels).reset_index()
    data['act'] = None

    # Iterate over each label interval
    for _, label_row in labels.iterrows():
        ts_start = label_row['ts_start']
        ts_end = label_row['ts_end']
        activity:str = label_row['act']


        # Label the data rows that fall within the current activity window
        within_window = (data['ts'] >= ts_start) & (data['ts'] <= ts_end)
        data.loc[within_window, 'act'] = activity.capitalize()
    
    return data

def collect_file_paths(dataset_dir) -> list[tuple[str, str, str]]:
    file_paths = []
    subjects = {}

    # Traverse the dataset directory
    for scenario in os.listdir(dataset_dir):
        if "1" not in scenario :
            continue

        scenario_path = os.path.join(dataset_dir, scenario)

        # Traverse instances within each single-inhabitant scenario
        for instance in os.listdir(scenario_path):
            instance_path = os.path.join(scenario_path, instance)

            # Retrieve paths to required files if they exist
            subject_dir = os.path.join(instance_path, next(os.walk(instance_path))[1][0])  # get subject-id folder

            accelerometer_path = os.path.join(subject_dir, 'accelerometer.csv')
            labels_path = os.path.join(subject_dir, 'labels.csv')
            subject_id = os.path.basename(subject_dir)
            
            if not subjects.get(subject_id):
                subjects[subject_id] = list(string.ascii_lowercase[len(subjects.keys())])

            # Check if both files exist before adding them
            if os.path.exists(accelerometer_path) and os.path.exists(labels_path):
                file_paths.append((accelerometer_path, labels_path, *subjects[subject_id]))

    return file_paths

def combine_dataset(args):
    paths = collect_file_paths(args.dataset_loc)
    data_sep = []
    for accelerometer_path, labels_path, subject_id in paths:
        data = pd.read_csv(accelerometer_path)
        labels = pd.read_csv(labels_path)
        data = label_data(data, labels)
        data['user'] = subject_id

        data_sep.append(data)
    
    data_combined = pd.concat(data_sep)
    return data_combined

def get_data(args):
    contianing_folder:str = args.dataset_loc
    data_loc:str = os.path.join(contianing_folder, 'acc_data')
    label_loc:str = os.path.join(contianing_folder, 'labels')

    
    # # Getting the activity labels
    activity_id, activity_list = map_activity_to_id()

    # # Loading data from the csv
    data = pd.read_csv(data_loc)
    labels = pd.read_csv(label_loc)

    # data = data.dropna().reset_index(drop=True)
    
    data = label_data(data, labels)
    # data = combine_dataset(args)

    df = pd.DataFrame()
    df['acc_x'] = data['x']
    df['acc_y'] = data['y']
    df['acc_z'] = data['z']
    df['text_labels'] = data['act']
    df['gt'] = data['act'].map(activity_id)
    print("df['gt']", df['gt'].dropna())

    df['user'] = data['user']
    #adding demographic info
    df['demographic_info'] = 'age: 25 to 30 years'
    
    
    print("length of df", len(df))
   
    print('Done collecting!')
    print(df)
    return df

def perform_train_val_test_split(unique_subj, test_size=0.2, val_size=0.2):
    
    # Doing the train-test split
    train_val_subj, test_subj = train_test_split(unique_subj,
                                                 test_size=test_size,
                                                 random_state=args.seed)
    print('The train and validation subjects are: {}'.format(train_val_subj))
    print('The test subjects are: {}'.format(test_subj))

    # Splitting further into train and validation subjects
    train_subj, val_subj = train_test_split(train_val_subj, test_size=val_size,
                                            random_state=args.seed)

    subjects = {'train': train_subj, 'val': val_subj, 'test': test_subj}

    folder = os.path.join('all_data', date.today().strftime(
        "%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'subjects.pkl'), 'wb') as f:
        pickle.dump(subjects, f, pickle.HIGHEST_PROTOCOL)

    return subjects

def get_data_from_split(df, split, args, n_fold=0):
    activity_text_id , _ = map_activity_to_id()
    # Let us partition by train, val and test splits
    train_data = df[df['user'].isin(split['train'])]
    val_data = df[df['user'].isin(split['val'])]
    test_data = df[df['user'].isin(split['test'])]
    print('The shapes of the splits are: {}, {} and {}'.
          format(train_data.shape, val_data.shape, test_data.shape))

    print('The unique classes in train are: {}'
          .format(np.unique(train_data['gt'])))
    print('The unique classes in val are: {}'
          .format(np.unique(val_data['gt'])))
    print('The unique classes in test are: {}'
          .format(np.unique(test_data['gt'])))

    if args.num_sensor_channels == 3:
        sensors = ['acc_x', 'acc_y', 'acc_z']
    elif args.num_sensor_channels == 6:
        sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    #adding demographic col
    
  
    processed = {'train': {'data': train_data[sensors].values,
                           'labels': train_data['gt'].values,
                           'demographic_info': train_data['demographic_info'].values
                           },
                 'val': {'data': val_data[sensors].values,
                         'labels': val_data['gt'].values,
                         'demographic_info': val_data['demographic_info'].values
                         },
                 'test': {'data': test_data[sensors].values,
                          'labels': test_data['gt'].values,
                          'demographic_info': test_data['demographic_info'].values
                          },
                 'fold': split,
                 'activity_text_id': activity_text_id,
                 
                 }

    # Sanity check on the sizes
    for phase in ['train', 'val', 'test']:
        assert processed[phase]['data'].shape[0] == \
               len(processed[phase]['labels'])

    for phase in ['train', 'val', 'test']:
        print('The phase is: {}. The data shape is: {}, {}'
              .format(phase, processed[phase]['data'].shape,
                      processed[phase]['labels'].shape))

    # Before normalization
    print('Means before normalization')
    print(np.mean(processed['train']['data'], axis=0))

    # Creating logs by the date now. To make stuff easier
    folder = os.path.join('all_data', date.today().strftime(
        "%b-%d-%Y"))
    os.makedirs(folder, exist_ok=True)

    os.makedirs(os.path.join(folder, 'unnormalized'), exist_ok=True)
    args.n_fold = n_fold
    if args.n_fold_validation != 0:
        save_name = 'hhar_watch_sr_{0.sampling_rate}_fold_{0.n_fold}' \
            .format(args)
    else:
        save_name = 'hhar_watch_sr_{0.sampling_rate}'.format(args)

    # Saving the joblib file
    # save_name += '.joblib'
    # name = os.path.join(folder, 'unnormalized', save_name)
    # with open(name, 'wb') as f:
    #     dump(processed, f)

    # Saving the joblib file
    save_name_copy = copy.deepcopy(save_name)
    save_name += '.joblib'
    name = os.path.join(folder, 'unnormalized', save_name)
    with open(name, 'wb') as f:
        dump(processed, f)

    # Performing normalization
    scaler = StandardScaler()
    scaler.fit(processed['train']['data'])
    for phase in ['train', 'val', 'test']:
        processed[phase]['data'] = \
            scaler.transform(processed[phase]['data'])

    # After normalization
    print('Means after normalization')
    print(np.mean(processed['train']['data'], axis=0))

    # Saving into a joblib file
    name = os.path.join(folder, save_name)
    with open(name, 'wb') as f:
        dump(processed, f)

    # Saving the scaler
    name = os.path.join(folder, save_name_copy + '_scaler.joblib')
    with open(name, 'wb') as f:
        dump(scaler, f)

    print('Saved into a joblib file!')

    return

def prepare_data(args):
    # Reading in all the data
    df = get_data(args=args)

    # Getting the unique subject IDs for splitting
    unique_subj = np.unique(df['user'].values)
    print('The unique subjects are: {}'.format(unique_subj))

    train_val_split_ratio = getattr(args, 'train_val_split_ratio', 0.8)
    n_fold_validation = args.n_fold_validation
    num_test_subj = int(np.ceil((1.0 / n_fold_validation) * len(unique_subj)))

    print(f'The number of validation and test subjects per fold: {num_test_subj}')

    sanity = {'train': [], 'val': [], 'test': []}
    expected_test_subjects = len(unique_subj) // n_fold_validation

    for i in range(n_fold_validation):
        # Adjust remaining subjects for test allocation in the last fold
        if i == n_fold_validation - 1:
            test_subj = list(set(unique_subj) - set(sanity['test']))
        else:
            remaining_test = list(set(unique_subj) - set(sanity['test']))
            np.random.shuffle(remaining_test)
            test_subj = remaining_test[:num_test_subj]

        # Ensure that there are subjects in the test set for each fold
        if not test_subj:
            raise ValueError(f"Fold {i} has an empty test set. Please check subject allocation. \n Try changing the number of folds `n_fold_validation`.")

        # Assign remaining subjects for train and validation
        train_val = list(set(unique_subj) - set(test_subj))
        train_subj, val_subj = train_test_split(train_val, test_size=(1 - train_val_split_ratio), random_state=args.seed)

        sanity['train'].extend(train_subj)
        sanity['val'].extend(val_subj)
        sanity['test'].extend(test_subj)

        subjects = {'train': train_subj, 'val': val_subj, 'test': test_subj}
        print(f'Fold {i}: {subjects}')

        get_data_from_split(df, split=subjects, args=args, n_fold=i)

    # Sanity check across folds
    print(f'Sanity check across folds: {sanity}')
    unique_test_subjects, counts = np.unique(sanity['test'], return_counts=True)
    assert len(sanity['test']) == expected_test_subjects * n_fold_validation, (
        f"Expected {expected_test_subjects * n_fold_validation} test subjects, "
        f"but found {len(sanity['test'])}"
    )
    assert np.all(counts == 1), (
        f"Expected each test subject to appear only once; found counts: {dict(zip(unique_test_subjects, counts))}"
    )

    return


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    prepare_data(args)