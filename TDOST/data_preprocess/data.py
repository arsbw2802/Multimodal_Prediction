#!/usr/bin/env/python3


"""
About
-----

Classes
-------

Functions
---------

Variables
---------
"""

# Meta-data.
__author__ = 'Daniele Liciotti'
__copyright__ = ''
__credits__ = []
__license__ = ''
__version__ = '1.0'
__maintainer__ = 'Daniele Liciotti'
__email__ = 'd.liciotti@pm.univpm.it'
__status__ = ''

# Dependencies.

import datetime
import os
import re
from collections import Counter
from datetime import datetime
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='???')

datasets = ["../dataset/aruba/aruba", "../dataset/milan/milan", "../dataset/cairo/cairo"]
# datasets = ["../dataset/kyoto8/kyoto8", "../dataset/kyoto7/kyoto7", "../dataset/kyoto11/kyoto11"]
datasetsNames = [i.split('/')[-1] for i in datasets]


def load_dataset(filename):
    # dataset fields
    timestamps = []
    sensors = []
    values = []
    activities = []

    activity = ''  # empty activity placeholder

    with open(filename, 'rb') as features:
        database = features.readlines()
        for i, line in enumerate(database):  # each line is a sensor event

            f_info = line.decode().split()  # split the line to fields
            try:
                # Supporting both motion/temp/door (M, D, T) and IMU (W) sensors
                if 'M' == f_info[2][0] or 'D' == f_info[2][0] or 'T' == f_info[2][0] or 'W' in f_info[2][0]:
                    # Only process lines with relevant sensor IDs
                    if not ('.' in str(np.array(f_info[0])) + str(np.array(f_info[1]))):
                        f_info[1] = f_info[1] + '.000000'  # Ensure timestamp precision

                    # Add timestamp, sensor ID, value, and activity (if available)
                    timestamps.append(datetime.strptime(str(np.array(f_info[0])) + str(np.array(f_info[1])),
                                                        "%Y-%m-%d%H:%M:%S.%f"))
                    sensors.append(str(np.array(f_info[2])))
                    values.append(str(np.array(f_info[3])))

                    if len(f_info) == 4:  # if no activity is provided
                        activities.append(activity)
                    else:  # if activity is provided
                        des = str(' '.join(np.array(f_info[4:])))
                        if 'begin' in des:
                            activity = re.sub('begin', '', des).strip()
                        activities.append(activity if 'begin' not in des else '')

            except IndexError:
                print(i, line)

    # dictionaries for mapping sensors, activities, and values
    temperature = [float(v) for v in values if v.replace('.', '', 1).isdigit()]
    sensorsList = sorted(set(sensors))
    dictSensors = {sensor: i for i, sensor in enumerate(sensorsList)}
    activityList = sorted(set(activities))
    dictActivities = {activity: i for i, activity in enumerate(activityList)}
    valueList = sorted(set(values))
    dictValues = {v: i for i, v in enumerate(valueList)}

    # IMU data-specific handling: mapping textual categories
    imu_textual_mapping = {
        'no_change': 0,
        'low': 1,
        'medium': 2,
        'high': 3,
        'extreme': 4
    }

    dictObs = {}
    rangeObs = {}
    count = 0
    for key in dictSensors.keys():
        if key.startswith('M') or key.startswith('AD'):  # CASAS sensors
            dictObs[key + "OFF"] = count
            count += 1
            dictObs[key + "ON"] = count
            count += 1
        if key.startswith('D'):
            dictObs[key + "CLOSE"] = count
            count += 1
            dictObs[key + "OPEN"] = count
            count += 1
        if key.startswith('T'):
            rangeObs[key] = [round(min(temperature) + float(temp / 2.0), 2)
                             for temp in range(0, int((max(temperature) - min(temperature)) * 2) + 1)]
            for temp in rangeObs[key]:
                dictObs[key + str(temp)] = count
                count += 1
        if key.startswith('W'):  # IMU sensors: apply textual labels (e.g., 'low', 'high')
            for label in imu_textual_mapping.keys():
                dictObs[key + label] = count
                count += 1

    # Data preparation
    XX, YY, X, Y, X_sensor, X_value, X_time = [], [], [], [], [], [], []
    sensor_x, value_x, time_x = [], [], []

    for kk, s in enumerate(sensors):
        sensor_x.append(s)
        time_x.append(timestamps[kk])

        if "c" in values[kk]:
            values[kk] = values[kk].replace("c", "")

        if s.startswith('T'):  # Temperature sensor: map to closest temperature range
            closest_value = min(rangeObs[s], key=lambda x: abs(x - float(values[kk])))
            XX.append(closest_value)
            value_x.append(closest_value)
        elif s.startswith('W'):  # IMU sensor: map to categorical labels
            XX.append(values[kk])
            value_x.append(values[kk])
        else:  # CASAS motion/door sensors
            XX.append(values[kk])
            value_x.append(values[kk])

        YY.append(dictActivities[activities[kk]])

    x, x_sensor, x_value, x_time = [], [], [], []
    for i, y in enumerate(YY):
        if i == 0 or y != YY[i - 1]:
            if i > 0:
                Y.append(y)
                X.append(x)
                X_sensor.append(x_sensor)
                X_value.append(x_value)
                X_time.append(x_time)
            x, x_sensor, x_value, x_time = [XX[i]], [sensor_x[i]], [value_x[i]], [time_x[i]]
        else:
            x.append(XX[i])
            x_sensor.append(sensor_x[i])
            x_value.append(value_x[i])
            x_time.append(time_x[i])

        if i == len(YY) - 1:  # Add final segment
            X.append(x)
            X_sensor.append(x_sensor)
            X_value.append(x_value)
            X_time.append(x_time)

    return X_sensor, X_value, X_time, Y, dictActivities


def get_files_without_extension(directory):
    no_extension_files = []
    
    # List all items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Check if the item is a file and does not have an extension
        if os.path.isfile(item_path) and not os.path.splitext(item)[1]:
            full_path = os.path.join(directory, item)
            full_path = os.path.normpath(full_path).replace("\\", "/")
            no_extension_files.append(full_path)
    
    return no_extension_files


def extend_with_marble_dataset(datasets, marble_dataset_path):
    if not os.path.exists(marble_dataset_path):
        print("Error reading marble dataset path; does it exist?")
        raise FileNotFoundError(marble_dataset_path)

    marble_files = get_files_without_extension(marble_dataset_path)
    datasets = datasets + marble_files
    return datasets


def ensure_2_of_each(count: dict):
    """
    sanity check to ensure that the data is able to be K fold split
    """    
    for key, value in count.items():
        assert value >= 2, f"Key '{key}' has a value of {value}, which is less than 2. This will be problematic with further scripts, especially `prepare_folds.py`!"

if __name__ == '__main__':
    parser.add_argument('--npy_path', type=str, help='absolute path where stuff will be saved.',default="/coc/pcba1/mthukral3/gt/TDOST/npy/pre-segmented")
    parser.add_argument('--marble_dataset_path', type=str, help='???',default=None)
    

    args = parser.parse_args()
    
    npy_path = args.npy_path
    marble_dataset_path = args.marble_dataset_path
    
    if marble_dataset_path:
        datasets = extend_with_marble_dataset(datasets, marble_dataset_path)
    
    print("Discovered these datasets:", datasets, "\n")

    for filename in datasets:

        datasetName = filename.split("/")[-1]
        print('Loading ' + datasetName + ' dataset ...')
        X_sensor, X_value, X_time, Y, dictActivities = load_dataset(filename)

        print(sorted(dictActivities, key=dictActivities.get))
        print("n° instances post-filtering:\t" + str(len(X_sensor)))

        print(Counter(Y))

        ensure_2_of_each(Counter(Y))

        X_sensor = np.array(X_sensor, dtype=object)
        X_value = np.array(X_value, dtype=object)
        X_time = np.array(X_time, dtype=object)
        Y = np.array(Y, dtype=object)

        if not os.path.exists('../npy'):
            os.makedirs('../npy')
            os.makedirs('../npy/pre-segmented')

        base_path = os.path.normpath(os.path.join(npy_path, datasetName)).replace("\\", "/")
        np.save(f'{base_path}-x_sensor.npy', X_sensor)
        np.save(f'{base_path}-x_value.npy', X_value)
        np.save(f'{base_path}-x_time.npy', X_time)
        np.save(f'{base_path}-y.npy', Y)
        np.save(f'{base_path}-labels.npy', dictActivities)
