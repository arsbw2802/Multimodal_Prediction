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
__author__ = 'Sourish Gunesh Dhekane'
__copyright__ = ''
__credits__ = []
__license__ = ''
__version__ = '1.0'
__maintainer__ = 'Sourish Gunesh Dhekane'
__email__ = 'sourish.dhekane@gatech.edu'
__status__ = ''

# Dependencies.
import os
import numpy as np
from collections import Counter
import argparse

parser = argparse.ArgumentParser(description='???')


# TODO: these could be just saved as a dict from ./data_preprocess/data.py, then loaded from a new .npy file.
arubaDict = {0 : "Other", 1 : "Bed_to_Toilet", 2 : "Eat", 3 : "Enter_Home", 4 : "Housekeeping", 5 : "Leave_Home", 6 : "Meal_Preparation", 7 : "Relax", 8 : "Resperate", 9 : "Sleeping", 10 : "Wash_Dishes", 11 : "Work"}
milanDict = {0 : "Other", 1 : "Bed_to_Toilet", 2 : "Chores", 3 : "Desk_Activity", 4 : "Dining_Rm_Activity", 5 : "Evening_Meds", 6 : "Guest_Bathroom", 7 : "Kitchen_Activity", 8 : "Leave_Home", 9 : "Master_Bathroom", 10 : "Master_Bedroom_Activity", 11 : "Meditate", 12 : "Morning_Meds", 13 : "Read", 14 : "Sleep", 15 : "Watch_TV"}
cairoDict = {0 : "Other", 1 : "Bed_to_Toilet", 2 : "Breakfast", 3 : "Dinner", 4 : "Laundry", 5 : "Leave_Home", 6 : "Lunch", 7 : "Night_Wandering", 8 : "R1_Sleep", 9 : "R1_Wake", 10 : "R1_Work_in_Office", 11 : "R2_Sleep", 12 : "R2_Take_Meds", 13 : "R2_Wake"}
kyoto7Dict = {0: "Other", 1: "Clean", 2: "Meal_Preparation", 3: "R1_Bed_to_Toilet", 4: "R1_Personal_Hygiene", 5: "R1_Sleep", 6: "R1_Work", 7: "R2_Bed_to_Toilet", 8: "R2_Personal_Hygiene", 9: "R2_Sleep", 10: "R2_Work", 11: "Study", 12: "Wash_Bathtub", 13: "Watch_TV"}

# marble_combined = {0: 'Other', 4: 'Eating', 10: 'Taking_medicines', 13: 'Watching_tv', 3: 'Cooking', 7: 'Making_phone_call', 12: 'Washing_dishes', 5: 'Entering_home', 1: 'Answering_phone', 6: 'Leaving_home', 11: 'Using_pc', 9: 'Setting_up_table', 2: 'Clearing_table', 8: 'Preparing_cold_meal'} #original
marble_combined = {0: 'Other', 4: "Eat", 10: 'Take_Medicine', 13: 'Watch_TV', 3: 'Cook', 7: 'Work', 12: 'Wash_Dishes', 5: 'Enter_Home', 1: 'Work', 6: 'Leave_Home', 11: 'Work', 9: 'Meal_Preparation', 2: 'Clean', 8: 'Meal_Preparation'}                                       #adapted

# TODO: This dict is problematic; some marble activity labels dont seem to corespond to an entry in the translateDict. see `marble_A1a`
translateDict = {"Other" : 0, "Relax" : 1, "Cook" : 2, "Leave_Home" : 3, "Enter_Home" : 4, "Sleep" : 5, "Eat" : 6, "Work" : 7, "Bed_to_Toilet" : 8, "Bathing" : 9, "Take_Medicine" : 10, "Wash_Dishes" : 7, "Housekeeping" : 7, "Resperate" : 0, "Meal_Preparation" : 2, "Sleeping" : 5, "Eating" : 6, "Kitchen_Activity" : 2, "Guest_Bathroom" : 9, "Read" : 1, "Master_Bathroom" : 9, "Master_Bedroom_Activity" : 0, "Watch_TV" : 1, "Desk_Activity" : 7, "Morning_Meds" : 10, "Chores" : 7, "Dining_Rm_Activity" : 6, "Evening_Meds" : 10, "Meditate" : 0, "Night_Wandering" : 0, "R1_Wake" : 0, "R2_Wake" : 0, "R2_Sleep" : 5, "R1_Sleep" : 5, "Breakfast" : 6, "R1_Work_in_Office" : 7, "R2_Take_Meds" : 10, "Dinner" : 6, "Lunch" : 6, "Laundry" : 7, "Clean": 7, "R1_Bed_to_Toilet": 8, "R1_Personal_Hygiene": 11, "R2_Bed_to_Toilet": 8, "R2_Personal_Hygiene": 11, "R2_Work": 7, "Study": 0, "Wash_Bathtub": 0, "R1_Work": 7} 
def open_raw_data_file(path):

    data = np.load(path, allow_pickle=True)
    return data

def convert_num_labels_to_text_labels(labels, datasetDict):

    text_labels = []

    for num_label in labels:
        text_labels.append(datasetDict[num_label])

    text_labels = np.array(text_labels)

    return text_labels

def convert_text_labels_to_global_labels(labels, translateDict):

    global_labels = []

    for text_label in labels:
        global_labels.append(translateDict[text_label])

    global_labels = np.array(global_labels)

    return global_labels

def merge_classes(dataset_name):

    labels = open_raw_data_file(f"{npy_path}/{dataset_name}-y.npy")

    if dataset_name == "aruba":
        text_labels = convert_num_labels_to_text_labels(labels, arubaDict)
    elif dataset_name == "milan":
        text_labels = convert_num_labels_to_text_labels(labels, milanDict)
    elif dataset_name == "cairo":
        text_labels = convert_num_labels_to_text_labels(labels, cairoDict)
    elif dataset_name == "kyoto7":
        text_labels = convert_num_labels_to_text_labels(labels, kyoto7Dict)
    
    elif "MARBLE" in dataset_name:
        text_labels = convert_num_labels_to_text_labels(labels, marble_combined)

        
    else:
        print("Dataset Not Incorporated Here!")

    global_labels = convert_text_labels_to_global_labels(text_labels, translateDict)

    print("Dataset:\t" + str(dataset_name))
    print(Counter(global_labels))

    np.save(f"{npy_path}/{dataset_name}-global_labels.npy", global_labels)


def discover_datasets_from_npy_files(npy_path:str):
    files_in_npy = os.listdir(npy_path)
    discovered_datasets = set()
    discard_keywords = {"embeddings", "v1", "sentence"}
    
    for file in files_in_npy: #loop 1
        file_desc = file.split("-")
        data_name = file_desc[0]

        name_decomp = data_name.split("_")

        skip_outer_loop = False

        for item in name_decomp:  # loop 2
            if item in discard_keywords:
                skip_outer_loop = True  
                break  

        if skip_outer_loop:
            continue  
        else:
            discovered_datasets.add(data_name)
        # discovered_datasets.add()
    discovered_datasets_list = list(discovered_datasets)
    print("Discovered these datasets from ./npy/:", discovered_datasets_list, "\n")
    return discovered_datasets_list


# Main.
if __name__ == '__main__':

    parser.add_argument('--npy_path', type=str, help='absolute path where stuff will be saved.',default="/coc/pcba1/mthukral3/gt/TDOST/npy/pre-segmented")
    parser.add_argument(
        '--datasets', 
        type=str, 
        help='Names of datasets to process', 
        nargs='+',
        default=['aruba', 'cairo', 'milan']
    )
    args = parser.parse_args()
    
    npy_path = args.npy_path

    # datasets = discover_datasets_from_npy_files(npy_path)
    datasets = args.datasets
    
    assert len(datasets)!=0, "No dataset provided!"
    # print("Processing the following datasets:", datasets, "\n")
    
    # Merge the classes as per DeepCasas
    
    for dataset_name in datasets:
        merge_classes(dataset_name)