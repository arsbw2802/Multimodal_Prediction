import numpy as np
import socket

arubaDict = {0 : "Other", 1 : "Bed_to_Toilet", 2 : "Eat", 3 : "Enter_Home", 4 : "Housekeeping", 5 : "Leave_Home", 6 : "Meal_Preparation", 7 : "Relax", 8 : "Resperate", 9 : "Sleeping", 10 : "Wash_Dishes", 11 : "Work"}
milanDict = {0 : "Other", 1 : "Bed_to_Toilet", 2 : "Chores", 3 : "Desk_Activity", 4 : "Dining_Rm_Activity", 5 : "Evening_Meds", 6 : "Guest_Bathroom", 7 : "Kitchen_Activity", 8 : "Leave_Home", 9 : "Master_Bathroom", 10 : "Master_Bedroom_Activity", 11 : "Meditate", 12 : "Morning_Meds", 13 : "Read", 14 : "Sleep", 15 : "Watch_TV"}
cairoDict = {0 : "Other", 1 : "Bed_to_Toilet", 2 : "Breakfast", 3 : "Dinner", 4 : "Laundry", 5 : "Leave_Home", 6 : "Lunch", 7 : "Night_Wandering", 8 : "R1_Sleep", 9 : "R1_Wake", 10 : "R1_Work_in_Office", 11 : "R2_Sleep", 12 : "R2_Take_Meds", 13 : "R2_Wake"}

translateDict = {"Other" : 0, "Relax" : 1, "Cook" : 2, "Leave_Home" : 3, "Enter_Home" : 4, "Sleep" : 5, "Eat" : 6, "Work" : 7, "Bed_to_Toilet" : 8, "Bathing" : 9, "Take_Medicine" : 10, "Wash_Dishes" : 7, "Housekeeping" : 7, "Resperate" : 0, "Meal_Preparation" : 2, "Sleeping" : 5, "Eating" : 6, "Kitchen_Activity" : 2, "Guest_Bathroom" : 9, "Read" : 1, "Master_Bathroom" : 9, "Master_Bedroom_Activity" : 0, "Watch_TV" : 1, "Desk_Activity" : 7, "Morning_Meds" : 10, "Chores" : 7, "Dining_Rm_Activity" : 6, "Evening_Meds" : 10, "Meditate" : 0, "Leave_Home" : 3, "Night_Wandering" : 0, "R1_Wake" : 0, "R2_Wake" : 0, "R2_Sleep" : 5, "R1_Sleep" : 5, "Breakfast" : 6, "R1_Work_in_Office" : 7, "R2_Take_Meds" : 10, "Dinner" : 6, "Lunch" : 6, "Laundry" : 7}


def open_raw_data_file(path):

    data = np.load(path, allow_pickle=True)

    return data

def get_npy_path():
    
    if "mt" in socket.gethostname():
        npy_path= "/mnt/attached1/TDOST/npy"
    return npy_path

def convert_num_labels_to_text_labels(labels, datasetDict):

    text_labels = []

    for num_label in labels:
        text_labels.append(datasetDict[num_label])

    text_labels = np.array(text_labels)

    return text_labels