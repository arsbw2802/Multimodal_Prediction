import socket
import math
import inflect
from datetime import datetime
import numpy as np



arubaDict = {0 : "Other", 1 : "Bed_to_Toilet", 2 : "Eat", 3 : "Enter_Home", 4 : "Housekeeping", 5 : "Leave_Home", 6 : "Meal_Preparation", 7 : "Relax", 8 : "Resperate", 9 : "Sleeping", 10 : "Wash_Dishes", 11 : "Work"}
milanDict = {0 : "Other", 1 : "Bed_to_Toilet", 2 : "Chores", 3 : "Desk_Activity", 4 : "Dining_Rm_Activity", 5 : "Evening_Meds", 6 : "Guest_Bathroom", 7 : "Kitchen_Activity", 8 : "Leave_Home", 9 : "Master_Bathroom", 10 : "Master_Bedroom_Activity", 11 : "Meditate", 12 : "Morning_Meds", 13 : "Read", 14 : "Sleep", 15 : "Watch_TV"}
cairoDict = {0 : "Other", 1 : "Bed_to_Toilet", 2 : "Breakfast", 3 : "Dinner", 4 : "Laundry", 5 : "Leave_Home", 6 : "Lunch", 7 : "Night_Wandering", 8 : "R1_Sleep", 9 : "R1_Wake", 10 : "R1_Work_in_Office", 11 : "R2_Sleep", 12 : "R2_Take_Meds", 13 : "R2_Wake"}

translateDict = {"Other" : 0, "Relax" : 1, "Cook" : 2, "Leave_Home" : 3, "Enter_Home" : 4, "Sleep" : 5, "Eat" : 6, "Work" : 7, "Bed_to_Toilet" : 8, "Bathing" : 9, "Take_Medicine" : 10, "Wash_Dishes" : 7, "Housekeeping" : 7, "Resperate" : 0, "Meal_Preparation" : 2, "Sleeping" : 5, "Eating" : 6, "Kitchen_Activity" : 2, "Guest_Bathroom" : 9, "Read" : 1, "Master_Bathroom" : 9, "Master_Bedroom_Activity" : 0, "Watch_TV" : 1, "Desk_Activity" : 7, "Morning_Meds" : 10, "Chores" : 7, "Dining_Rm_Activity" : 6, "Evening_Meds" : 10, "Meditate" : 0, "Leave_Home" : 3, "Night_Wandering" : 0, "R1_Wake" : 0, "R2_Wake" : 0, "R2_Sleep" : 5, "R1_Sleep" : 5, "Breakfast" : 6, "R1_Work_in_Office" : 7, "R2_Take_Meds" : 10, "Dinner" : 6, "Lunch" : 6, "Laundry" : 7}


def convert_num_labels_to_text_labels(labels, datasetDict):

    text_labels = []

    for num_label in labels:
        text_labels.append(datasetDict[num_label])

    text_labels = np.array(text_labels)

    return text_labels

def open_raw_data_file(path):

    data = np.load(path, allow_pickle=True)

    return data
def convert_num_to_text(num):

    p = inflect.engine()
    word = p.number_to_words(num)

    return word

def prepare_time_sentence_v11(text_hour, text_minute, am_pm_flag):
    sentence = "at " + text_hour + " hours " + text_minute + " minutes " + am_pm_flag

    return sentence

def process_time_v11(raw_time, j, prev_reading):

    current_split_time = raw_time.split(":")
    current_hour = current_split_time[0]
    current_minute = current_split_time[1]

    if j == 0:
        if int(current_hour) < 12:
            am_pm_flag = "AM "
        else:
            am_pm_flag = "PM "

        text_hour = convert_num_to_text(current_hour)
        text_minute = convert_num_to_text(current_minute)

        time_sentence = prepare_time_sentence_v11(text_hour, text_minute, am_pm_flag)
    else:
        current_second_data = current_split_time[2]
        current_split_second = current_second_data.split(".")
        current_second = current_split_second[0]

        prev_split_time = prev_reading.split(":")
        prev_hour = prev_split_time[0]
        prev_minute = prev_split_time[1]

        prev_second_data = prev_split_time[2]
        prev_split_second = prev_second_data.split(".")
        prev_second = prev_split_second[0]

        current_t = current_hour + ":" + current_minute + ":" + current_second
        prev_t = prev_hour + ":" + prev_minute + ":" + prev_second

        t1 = datetime.strptime(current_t, "%H:%M:%S")
        t2 = datetime.strptime(prev_t, "%H:%M:%S")
        delta = t1 - t2

        time_diff = convert_num_to_text(math.floor(float(delta.total_seconds())))

        time_sentence = "After " + time_diff + " seconds, "

    return time_sentence


def get_npy_path():
    
    if "mt" in socket.gethostname():
        npy_path= "/mnt/attached1/TDOST/npy"
    return npy_path


def get_time_period(hour):
    if 0 <= hour < 5:
        return "Night"
    elif 5 <= hour < 8:
        return "Early Morning"
    elif 8 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:  # From 21 to 23
        return "Late Night"
    
    
def get_sensor_layout(dataset):
    if dataset=="aruba":
        sensor_layout = {
            "T003": {"location": "in Kitchen", "type": "Temperature"},
            "M015": {"location": "in Kitchen", "type": "Motion"},
            "M019": {"location": "in Kitchen", "type": "Motion"},
            "M017": {"location": "in Kitchen", "type": "Motion"},
            "M016": {"location": "in Kitchen", "type": "Motion"},
            "D002": {"location": "between kitchen and back door", "type": "Door"},
            "M018": {"location": "between Kitchen and Dining area", "type": "Motion"},
            "M014": {"location": "in Dining area", "type": "Motion"},
            "M013": {"location": "between Dining area and Living room", "type": "Motion"},
            "T002": {"location": "in Living room", "type": "Temperature"},
            "M012": {"location": "in Living room", "type": "Motion"},
            "M020": {"location": "in Living room", "type": "Motion"},
            "M009": {"location": "between Living room and home entrance aisle", "type": "Motion"},
            "M010": {"location": "between Living room and home entrance aisle", "type": "Motion"},
            "M011": {"location": "in home entrance aisle", "type": "Motion"},
            "M008": {"location": "in home entrance aisle", "type": "Motion"},
            "D001": {"location": "in home entrance aisle", "type": "Door"},
            "M001": {"location": "in the first bedroom", "type": "Motion"},
            "M002": {"location": "in the first bedroom", "type": "Motion"},
            "M003": {"location": "in the first bedroom", "type": "Motion"},
            "M005": {"location": "in the first bedroom", "type": "Motion"},
            "M007": {"location": "in the first bedroom", "type": "Motion"},
            "T001": {"location": "in the first bedroom", "type": "Temperature"},
            "M006": {"location": "between the first bedroom and home entrance aisle", "type": "Motion"},
            "M004": {"location": "between the first bedroom and first bathroom", "type": "Motion"},
            "M021": {"location": "in the aisle between second bathroom and dining area", "type": "Motion"},
            "T004": {"location": "in the aisle between second bathroom and dining area", "type": "Temperature"},
            "M031": {"location": "in second bathroom", "type": "Motion"},
            "D003": {"location": "in second bathroom", "type": "Door"},
            "M022": {"location": "in the aisle between second bathroom and second bedroom", "type": "Motion"},
            "M024": {"location": "in second bedroom", "type": "Motion"},
            "M023": {"location": "in second bedroom", "type": "Motion"},
            "M029": {"location": "between aisle and second bathroom", "type": "Motion"},
            "M030": {"location": "in aisle between garage door and second bathroom", "type": "Motion"},
            "D004": {"location": "on garage door", "type": "Door"},
            "M028": {"location": "between office and garage door aisle", "type": "Motion"},
            "M027": {"location": "in office", "type": "Motion"},
            "M026": {"location": "in office", "type": "Motion"},
            "M025": {"location": "in office", "type": "Motion"},
            "T005": {"location": "in office", "type": "Temperature"}
        }
    elif dataset =="milan":
        sensor_layout = {
            'M001': {'location': 'near home entrance', 'type': 'Motion'},
            'M002': {'location': 'near home entrance towards living room', 'type': 'Motion'},
            'M003': {'location': 'in dinning room', 'type': 'Motion'},
            'M004': {'location': 'in living room', 'type': 'Motion'},
            'M005': {'location': 'in living room near slider door', 'type': 'Motion'},
            'M006': {'location': 'between living room and workspace / TV room', 'type': 'Motion'},
            'M007': {'location': 'in workspace / TV room near desk', 'type': 'Motion'},
            'M008': {'location': 'in workspace / TV room near corridor', 'type': 'Motion'},
            'M009': {'location': 'in corridor near washer and dryer', 'type': 'Motion'},
            'M010': {'location': 'in corridor between dinning room and kitchen', 'type': 'Motion'},
            'M011': {'location': 'in corridor between kitchen and guest bathroom sink', 'type': 'Motion'},
            'M012': {'location': 'between dinning room and kitchen', 'type': 'Motion'},
            'M013': {'location': 'in bathroom near sink', 'type': 'Motion'},
            'M014': {'location': 'in kitchen near door', 'type': 'Motion'},
            'M015': {'location': 'in kitchen near fridge', 'type': 'Motion'},
            'M016': {'location': 'in kitchen near corridor', 'type': 'Motion'},
            'M017': {'location': 'in guest bathroom sink', 'type': 'Motion'},
            'M018': {'location': 'in toilet / shower', 'type': 'Motion'},
            'M019': {'location': 'in corridor near workspace / TV room', 'type': 'Motion'},
            'M020': {'location': 'in bedroom', 'type': 'Motion'},
            'M021': {'location': 'in bedroom on bed', 'type': 'Motion'},
            'M022': {'location': 'in kitchen near stove', 'type': 'Motion'},
            'M023': {'location': 'in kitchen', 'type': 'Motion'},
            'M024': {'location': 'in guest bedroom', 'type': 'Motion'},
            'M025': {'location': 'in walk-in closet', 'type': 'Motion'},
            'M026': {'location': 'in workspace / TV room', 'type': 'Motion'},
            'M027': {'location': 'in living room', 'type': 'Motion'},
            'M028': {'location': 'in bedroom', 'type': 'Motion'},
            'T001': {'location': 'in kitchen near stove', 'type': 'Temperature'},
            'T002': {'location': 'in corridor near guest bathroom sink', 'type': 'Temperature'},
            'D001': {'location': 'on home entrance door', 'type': 'Door'},
            'D002': {'location': 'on coat cabinet near home entrance door', 'type': 'Door'},
            'D003': {'location': 'in kitchen', 'type': 'Door'}
            }
    elif dataset=='cairo':
        sensor_layout={
            'M001': {'location': 'near work area in office', 'type': 'Motion'},
            'M002': {'location': 'in corridor near bedroom and guest bedroom', 'type': 'Motion'},
            'M003': {'location': 'in corridor near stairs', 'type': 'Motion'},
            'M004': {'location': 'in guest bedroom', 'type': 'Motion'},
            'M005': {'location': 'in bedroom', 'type': 'Motion'},
            'M006': {'location': 'in bedroom near its entrance', 'type': 'Motion'},
            'M007': {'location': 'in bedroom', 'type': 'Motion'},
            'M008': {'location': 'in bedroom near bed', 'type': 'Motion'},
            'M009': {'location': 'in bedroom near bed', 'type': 'Motion'},
            'M010': {'location': 'near top of stairs', 'type': 'Motion'},
            'M011': {'location': 'in living room near bottom of stairs', 'type': 'Motion'},
            'M012': {'location': 'in kitchen', 'type': 'Motion'},
            'M013': {'location': 'near couch in living room', 'type': 'Motion'},
            'M014': {'location': 'in living room near stairs', 'type': 'Motion'},
            'M015': {'location': 'in living room near entrance door', 'type': 'Motion'},
            'M016': {'location': 'in living room', 'type': 'Motion'},
            'M017': {'location': 'near couch in living room', 'type': 'Motion'},
            'M018': {'location': 'in living room', 'type': 'Motion'},
            'M019': {'location': 'in kitchen', 'type': 'Motion'},
            'M020': {'location': 'in dinning area near kitchen', 'type': 'Motion'},
            'M021': {'location': 'near medicine cabinet in kitchen', 'type': 'Motion'},
            'M022': {'location': 'in kitchen', 'type': 'Motion'},
            'M023': {'location': 'in living room', 'type': 'Motion'},
            'M024': {'location': 'in kitchen', 'type': 'Motion'},
            'M025': {'location': 'in other room near garage', 'type': 'Motion'},
            'M026': {'location': 'in laundry room near garage', 'type': 'Motion'},
            'M027': {'location': 'near garage door', 'type': 'Motion'},
            'T001': {'location': 'in bedroom', 'type': 'Temperature'},
            'T002': {'location': 'in office near work area', 'type': 'Temperature'},
            'T003': {'location': 'in living room near stairs', 'type': 'Temperature'},
            'T004': {'location': 'near medicine cabinet in kitchen', 'type': 'Temperature'},
            'T005': {'location': 'in living room', 'type': 'Temperature'}
            }

    elif dataset=='kyoto7':
        sensor_layout = {
        "M01": {"location": "living room", "type": "Motion"},
        "M02": {"location": "living room", "type": "Motion"},
        "M03": {"location": "living room", "type": "Motion"},
        "M04": {"location": "living room", "type": "Motion"},
        "M05": {"location": "living room", "type": "Motion"},
        "M06": {"location": "living room", "type": "Motion"},
        "M07": {"location": "living room", "type": "Motion"},
        "M08": {"location": "living room", "type": "Motion"},
        "M09": {"location": "dining room", "type": "Motion"},
        "M10": {"location": "dining room", "type": "Motion"},
        "M11": {"location": "living room", "type": "Motion"},
        "M12": {"location": "living room", "type": "Motion"},
        "M13": {"location": "living room", "type": "Motion"},
        "M14": {"location": "living room", "type": "Motion"},
        "M15": {"location": "living room", "type": "Motion"},
        "M16": {"location": "kitchen", "type": "Motion"},
        "M17": {"location": "kitchen", "type": "Motion"},
        "M18": {"location": "kitchen", "type": "Motion"},
        "M19": {"location": "kitchen", "type": "Motion"},
        "M20": {"location": "kitchen", "type": "Motion"},
        "M21": {"location": "home entrance aisle", "type": "Motion"},
        "M22": {"location": "home entrance aisle", "type": "Motion"},
        "M23": {"location": "home entrance aisle", "type": "Motion"},
        "M24": {"location": "home entrance aisle", "type": "Motion"},
        "M25": {"location": "home entrance aisle", "type": "Motion"},
        "M26": {"location": "stairs near home entrance aisle", "type": "Motion"},
        "M27": {"location": "aisle near bedroom", "type": "Motion"},
        "M28": {"location": "aisle near bedroom", "type": "Motion"},
        "M29": {"location": "aisle near bedroom", "type": "Motion"},
        "M30": {"location": "between aisle and first bedroom", "type": "Motion"},
        "M31": {"location": "first bedroom", "type": "Motion"},
        "M32": {"location": "first bedroom", "type": "Motion"},
        "M33": {"location": "first bedroom", "type": "Motion"},
        "M34": {"location": "first bedroom", "type": "Motion"},
        "M35": {"location": "first bedroom", "type": "Motion"},
        "M36": {"location": "first bedroom", "type": "Motion"},
        "M37": {"location": "between aisle and bathroom", "type": "Motion"},
        "M38": {"location": "bathroom", "type": "Motion"},
        "M39": {"location": "bathroom", "type": "Motion"},
        "M40": {"location": "bathroom", "type": "Motion"},
        "M41": {"location": "bathroom", "type": "Motion"},
        "M42": {"location": "between aisle and work area", "type": "Motion"},
        "M43": {"location": "between aisle and second bedroom", "type": "Motion"},
        "M44": {"location": "between aisle and second bedroom", "type": "Motion"},
        "M45": {"location": "second bedroom", "type": "Motion"},
        "M46": {"location": "second bedroom", "type": "Motion"},
        "M47": {"location": "second bedroom", "type": "Motion"},
        "M48": {"location": "second bedroom", "type": "Motion"},
        "M49": {"location": "second bedroom", "type": "Motion"},
        "M50": {"location": "second bedroom", "type": "Motion"},
        "M51": {"location": "kitchen", "type": "Motion"},
        "D01": {"location": "front door", "type": "Door"},
        "D02": {"location": "back door", "type": "Door"},
        "D03": {"location": "door of first bedroom", "type": "Door"},
        "D04": {"location": "door of second bedroom", "type": "Door"},
        "D05": {"location": "bathroom door", "type": "Door"},
        "D06": {"location": "bathroom door", "type": "Door"},
        "D07": {"location": "kitchen cabinet door", "type": "Door"},
        "D08": {"location": "refrigerator door", "type": "Door"},
        "D09": {"location": "fridge door", "type": "Door"},
        "D10": {"location": "microwave door", "type": "Door"},
        "D11": {"location": "kitchen pantry door", "type": "Door"},
        "D12": {"location": "home entrance closet door", "type": "Door"},
        "D13": {"location": "living room TV door", "type": "Door"},
        "D14": {"location": "kitchen cabinet door", "type": "Door"},
        "D15": {"location": "kitchen cabinet door", "type": "Door"},
        "D16": {"location": "kitchen cabinet door", "type": "Door"},
        "I03": {"location": "living room TV shelf", "type": "Item"},
        "L04": {"location": "living space", "type": "Light switch"},
        "L06": {"location": "stairs", "type": "Light switch"},
        "L09": {"location": "workplace", "type": "Light switch"},
        "L10": {"location": "first bedroom", "type": "Light switch"},
        "L11": {"location": "bathroom", "type": "Light switch"},
        "L12": {"location": "bathroom", "type": "Light switch"},
        "L13": {"location": "bathroom", "type": "Light switch"},
        "AD1-A": {"location": "kitchen", "type": "Burner"},
        "AD1-B": {"location": "kitchen", "type": "Hot water"},
        "AD1-C": {"location": "kitchen", "type": "Cold water"},
    }

    else:
        print(f"no sensor map for  {dataset}")
        exit(0)

    return sensor_layout