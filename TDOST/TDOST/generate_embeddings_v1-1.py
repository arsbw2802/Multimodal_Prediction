import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import inflect
import os
from datetime import datetime
import math
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_arguments():

    parser = argparse.ArgumentParser(description='arguments for smart home dataset preprocessing')
    parser.add_argument('--datasets', type=str, nargs='+', default=['milan', 'aruba', 'cairo', 'kyoto7'], help='name(s) of the dataset(s)')
    parser.add_argument('--context_length', type=int, default=100, help='context length')
    parser.add_argument('--sentence_encoder_name', type=str, default='sentence-t5-base', help='name of the sentence encoder')
    parser.add_argument('--embedding_dimension', type=int, default=768, help='context length')
    parser.add_argument('--npy_path',  type=str, default='/coc/pcba1/mthukral3/gt/TDOST/npy/pre-segmented', help='path of the pre-processd data files')

    args = parser.parse_args()

    return args


def open_raw_data_file(path):

    data = np.load(path, allow_pickle=True)

    return data

def convert_num_to_text(num):

    p = inflect.engine()
    word = p.number_to_words(num)

    return word

def prepare_month(text_month):
    month_mapping = {
        "eleven": "november",
        "twelve": "december",
        "one": "january",
        "two": "february",
        "three": "march",
        "four": "april",
        "five": "may",
        "six": "june",
        "seven": "july",
        "eight": "august",
        "nine": "september",
        "ten": "october"
    }
    
    return month_mapping.get(text_month)

def prepare_date_sentence(text_day, text_month):
    proper_text_month = prepare_month(text_month)
    return f" on {text_day} {proper_text_month}"


def process_date(raw_date):
    _, month, day = raw_date.split("-")  # Unpack and ignore the year

    text_day = convert_num_to_text(day)
    text_month = prepare_month(month)  # Directly convert the month using prepare_month

    return prepare_date_sentence(text_day, text_month)


def process_time(raw_time, j, prev_reading=None):
    current_hour, current_minute, *current_second_data = raw_time.split(":")
    am_pm_flag = "AM" if int(current_hour) < 12 else "PM"
    
    text_hour = convert_num_to_text(current_hour)
    text_minute = convert_num_to_text(current_minute)

    if j == 0:
        return f" at {text_hour} hours {text_minute} minutes {am_pm_flag}"
    
    current_second = current_second_data[0].split(".")[0] if current_second_data else "00"

    prev_hour, prev_minute, prev_second_data = prev_reading.split(":")
    prev_second = prev_second_data.split(".")[0]

    current_t = f"{current_hour}:{current_minute}:{current_second}"
    prev_t = f"{prev_hour}:{prev_minute}:{prev_second}"

    delta = datetime.strptime(current_t, "%H:%M:%S") - datetime.strptime(prev_t, "%H:%M:%S")
    time_diff = convert_num_to_text(math.floor(delta.total_seconds()))

    return f"After {time_diff} seconds, "


def convert_sensor_type_to_text(sensor_type):
    sensor_mapping = {
        "M": "Motion sensor ",
        "D": "Door sensor ",
        "T": "Temperature sensor ",
        "I": "Item sensor ",
        "L": "Light switch sensor ",
        "AD1-A": "Burner sensor ",
        "AD1-B": "Hot water sensor ",
        "AD1-C": "Cold water sensor "
    }
    
    # Default to "Motion sensor" if the sensor_type is not found in the dictionary
    return sensor_mapping.get(sensor_type, "Motion sensor ")

def get_kyoto7_sensor_global_context(raw_sensor):
    sensor_mapping = {
        ("M01",): "between home entrance aisle and living room near TV shelf",
        ("M02",): "in living room near TV shelf",
        ("M03", "M04", "M05", "M11", "M13", "M14"): "in living room",
        ("M06", "M07"): "in living room near couch",
        ("M08",): "in living room near hall",
        ("M09", "M10"): "in dining room",
        ("M12",): "in living room near back door",
        ("M15",): "in living room near kitchen",
        ("M16",): "in kitchen near living room",
        ("M17",): "in kitchen",
        ("M18",): "in kitchen near sink",
        ("M19",): "in kitchen near kitchen closet",
        ("M20",): "in kitchen inside kitchen closet",
        ("M21", "M22", "M25"): "in home entrance aisle",
        ("M23",): "in home entrance aisle near living room",
        ("M24",): "in home entrance aisle near front door",
        ("M26",): "on stairs near home entrance aisle",
        ("M27", "M28", "M29"): "in aisle near first bedroom",
        ("M30",): "between aisle and first bedroom",
        ("M31", "M32"): "in first bedroom near desk",
        ("M33", "M36"): "in first bedroom",
        ("M34", "M35"): "in first bedroom near bed",
        ("M37",): "between aisle and bathroom",
        ("M38",): "in bathroom near basin",
        ("M39",): "in bathroom near bath",
        ("M40",): "in bathroom",
        ("M41",): "in bathroom near toilet",
        ("M42",): "between aisle and work area",
        ("M43", "M44"): "between aisle and second bedroom",
        ("M45", "M50"): "in second bedroom",
        ("M46", "M47"): "in second bedroom near bed",
        ("M48", "M49"): "in second bedroom near desk",
        ("M51",): "in kitchen inside pantry",
        ("D01",): "on front door",
        ("D02",): "on back door",
        ("D03",): "on the door of first bedroom",
        ("D04",): "on the door of second bedroom",
        ("D05",): "on bathroom door",
        ("D06",): "on toilet door inside bathroom",
        ("D07", "D14", "D15", "D16"): "on kitchen cabinet door",
        ("D08",): "on refrigerator door",
        ("D09",): "on fridge door",
        ("D10",): "on microwave door",
        ("D11",): "on kitchen pantry door",
        ("D12",): "on home entrance closet door",
        ("D13",): "on living room TV door",
        ("I03",): "on living room TV shelf",
        ("L04",): "for living space",
        ("L06",): "for stairs near home entrance",
        ("L09",): "for workplace",
        ("L10",): "for first bedroom",
        ("L11",): "for bathroom",
        ("L12",): "for shower in bathroom",
        ("L13",): "for shower fan in bathroom",
        ("AD1-A",): "in kitchen near stove",
        ("AD1-B", "AD1-C"): "in kitchen near sink"
    }

    # Iterate through the dictionary and return the context if the raw_sensor matches
    for sensor_codes, context in sensor_mapping.items():
        if raw_sensor in sensor_codes:
            return context

    return None  # Or a default context if no match is found

def get_aruba_sensor_global_context(raw_sensor):
    sensor_mapping = {
        ("T003", "M015", "M019", "M017", "M016"): "in Kitchen",
        ("D002",): "between kitchen and back door",
        ("M018",): "between Kitchen and Dining area",
        ("M014",): "in Dining area",
        ("M013",): "between Dining area and Living room",
        ("T002", "M012", "M020"): "in Living room",
        ("M009", "M010"): "between Living room and home entrance aisle",
        ("M011", "M008", "D001"): "in home entrance aisle",
        ("M001", "M002", "M003", "M005", "M007", "T001"): "in the first bedroom",
        ("M006",): "between the first bedroom and home entrance aisle",
        ("M004",): "between the first bedroom and first bathroom",
        ("M021", "T004"): "in the aisle between second bathroom and dining area",
        ("M031", "D003"): "in second bathroom",
        ("M022",): "in the aisle between second bathroom and second bedroom",
        ("M024", "M023"): "in second bedroom",
        ("M029",): "between aisle and second bathroom",
        ("M030",): "in aisle between garage door and second bathroom",
        ("D004",): "on garage door",
        ("M028",): "between office and garage door aisle",
        ("M027", "M026", "M025", "T005"): "in office"
    }

    for sensor_codes, context in sensor_mapping.items():
        if raw_sensor in sensor_codes:
            return context
    
    return None  # Or a default context if needed

def get_milan_sensor_global_context(raw_sensor):
    sensor_mapping = {
        ("M001",): "near home entrance",
        ("M002",): "near home entrance towards living room",
        ("M003",): "in dining room",
        ("M004",): "in living room",
        ("M005",): "in living room near slider door",
        ("M006",): "between living room and workspace / TV room",
        ("M007",): "in workspace / TV room near desk",
        ("M008",): "in workspace / TV room near corridor",
        ("M009",): "in corridor near washer and dryer",
        ("M010",): "in corridor between dining room and kitchen",
        ("M011",): "in corridor between kitchen and guest bathroom sink",
        ("M012",): "between dining room and kitchen",
        ("M013",): "in bathroom near sink",
        ("M014",): "in kitchen near door",
        ("M015",): "in kitchen near fridge",
        ("M016",): "in kitchen near corridor",
        ("M017",): "in guest bathroom sink",
        ("M018",): "in toilet / shower",
        ("M019",): "in corridor near workspace / TV room",
        ("M020", "M021", "M028"): "in bedroom",
        ("M022", "M023", "T001"): "in kitchen near stove",
        ("M024",): "in guest bedroom",
        ("M025",): "in walk-in closet",
        ("M026",): "in workspace / TV room",
        ("M027",): "in living room",
        ("T002",): "in corridor near guest bathroom sink",
        ("D001",): "on home entrance door",
        ("D002",): "on coat cabinet near home entrance door",
        ("D003",): "in kitchen"
    }

    for sensor_codes, context in sensor_mapping.items():
        if raw_sensor in sensor_codes:
            return context
    
    return None  # Or a default context if needed


def get_cairo_sensor_global_context(raw_sensor):
    sensor_mapping = {
        ("M001",): "near work area in office",
        ("M002",): "in corridor near bedroom and guest bedroom",
        ("M003",): "in corridor near stairs",
        ("M004",): "in guest bedroom",
        ("M005", "M007", "M008", "M009", "T001"): "in bedroom",
        ("M006",): "in bedroom near its entrance",
        ("M010",): "near top of stairs",
        ("M011",): "in living room near bottom of stairs",
        ("M012", "M019", "M022", "M024"): "in kitchen",
        ("M013", "M017"): "near couch in living room",
        ("M014", "M016", "M018", "M023", "T003", "T005"): "in living room",
        ("M015",): "in living room near entrance door",
        ("M020",): "in dining area near kitchen",
        ("M021", "T004"): "near medicine cabinet in kitchen",
        ("M025",): "in other room near garage",
        ("M026",): "in laundry room near garage",
        ("M027",): "near garage door",
        ("T002",): "in office near work area"
    }

    for sensor_codes, context in sensor_mapping.items():
        if raw_sensor in sensor_codes:
            return context
    
    return None  # Or a default context if needed

def get_MARBLE_sensor_global_context(raw_sensor):
    sensor_mapping = {
        # synth:
        ("M811212",): "in hall",
        ("M1192038514",): "in kitchen",
        ("M4914914718151513",): "in dining_room",
        ("M135493914511851",): "in medicine_area",
        ("M12922914718151513",): "in living_room",
        ("M1566935",): "in office",
        ("M152120",): "in out",
        ("T001",): "This does not exist", #TODO: idk how to handle this lol


        # real:
        # magnetic 
        ("MR1",): "using  pantry",
        ("MR2",): "using cutlery drawer",
        ("MR5",): "using pots drawer",
        ("MR6",): "using medicines cabinet",
        ("MR7",): "using fridge",
        
        # smart plug
        ("ME1",): "using stove plug",
        ("ME2",): "using television plug",

        # pressure mat
        ("MP1",): "on dining room chair",
        ("MP2",): "on office chair",
        ("MP3",): "on living room couch",
        ("MP4",): "on dining room chair",
        ("MP5",): "on dining room chair",
        ("MP6",): "on dining room chair",
        ("MP7",): "on living room couch",
        ("MP8",): "on living room couch",
        ("MP9",): "on living room couch",
        
        # IMU sensors
        ("WA1",): "on user's hand (acceleration)",
        ("WG1",): "on user's hand (gyroscope)",
        ("WO1",): "on user's hand (magnetometer)",
        ("WB1",): "on user's hand (barometer)",

    }

    for sensor_codes, context in sensor_mapping.items():
        if raw_sensor in sensor_codes:
            return context
    
    return None  # Or a default context if needed

def prepare_sensor_sentence(sensor_text, sensor_context):

    sentence = sensor_text + sensor_context

    return sentence

def process_sensor(raw_sensor, args):

    sensor_type = raw_sensor[0]
    sensor_text = convert_sensor_type_to_text(sensor_type)

    if args.dataset_name == "aruba":
        sensor_context = get_aruba_sensor_global_context(raw_sensor)
    elif args.dataset_name == "milan":
        sensor_context = get_milan_sensor_global_context(raw_sensor)
    elif args.dataset_name == "cairo":
        sensor_context = get_cairo_sensor_global_context(raw_sensor)
    elif args.dataset_name == "kyoto7":
        sensor_context = get_kyoto7_sensor_global_context(raw_sensor)
    
    elif "MARBLE" in args.dataset_name:
        sensor_context = get_MARBLE_sensor_global_context(raw_sensor)


    sensor_sentence = prepare_sensor_sentence(sensor_text, sensor_context)

    return sensor_sentence

def prepare_value_sentence(filler_text, value_text):
    sentence = filler_text + value_text

    return sentence

def process_value(raw_value):
    filler_text = " fired with value "
    
    # Extended dictionary for additional value mappings
    value_mapping = {
        "ON": "ON", "ON5": "ON", "ONc": "ON", "ON55": "ON", "ON0": "ON", "ONM026": "ON", "ONM009": "ON", "ONM024": "ON", 'ON`': "ON",
        "OFF": "OFF", "OFF5": "OFF", "OFFc": "OFF", "OF": "OFF", "OFF0": "OFF", "OFFM026": "OFF", "OFFM009": "OFF", "OFFM024": "OFF",
        "OPEN": "OPEN", "OPEN5": "OPEN", "OPENc": "OPEN", "OPN": "OPEN", "OPENED": "OPEN",
        "CLOSE": "CLOSE", "CLOSE5": "CLOSE", "CLOSEc": "CLOSE", "CLOSED": "CLOSE", "CLS": "CLOSE",
        "PRESENT": "PRESENT", "ABSENT": "ABSENT",

        
        'no_hange': "no_hange",
        'low':      "low",
        'medium':   "medium",
        'high':     "high",
        'extreme':  "extreme",
    }

    if raw_value in value_mapping:
        value_text = value_mapping[raw_value]
    elif raw_value == "O":
        value_text = convert_num_to_text(0)
    else:
        value_text = convert_num_to_text(int(round(float(raw_value))))

    value_sentence = prepare_value_sentence(filler_text, value_text)
    
    return value_sentence


def initialize_sentence_encoder(model_name):
    model = SentenceTransformer(model_name)

    return model

def create_encoded_data_file(filename):
    encoded_data_file = open(filename, "a")

    return encoded_data_file

def set_activity_flag(activity_name):
    final_activity_name = activity_name # Do nothing for now; Add activity semantics later!

    return final_activity_name



def generate_embeddings(dataset, model, args):

    args.dataset_name = dataset

    data_sensor = open_raw_data_file(f"{args.npy_path}/{dataset}-x_sensor.npy")
    data_value = open_raw_data_file(f"{args.npy_path}/{dataset}-x_value.npy")
    data_time = open_raw_data_file(f"{args.npy_path}/{dataset}-x_time.npy")
    labels = open_raw_data_file(f"{args.npy_path}/{dataset}-global_labels.npy")


    emb = np.zeros((labels.shape[0], args.context_length, args.embedding_dimension))
    emb_dict = {}

    for i in range(labels.shape[0]):

        if i % 500 == 0:
            print("Encoding Data Point No:\t" + str(i))

        # This is for a single data point which is a sequence of a number of sensor readings
        sensor_seq = data_sensor[i]
        value_seq = data_value[i]
        time_seq = data_time[i]
            # We need to pad the embeddings (since we initialize emb with zero, it takes care by itself)

        for j in range(min(len(sensor_seq), args.context_length)):

            # For a single sensor reading
            sensor_reading = sensor_seq[j]
            value_reading = value_seq[j]
            time_reading = time_seq[j].time().strftime("%H:%M:%S")

            if j == 0:
                prev_reading = time_reading

            # Form a sentence
            sensor_sentence = process_sensor(sensor_reading, args)
            value_sentence = process_value(str(value_reading))
            time_sentence = process_time(time_reading, j, prev_reading)

            # Join
            if j == 0:
                final_sentence = sensor_sentence + value_sentence + time_sentence
            else:
                final_sentence = time_sentence + sensor_sentence + value_sentence

            # print(final_sentence)
            # Get embeddings
            if final_sentence in emb_dict:
                emb[i, j, :] = emb_dict[final_sentence]
            else:
                sentence_embeddings = model.encode(final_sentence)
                emb[i, j, :] = sentence_embeddings
                emb_dict.update({final_sentence: sentence_embeddings})

            ### updating prev reading
            prev_reading = time_reading

    np.save(f"{args.npy_path}/{dataset}_embeddings_v1-1_{args.sentence_encoder_name}_3000", emb)

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

    args = parse_arguments()
    print(args)

    model_name = args.sentence_encoder_name
    model = initialize_sentence_encoder(model_name)
    # datasets = args.datasets

    datasets = args.datasets

    for dataset in datasets:

        print("Generating Embeddings for Dataset:\t" + str(dataset))
        generate_embeddings(dataset, model, args)