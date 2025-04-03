import os
import joblib

# Path to the folder containing .joblib files
folder_path = '/mnt/attached1/TDOST/TDOST/gpt/combine'

# List all .joblib files in the folder
joblib_files = [f for f in os.listdir(folder_path) if f.endswith('.joblib')]

# Initialize an empty dictionary to hold the combined data
combined_dict = {}

# Load each .joblib file and combine the dictionaries
for file_name in joblib_files:
    print(file_name)
    file_path = os.path.join(folder_path, file_name)
    # Load the dictionary from the .joblib file
    data = joblib.load(file_path)
    # print(len(data.keys()))
    print(data)
    
    
    # Combine the dictionary with the existing combined_dict
    combined_dict.update(data)
    
combined_dict["('Wednesday', 'Morning', 'Motion', 'in dining area near kitchen', 'OFF')"]=  ['On Wednesday morning, the motion sensor in the dining area near the kitchen registered no movement, indicating the space was unoccupied.', 'The absence of activity was detected on Wednesday morning by the motion sensor situated in the dining area adjacent to the kitchen, as it remained off.', "The dining area near the kitchen remained still and undisturbed on Wednesday morning, with the motion sensor indicating an 'OFF' status."]
combined_dict["('Wednesday', 'Morning', 'Motion', 'in dining area near kitchen', 'ON')"] = ['As the sun rose on Wednesday morning, the motion sensor in the dining area near the kitchen sprung to life, detecting movement within the space.', "The motion sensor, located in the dining area next to the kitchen, was triggered 'ON' on Wednesday morning, signaling that someone might be starting their day.", "Activity was detected on Wednesday morning in the dining area close to the kitchen, with the motion sensor switching to the 'ON' position, possibly as the household prepares for breakfast."]
    
joblib.dump(combined_dict, "/mnt/attached1/TDOST/TDOST/gpt/kyoto7_gpt_generated_v1.joblib")
print(len(combined_dict.keys()))
# combined_dict now contains the combined data from all .joblib files
