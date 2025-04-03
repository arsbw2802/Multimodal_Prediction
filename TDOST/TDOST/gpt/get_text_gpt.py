# from openai import AzureOpenAI
import json
import numpy as np
import socket
import os
import sys
import joblib
from collections import Counter
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print(current_dir, parent_dir)
sys.path.append(parent_dir)

from utils import *

from openai import OpenAI





def generate_text_for_sensor_trigger(client, sensor_reading):
        
    message_text = [
        {"role":"system",
         "content":'''You are an AI assistant responsible for generating diverse text descriptions and contextual information for each sensor reading, using relevant world knowledge. Respond clearly and formally, avoiding poetic language, figurative expressions, or abstract phrasing.'''
        },
        {"role":"user",
         "content": '''
            Generate 3 diverse text sentences for each sensor trigger in a given window of 5 triggers. The sensor trigger format is: (Day of Week, Time, Sensor Type, Location, Sensor Value). Output the results as a JSON object where the key is the sensor trigger (Day of Week, Time, Sensor Type, Location, Sensor Value) and the value is a list of generated sentences.
            Sensor Trigger Window: {}             
        '''.format( sensor_reading)
        }
    ]
    print(message_text)
    
 
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={ "type": "json_object"},
        messages = message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        seed=42
    )

    return completion



    

if __name__=="__main__":
    
    npy_path = "/coc/pcba1/mthukral3/gt/TDOST/npy/pre-segmented"
    client = OpenAI(
        # organization='org-H2k5NXsKmjmzz9Y9qOn5gxzI',
        # project='proj_pU1P93yEvGKdgOn2gKjmpjTr'
    )

    datasets = ["kyoto7", "aruba", "milan", "cairo"]
    sr_dict={}
    for dataset in datasets:
        
        
        sensor_to_key_map ={}

        sensor_map = get_sensor_layout(dataset)
        
        sensor_data = np.load(os.path.join(npy_path, "{}-x_sensor.npy".format(dataset)), allow_pickle=True)
        time_data = np.load(os.path.join(npy_path,"{}-x_time.npy".format(dataset)), allow_pickle=True)
        value_data = np.load(os.path.join(npy_path,"{}-x_value.npy".format(dataset)), allow_pickle=True)
        
        for time_data_window, sensor_data_window, value_data_window in zip( time_data, sensor_data, value_data):
            sensor_reading_window=  list(zip(time_data_window, sensor_data_window, value_data_window))
            
            for sensor_reading in sensor_reading_window:
                time_of_day = get_time_period(sensor_reading[0].hour)
                day_of_week = sensor_reading[0].strftime('%A')
                sensor_location = sensor_map[sensor_reading[1]]["location"]
                sensor_type = sensor_map[sensor_reading[1]]["type"]
                
                sensor_reading = (day_of_week, time_of_day, sensor_type, sensor_location,  sensor_reading[2])
                print(sensor_reading)
                if sensor_reading in sr_dict:
                    sr_dict[sensor_reading]+=1
                else:
                    sr_dict[sensor_reading]=1

    assert 1==0
    joblib.dump(sr_dict, "/coc/pcba1/mthukral3/gt/TDOST/TDOST/gpt/sr_dict_casas_datasets.joblib")     
    keys = list(sr_dict.keys())

    print(keys)

    keys = list(joblib.load("/coc/pcba1/mthukral3/gt/TDOST/TDOST/gpt/sr_dict_casas_datasets.joblib").keys())

    # curl https://api.openai.com/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer $OPENAI_API_KEY"   -d '{
    #  "model": "gpt-4o-mini",
    #  "messages": [{"role": "user", "content": "Please generate diverse text sentences (3) for the each  sensor trigger. You will be given a window of 5 sensor triggers. The given sensor trigger has format: (Day of Week, Time in that day, sensor type, location context of the sensor,  Value of the Sensor). The output should be a json (key : (Day of Week, Time in that day, sensor type, location context of the sensor,  Value of the Sensor) ) containing list with the generated sentences. Sensor Trigger Window (Monday, Early Morning, Motion, first bedroom, ON)"}]
    # }'
  
   
    chat_text ={}
    i = 0
    print(len(keys))
    while i < len(keys):
        print("i: ",i)
        sr = keys[i:i+5]
        
        response = generate_text_for_sensor_trigger(client, sr)
        dict_response = json.loads(response.choices[0].message.content)
        
        for key in dict_response:
            chat_text[key] = dict_response[key]
        i = i +5 
        
        joblib.dump(chat_text, "gpt_generated_sent_casas.joblib")
        
        
        
        
    


