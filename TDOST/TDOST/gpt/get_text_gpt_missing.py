from openai import AzureOpenAI
import json
import numpy as np
import socket
import os
import sys
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print(current_dir, parent_dir)
sys.path.append(parent_dir)

from utils import *






def generate_text_for_sensor_trigger(client, sensor_reading):
        
    message_text = [
        {"role":"system",
         "content":"You are an AI asistant that is helping in generating diverse texts and adding a context to each sensor readings leveraging world knowledge"},
        {"role":"user",
         "content": '''
         Please generate diverse text sentences (3) for the each  sensor trigger. You will be given a window of 5 sensor triggers. 
    
         The given sensor trigger has format: (Day of Week, Time in that day, sensor type, location context of the sensor,  Value of the Sensor). 
         The output should be a json (key : (Day of Week, Time in that day, sensor type, location context of the sensor,  Value of the Sensor) ) containing list with the generated sentences. 
         Sensor Trigger Window {}
                     
         '''.format( sensor_reading)
        }
    ]
    # print(message_text)
    
    # assert 1==0
    
    completion = client.chat.completions.create(
        model="gpt-4",
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
    
    npy_path = get_npy_path()
    client = AzureOpenAI(
        azure_endpoint = "https://activity-description-generator.openai.azure.com/", 
        api_key="44e61450aa5940fa9cd87d0455ee0b80",  
        api_version="2023-07-01-preview"
        )
    
    dataset = "kyoto7"
    
    sr_dict = joblib.load(f"/mnt/attached1/TDOST/TDOST/gpt/{dataset}_missing.joblib")
                
    keys = list(set(sr_dict))

  
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
        
        joblib.dump(chat_text, "{}_gpt_generated_new.joblib".format(dataset))
        
        
        
    


