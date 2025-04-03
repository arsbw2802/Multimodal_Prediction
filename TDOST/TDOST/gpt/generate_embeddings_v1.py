import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import socket
import os
import joblib
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print(current_dir, parent_dir)
sys.path.append(parent_dir)

from utils import *

def parse_arguments():

    parser = argparse.ArgumentParser(description='arguments for smart home dataset preprocessing')
    parser.add_argument('--datasets', type=str, nargs='+', default=['milan', 'aruba', 'cairo', 'kyoto7'], help='name(s) of the dataset(s)')
    parser.add_argument('--context_length', type=int, default=100, help='context length')
    parser.add_argument('--sentence_encoder_name', type=str, default='sentence-t5-base',  choices=["all-MiniLM-L6-v2", "all-distilroberta-v1",'sentence-t5-base'], help='name of the sentence encoder')
    parser.add_argument('--embedding_dimension', type=int, default=768, help='context length')
    parser.add_argument('--npy_path',  type=str, default='/coc/pcba1/mthukral3/gt/TDOST/npy/pre-segmented', help='path of the pre-processd data files')

    args = parser.parse_args()

    return args

def get_gpt_senstences():
    dict = joblib.load("gpt_generated_sent_casas.joblib")
    new_dict = {}

    for key, sent in dict.items():
        # print(key)
        key_split = key.strip().replace("'", "").replace("(", "").replace(")", "").split(',')
        key_split = [part.strip().replace(" ", "") for part in key_split]

        new_key = "_".join(key_split).replace("dinning", "dining")

        
        new_dict[new_key]=sent

        
    return new_dict






def generate_embeddings(dataset, model,  args):
    

    gpt_generated_sentences =  get_gpt_senstences()
    



    # print(f"gpt generated sentences {gpt_generated_sentences}")



    sensor_map = get_sensor_layout(dataset)

    

    data_sensor = np.load(os.path.join(args.npy_path, "{}-x_sensor.npy".format(dataset)), allow_pickle=True)
    data_value = np.load(os.path.join(args.npy_path,"{}-x_value.npy".format(dataset)), allow_pickle=True)
    data_time = np.load(os.path.join(args.npy_path,"{}-x_time.npy".format(dataset)), allow_pickle=True)
    labels = np.load(os.path.join(args.npy_path,"{}-global_labels.npy".format(dataset)), allow_pickle=True)
    number_of_gpt_sent = 3
    
    emb = np.zeros(( number_of_gpt_sent, labels.shape[0], args.context_length, args.embedding_dimension))
    emb_dict = {}
    
    missing_keys = []

    for i in range(labels.shape[0]):
      

        if i % 500 == 0:
            print("Encoding Data Point No:\t" + str(i))

        # This is for a single data point which is a sequence of a number of sensor readings
        sensor_seq = data_sensor[i]
        value_seq = data_value[i]
        time_seq = data_time[i]

        for j in range(min(len(sensor_seq), args.context_length)):

            # For a single sensor reading
            sensor_reading = sensor_seq[j]
            value_reading = value_seq[j]
            time_reading = time_seq[j].time().strftime("%H:%M:%S")

            if j == 0:
                prev_reading = time_reading

            time_sentence = process_time_v11(time_reading, j, prev_reading)
            
            time_of_day = get_time_period(time_seq[j].hour).strip().replace(" ", "")
            day_of_week = time_seq[j].strftime('%A').strip().replace(" ", "")
            sensor_location = sensor_map[sensor_reading]["location"].strip().replace(" ", "").replace("dinning", "dining")
            sensor_type = sensor_map[sensor_reading]["type"].strip().replace(" ", "")
            
            sensor_key = f'{day_of_week}_{time_of_day}_{sensor_type}_{sensor_location}_{value_reading}'

    
            if sensor_key not in gpt_generated_sentences:
                print(sensor_key)
                missing_keys.append(sensor_key)
                continue

            # if sensor_key not in gpt_generated_sentences:
            #     sensor_key = f"('{day_of_week}', '{time_of_day}', '{sensor_type}', '{sensor_location}',  '{value_reading}')"
            #     if sensor_key not in gpt_generated_sentences:
            #         print(sensor_key)


            
            
            list_of_sent = gpt_generated_sentences[sensor_key]
            # print(sensor_reading, value_reading, time_reading, sensor_key, list_of_sent)
            for sent_no, sent in enumerate(list_of_sent):
                final_sentence = sent
                
                # print(final_sentence)
                if final_sentence in emb_dict:
                    emb[sent_no, i, args.context_length - 1- j, :] = emb_dict[final_sentence]
                else:
                    sentence_embeddings = model.encode(final_sentence)
                    emb[sent_no, i, args.context_length - 1- j, :] = sentence_embeddings
                    emb_dict.update({final_sentence: sentence_embeddings})

     
    emb = emb.reshape(emb.shape[0]* emb.shape[1], emb.shape[2], emb.shape[3])
    np.save(os.path.join(args.npy_path, "{}_embeddings_gpt_v1_{}".format(dataset, args.sentence_encoder_name)), emb)
    # joblib.dump(missing_keys, f"{dataset}_missing.joblib")




if __name__=="__main__":
    
    args = parse_arguments()
            
    model_name = args.sentence_encoder_name
    model = SentenceTransformer(model_name)

    for dataset in args.datasets:
        generate_embeddings(dataset, model, args)
    