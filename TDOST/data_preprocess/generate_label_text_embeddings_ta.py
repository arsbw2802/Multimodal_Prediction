
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from gutils import *
from sentence_transformers import SentenceTransformer

npy_path = get_npy_path()

def generate_text_embeddings(dataset_name):
    

    labels = open_raw_data_file(os.path.join(npy_path,  dataset_name + "-y.npy"))
    prefix_text = "This is a window of sensor readings captured in a home for the activity - "
    

    if dataset_name == "aruba":
        text_labels = convert_num_labels_to_text_labels(labels, arubaDict)
    elif dataset_name == "milan":
        text_labels = convert_num_labels_to_text_labels(labels, milanDict)
    elif dataset_name == "cairo":
        text_labels = convert_num_labels_to_text_labels(labels, cairoDict)
    else:
        print("Dataset Not Incorporated Here!")

    full_texts = [f"{prefix_text} {activity}" for activity in text_labels]
    
    model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
    
    embeddings = model_encode(model, full_texts)
    return embeddings

def model_encode(model, full_texts):
    sentence_embeddings = model.encode(full_texts)
    return sentence_embeddings


    
    
if __name__ == '__main__':

    datasets = ["aruba", "milan", "cairo"]

    for dataset_name in datasets:
        
        embeddings = generate_text_embeddings(dataset_name)
        print("saving for  ", dataset_name)
        np.save(os.path.join(npy_path, "{}-labels_embeddings_ta.npy".format(dataset_name)), embeddings )