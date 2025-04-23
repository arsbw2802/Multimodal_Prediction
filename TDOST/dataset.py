import os
from time import time

import joblib
import numpy as np
import torch
from .sliding_window import sliding_window
from torch.utils.data import Dataset, DataLoader
import inflect
import random
from sentence_transformers import SentenceTransformer

def open_raw_data_file(path):

    data = np.load(path, allow_pickle=True)

    return data

def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1], data_x.shape[2]), (ss, 1, 1))
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float16), data_y.reshape(len(data_y)).astype(np.uint8)

  
# Defining the data loader for the implementation
class HARDataset(Dataset):
    def __init__(self, directory, sentence_encoder_name, phase, window_size=50, step_size=25):
        raw_data = open_raw_data_file(os.path.join(directory, f"MARBLE_{phase}_embeddings_v1_{sentence_encoder_name}.npy")).astype(np.float16)
        raw_labels = open_raw_data_file(os.path.join(directory, f"MARBLE_{phase}_labels.npy")).astype(np.uint8)

        # Apply sliding window on embedded data and labels
        self.data, self.labels = opp_sliding_window(raw_data, raw_labels, window_size, step_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.from_numpy(self.data[index]).double()
        label = torch.tensor(self.labels[index]).double()
        return data, label



    def load_dataset(self, filename):
        since = time()
        data_raw = joblib.load(filename)

        time_elapsed = time() - since
        print('Data loading completed in {:.0f}m {:.0f}s'
              .format(time_elapsed // 60, time_elapsed % 60))

        return data_raw

