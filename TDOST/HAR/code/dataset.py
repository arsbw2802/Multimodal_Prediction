import os
from time import time

import joblib
import numpy as np
import torch
from sliding_window import sliding_window
from torch.utils.data import Dataset, DataLoader
import inflect
import random
from sentence_transformers import SentenceTransformer

def open_raw_data_file(path):

    data = np.load(path, allow_pickle=True)

    return data

def opp_sliding_window(data_x, data_y, ws, ss):

    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
 
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)
  
# Defining the data loader for the implementation
class HARDataset(Dataset):
    def __init__(self, args, phase):
        
        print(args.root_dir)
      
        print( args.root_dir + args.dataset + "-fold_" + str(args.fold) + "_" + phase + f"_{args.embedding_type}.npy")

      
        self.data = open_raw_data_file(args.root_dir + args.dataset + "-fold_" + str(args.fold) + "_" + phase + f"_{args.embedding_type}_{args.sentence_encoder_name}.npy").astype(np.float32)
        self.labels = open_raw_data_file(args.root_dir + args.dataset + "-fold_" + str(args.fold) + "_" + phase + "_labels_" + f"{args.embedding_type}_{args.sentence_encoder_name}" +  ".npy").astype(np.uint8)


    def load_dataset(self, filename):
        since = time()
        data_raw = joblib.load(filename)

        time_elapsed = time() - since
        print('Data loading completed in {:.0f}m {:.0f}s'
              .format(time_elapsed // 60, time_elapsed % 60))

        return data_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index, :, :]

        data = torch.from_numpy(data).double()

        label = torch.from_numpy(np.asarray(self.labels[index])).double()
        return data, label


def load_dataset(args,  classifier=True):

    datasets = {x: HARDataset(args=args, phase=x) for x in ['train', 'val', 'test']}

    def get_batch_size():
        if classifier:
            batch_size = args.classifier_batch_size
        else:
            batch_size = args.batch_size

        return batch_size

    data_loaders = {x: DataLoader(datasets[x],
                                  batch_size=get_batch_size(),
                                  shuffle=True if x == 'train' else False,
                                  num_workers=2, pin_memory=True, drop_last=True)
                    for x in ['train', 'val', 'test']}

    # Printing the batch sizes
    for phase in ['train', 'val', 'test']:
        print('The batch size for {} phase is: {}'
              .format(phase, data_loaders[phase].batch_size))

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

    return data_loaders, dataset_sizes
