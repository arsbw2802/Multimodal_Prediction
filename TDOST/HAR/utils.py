import numpy as np
import os
import sys
import socket


def get_npy_path():
    
    if "mt" in socket.gethostname():
        npy_path= "/mnt/attached1/TDOST/npy"
    return npy_path

def open_raw_data_file(path):

    data = np.load(path, allow_pickle=True)

    return data