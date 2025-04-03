import numpy as np
from torch.utils.data import Dataset
from sliding_window import *
from joblib import load
import copy

class MyDataset(Dataset):
    def __init__(
        self,
        filepath,
        phase,
        sliding_window_length=50,
        sliding_window_step_size=25,
    ):

        self.sliding_window_length = sliding_window_length
        self.sliding_window_step_size = sliding_window_step_size
        self.phase = phase
        unsegmented_X, unsegmented_y, fold = self.load_datafiles(filepath)
        self.fold = fold

        self.X = self.app_sliding_window(
            unsegmented_X, self.sliding_window_length, self.sliding_window_step_size
        )
        self.y = self.app_sliding_window(
            unsegmented_y, self.sliding_window_length, self.sliding_window_step_size
        )

        self.y = self.y[:, -1]

    def load_datafiles(self, filepath):

        # code for joblib file
        data = load(filepath)

        X = data[self.phase]["data"]
        y = data[self.phase]["labels"]
        fold = data["fold"]
        return X, y, fold

    def app_sliding_window(self, arr, window_size, step_size):
        if len(arr.shape) == 1:
            arr = arr.reshape(arr.shape[0], -1)
        return sliding_window(arr, (window_size, arr.shape[1]), (step_size, 1))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        data_x = self.X[i]
        data_y = self.y[i]

        return (data_x, data_y)




if __name__ == "__main__":
    pass
