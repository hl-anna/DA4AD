import sys
sys.path.append('./ast')
import os
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from tqdm import tqdm


class Dataset_AST(Dataset):

    def __init__(self, aug_type):
        self.data_dir = "./data_" + aug_type
        self.data_list = os.listdir(self.data_dir)
        data_list_int = np.array([int(d[:-4]) for d in self.data_list])
        self.data_list = np.array(self.data_list)
        self.data_list = self.data_list[np.argsort(data_list_int)]
        if aug_type=="test":
            self.label_dir = "./test_segment_labels"
        else:
            self.label_dir = "./train_labels_" + aug_type
        self.labels = pd.read_csv(self.label_dir, header=None)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        data = pd.read_csv(os.path.join(self.data_dir, self.data_list[item]), header=None).to_numpy()
        return data


def log_spec_dB_dist(x, y):
    log_spec_dB_const = 10.0 * math.sqrt(2.0) / math.log(10.0)
    diff = x - y
    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

augmentation_type = 'noaug' #["masking", "shift", "loudness", "noise", "pitch", "vtlp", "shift", "speed", "normalisation", "specaugment"]
aug_list = ["masking", "shift", "loudness", "noise", "pitch", "vtlp", "speed", "normalisation", "specaugment", "random"]
r_dataset = Dataset_AST('noaug')
mcd_list = []
for augmentation_type in aug_list:
    min_cost_tot = 0.0
    s_dataset = Dataset_AST(augmentation_type)
    mcd_aug = [augmentation_type]
    for i in tqdm(range(2340)):
        mcd_array = []
        r_data = r_dataset[i].transpose()
        s_data = s_dataset[i].transpose()
        mcd_array = np.array([log_spec_dB_dist(r_data[j], s_data[j]) for j in range(len(r_data))])
        mcd_array_mean = np.mean(mcd_array)
        min_cost_tot += mcd_array_mean

    mean_mcd = min_cost_tot / 3086
    mcd_aug.append(mean_mcd)
    mcd_list.append(mcd_aug)

df = pd.DataFrame(mcd_list)
df.columns = ['method_name', 'mcd']
df = df.round(3)
df.to_csv('./mcd_results2.csv')