import os
from torch.utils.data import Dataset

import torch as t
import pandas as pd
from ast import literal_eval as listify


class SudokuDataset(Dataset):
    def __init__(self, root: os.PathLike, preprocess=None):
        super(SudokuDataset, self).__init__()
        self.data = pd.read_csv(root)
        self.data.dropna(inplace=True)
        self.preprocess = preprocess

    def __getitem__(self, index):
        row=self.data.loc[index]
        return self.preprocess(t.Tensor(listify(row['data'])), t.Tensor(listify(row['gt'])))


    def __len__(self):
        return len(self.data)


class SudokuStandardize(t.nn.Module):
    def forward(self, img, label):
        return img/9 - 0.5, label/9 - 0.5

if __name__ == '__main__':
    d = SudokuDataset(os.path.join(
        'C:\\Users\\paolo\\Desktop\\Universita\\Progetti Python\\SudokuAI\\dataset\\train.txt'))

    
    print('len:',len(d))
    print(d[100477])
    a=d[12]
    print(a[0][0],type(a[0][0]))
