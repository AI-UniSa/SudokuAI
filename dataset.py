import os
import itertools
from torch.utils.data import Dataset
from tqdm import tqdm
import json

import torch as t

class SudokuDataset(Dataset):
    def __init__(self, root:os.PathLike):
        super(SudokuDataset, self).__init__()
        self.data={}
        self.gt={}

        # TODO: maybe parallelize this for? As done in create dataset
        for file in tqdm(os.listdir(os.path.join(root,'data')),desc='Loading dataset'):
            if not os.path.isfile(os.path.join(root,'data',file)):
               continue
            
            i=int(file.split('.')[0])

            with open(os.path.join(root,'data',file)) as f:
                file_content=json.loads(f.read())
                self.data[i]=t.Tensor(list(itertools.chain.from_iterable(file_content)))
            
            with open(os.path.join(root,'gt',file)) as f:
                file_content=json.loads(f.read())
                self.gt[i]=t.Tensor(list(itertools.chain.from_iterable(file_content)))
        
    
    def __getitem__(self, index):
        return self.data[index],self.gt[index]
    
    def __len__(self):
        return len(self.gt)
    
if __name__=='__main__':
    d=SudokuDataset(os.path.join('C:\\Users\\paolo\\Desktop\\Universita\\Progetti Python\\SudokuAI\\dataset\\train'))

    print(len(d))
    print(d[12])
