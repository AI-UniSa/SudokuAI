import requests
from urllib.parse import urlencode
from tqdm import tqdm
from threading import Thread
import os
import sudokum
import random

NUM={'train':int(2e4),'test':int(2e2)}
NUM_THREADS=8

def get_sudoku()->list:
    difficulty=random.uniform(0.3,0.8)
    board = sudokum.generate(mask_rate=difficulty)
    return board

def get_last_value(path):
    with open(os.path.join(path,'count.txt'),'r') as f:
        return int(f.read())

def set_last_value(path,num):
    with open(os.path.join(path,'count.txt'),'w') as f:
        f.write(str(num))
        

def save_data(start,num,path,idx):
    for i in tqdm(range(num),desc='Thread {:3d}'.format(idx)):
        # Generating new sudokus until one of them is solvable
        solved=False
        while not solved:
            sudoku=get_sudoku()
            solved, solution= sudokum.solve(sudoku)
        
        with open(f'{path}\\data\\{start+i}.txt','w') as f:
            f.write(str(sudoku))
          
        with open(f'{path}\\gt\\{start+i}.txt','w') as f:
            f.write(str(solution))

if __name__=='__main__':    
    train_dir="dataset\\train"
    test_dir="dataset\\test"

    # Train set
    print("TRAIN SET CREATION")
    last_value=get_last_value(train_dir)
    threads=[0]*NUM_THREADS
    n=int(NUM['train']/NUM_THREADS)
    for i in range(NUM_THREADS):
        threads[i]=Thread(target=save_data,args=[last_value+n*i,n,train_dir,i])
        threads[i].start()
    
    # waiting for all the threads to end
    for i in range(NUM_THREADS):
        threads[i].join()
    
    # updating last value file
    set_last_value(train_dir,last_value+NUM['train'])

    # Test set
    print("TEST SET CREATION")
    last_value=get_last_value(train_dir)
    threads=[0]*NUM_THREADS
    n=int(NUM['test']/NUM_THREADS)
    for i in range(NUM_THREADS):
        threads[i]=Thread(target=save_data,args=[last_value+n*i,n,test_dir,i])
        threads[i].start()
    
    # waiting for all the threads to end
    for i in range(NUM_THREADS):
        threads[i].join()
    
    # updating last value file
    set_last_value(test_dir,last_value+NUM['test'])


