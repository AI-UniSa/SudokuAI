import os
import random
import sudokum
from tqdm import tqdm
from threading import Thread
import pandas as pd
import itertools

def get_sudoku()->list:
    difficulty=random.uniform(0.2,0.8)
    board = sudokum.generate(mask_rate=difficulty)
    return board

def add_data(start,num,df,idx):
    for i in tqdm(range(num),desc='Thread {:3d}'.format(idx)):
        # Generating new sudokus until one of them is solvable
        solved=False
        while not solved:
            sudoku=get_sudoku()
            solved, solution= sudokum.solve(sudoku)
        
        df.loc[start+i,'gt']=list(itertools.chain.from_iterable(solution))
        df.loc[start+i,'data']=list(itertools.chain.from_iterable(sudoku))

def add(df,num_threads,num_samples):
    last_value=len(df)
    threads=[0]*num_threads
    n=int(num_samples/num_threads)

    # Assigning n samples to all the threads except the last one
    for i in range(num_threads-1):
        threads[i]=Thread(target=add_data,args=[last_value+n*i,n,df,i])
        threads[i].start()

    # The last thread will have to generate all the samples missing due to integer approximation
    remaining_samples = num_samples - (num_threads-1)*n
    threads[-1]=Thread(target=add_data,args=[last_value+n*i,remaining_samples,df,num_threads-1])
    threads[-1].start()
    
    # Waiting for all the threads to end
    for i in range(num_threads):
        threads[i].join()


def new_samples(base_path, num_threads,num_train,num_test):   
    train_path=os.path.join(base_path,'train.txt')
    test_path=os.path.join(base_path,'test.txt')

    # Train set
    print("TRAIN SET CREATION")
    train_df=pd.read_csv(train_path)
    add(train_df,num_threads,num_train)
    train_df.to_csv(train_path,header=True,index=False)

    # Test set
    print("TEST SET CREATION")
    test_df=pd.read_csv(test_path)
    add(test_df,num_threads,num_test)
    test_df.to_csv(test_path,header=True,index=False)


def ask_for_positive_value(desc,error):
    try:
        value=int(input(desc))
    except:
        raise ValueError(error)
    if value < 0:
        raise ValueError(error)
    
    return value

if __name__=='__main__':
    base_path=os.path.join('.')
    num_threads=ask_for_positive_value('Number of threads to use: ',
                                           'The number of threads must be a positive integer')
    num_train=ask_for_positive_value('Number of training samples to generate: ',
                                         'The number of samples must be a positive integer')
    num_test=ask_for_positive_value('Number of test samples to generate: ',
                                        'The number of samples must be a positive integer')
    new_samples(base_path,num_threads,num_train,num_test)