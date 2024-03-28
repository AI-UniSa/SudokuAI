import os
import random
import sudokum
from tqdm import tqdm
from threading import Thread

def get_sudoku()->list:
    difficulty=random.uniform(0.2,0.8)
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

def create(path,num_threads,num_samples):
    last_value=get_last_value(path)
    threads=[0]*num_threads
    n=int(num_samples/num_threads)

    # Assigning n samples to all the threads except the last one
    for i in range(num_threads-1):
        threads[i]=Thread(target=save_data,args=[last_value+n*i,n,path,i])
        threads[i].start()

    # The last thread will have to generate all the samples missing due to integer approximation
    remaining_samples = num_samples - (num_threads-1)*n
    threads[-1]=Thread(target=save_data,args=[last_value+n*i,remaining_samples,path,num_threads-1])
    threads[-1].start()
    
    # Waiting for all the threads to end
    for i in range(num_threads):
        threads[i].join()
    
    # Updating last value file
    set_last_value(path,last_value+num_samples)


def new_samples(base_path, num_threads,num_train,num_test):   
    train_dir=os.path.join(base_path,'train')
    test_dir=os.path.join(base_path,'test')

    # Train set
    print("TRAIN SET CREATION")
    create(train_dir,num_threads,num_train)

    # Test set
    print("TEST SET CREATION")
    create(test_dir,num_threads,num_test)


