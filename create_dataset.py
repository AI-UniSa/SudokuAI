import requests
from urllib.parse import urlencode
from tqdm import tqdm
from threading import Thread

def get_sudoku(difficulty:str='random')->list:
    choiches=['random','easy','medium','hard']
    if difficulty not in choiches:
        raise ValueError(f"Invalid argument '{difficulty}'. Allowed values are: {', '.join(choiches)}")

    url=f'https://sugoku.onrender.com/board?difficulty={difficulty}'
    message=requests.get(url).json()
    return message['board']

def solve_sudoku(sudoku):
    url='https://sugoku.onrender.com/solve'
    header= {'Content-type': 'application/x-www-form-urlencoded'}
    payload = urlencode({'board':sudoku})
    message = requests.post(url, data=payload, headers=header).json()
    return message['solution']

def save_data(start,num,path):
    for i in tqdm(range(num),desc='Thread {:d}'.format(start/num)):
        sudoku=get_sudoku()
        with open(f'{path}\\data\\{start+i}.txt','w') as f:
            f.write(str(sudoku))
        
        solution=solve_sudoku(sudoku)
        with open(f'{path}\\gt\\{start+i}.txt','w') as f:
            f.write(str(solution))

if __name__=='__main__':
    NUM={'train':int(1e4),'test':int(1e2)}
    NUM_THREADS=16
    
    train_dir="dataset\\train"
    test_dir="dataset\\test"

    # Train set
    print("TRAIN SET CREATION")
    threads=[0]*NUM_THREADS
    n=int(NUM['train']/NUM_THREADS)
    for i in range(NUM_THREADS):
        threads[i]=Thread(target=save_data,args=[n*i,n,train_dir])
        threads[i].start()
    
    for i in range(NUM_THREADS):
        threads[i].join()

    # Test set
    # print("TEST SET CREATION")
    # threads=[0]*NUM_THREADS
    # n=int(NUM['test']/NUM_THREADS)
    # for i in range(NUM_THREADS):
    #     threads[i]=Thread(target=save_data,args=[n*i,n,test_dir])
    #     threads[i].start()
    
    # for i in range(NUM_THREADS):
    #     threads[i].join()


