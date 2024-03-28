"""
    This code manages the dataset, extracting it or by adding new samples
"""

import os
import zipfile
from argparse import ArgumentParser
from dataset.create_samples import new_samples
import time


def parse_args():
    ap=ArgumentParser()

    ap.add_argument('-path',dest='base_path',default=os.path.join('.','dataset'), help='The path where the dataset archive is stored on.')

    ap.add_argument('-add_samples',dest='add_samples',action="store_true",help='To add new samples to the already existing ones')

    args = ap.parse_args()
    return args

def ask_for_positive_value(desc,error):
    try:
        value=int(input(desc))
    except:
        raise ValueError(error)
    if value < 1:
        raise ValueError(error)
    
    return value

if __name__=='__main__':
    args=parse_args()
    base_path=os.path.join(args.base_path)

    ## Extracting the dataset ##
    # Checking if dataset path exists, otherwise raise error since other files will be missing too
    if not os.path.exists(os.path.join(base_path,'dataset.zip')):
        raise RuntimeError('Missing dataset file, clone the repository again to correctly setup the dataset')
    
    print("Extracting dataset, it can take a few minutes...")
    
    start_time=time.process_time()
    with zipfile.ZipFile(os.path.join(base_path,'dataset.zip'), 'r') as zip_ref:     
        zip_ref.extractall(base_path)
    
    end_time=time.process_time()
    print("Dataset extracted successfully in {:.2f} seconds !".format(end_time-start_time))

    ## Creating new samples if required ##
    if args.add_samples:
        num_threads=ask_for_positive_value('Number of threads to use: ',
                                           'The number of threads must be a positive integer')
        num_train=ask_for_positive_value('Number of training samples to generate: ',
                                         'The number of samples must be a positive integer')
        num_test=ask_for_positive_value('Number of test samples to generate: ',
                                        'The number of samples must be a positive integer')
        new_samples(base_path,num_threads,num_train,num_test)
