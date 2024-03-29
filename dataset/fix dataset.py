import os
from tqdm import tqdm

""" Used to reorder the dataset to fix gaps between dataset's files"""

def rename_files(directory):
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Filter out only files (not directories)
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]
    
    print('{} : {}'.format(directory,len(files)))
    # Sort files based on their names
    files.sort()
    
    # Rename files sequentially
    for i, filename in tqdm(enumerate(files)):
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, f'{i}.txt')
        os.rename(old_path, new_path)
        #print(f"Renamed {filename} to {i}")

# Replace 'directory_path' with the path to your directory containing numbered files

rename_files(os.path.join("train",'data'))
rename_files(os.path.join("train",'gt'))
rename_files(os.path.join("test",'data'))
rename_files(os.path.join("test",'gt'))