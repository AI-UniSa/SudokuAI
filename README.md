# Sudoku ai
In this repository, you can find the code to train and use a neural network to solve sudokus. The repo's structure is the following:
- dataset, the directory where the dataset is stored. It also contains an utility script to increase the dataset size;
- training, the directory containing everything related to the training phase;
- models, the directory containing all the supported models.

# Installation
To install this repository and play with it you have to run the following commands:

```
git clone https://github.com/AI-UniSa/SudokuAI.git
cd SudokuAI
pip install -r requirements.txt
```
Once you've downloaded everything you can begin the training, by launching:
```
python train.py
```
It will start with the default parameters. To customize its behavior run it with the ` -h ` flag to see all the available options.

# Actual project state
As of now, everything is working, but the network's performance is very poor, something in the neighborhood of an accuracy of 0.12 after 500 epochs. 
## Roadmap
The next updates to this repository will regard:
- A deep analysis and updates of the training code;
- The implementation of other networks;
- The setup of the inference code.
