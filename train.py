import os

import torch
from time import gmtime, strftime
import re
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision.transforms import v2

from training.train_hp import *
from training.model_zoo import ModelZoo
from training.dataset import SudokuDataset, SudokuStandardize

from torchinfo import summary


def mean(l):
    return sum(l)/len(l)


def parse_args():
    parser = ArgumentParser()
    # data
    # The dataset organization is the following:
    #   args.data:
    #       |---> train.txt 
    #       |
    #       L---> text.txt

    parser.add_argument('--data', dest='data',
                        help="Path to the dataset folder", default=os.path.join('.', 'dataset'))

    # training hyper-parameters
    parser.add_argument('--nw', dest='nw', help="number of workers for the dataloader",
                        default=WORKERS, type=int)
    parser.add_argument('--bs', dest='bs', help="batch size",
                        default=BATCH_SIZE, type=int)
    parser.add_argument('--lr', dest='lr', help="learning rate",
                        default=LEARNING_RATE, type=float)
    parser.add_argument('--device', dest='device', help="device to use",
                        default='cuda', type=str)
    parser.add_argument('--sched', dest='sched', action="store_true", help="To use or not use the learning rate scheduler",
                        default=True)
    parser.add_argument('--epochs', dest='epochs', help="Number of epochs to train",
                        default=EPOCHS, type=int)
    parser.add_argument('--model', dest='model', help="The model to train",
                        default='mlp')
    # checkpoint
    # The checkpoint name must comprise the string '_epoch_N' in order to start from epoch N
    parser.add_argument("--checkpoint", dest="checkpoint", help="path to checkpoint to restore",
                        default=None, type=str)
    args = parser.parse_args()
    return args


def one_epoch(model, criterion, optimizer, train_loader, val_loader, device):
    model.train()

    train_loss = []
    train_acc = []

    for X, y in tqdm(train_loader,desc='Training:'):
        X = X.to(device).float()
        y = y.to(device).float()

        optimizer.zero_grad()

        o = model(X)
        o_act = criterion.activate(o)


        loss = criterion.evaluate(o, y) # NOTE: in CNN is CrossEntropyLoss and requires not activeted output
        loss.backward()

        train_loss.append(loss.item())

        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = []
        val_acc = []
        val_sudoku_acc = []

        for X, y in tqdm(val_loader,desc='Validation:'):
            X = X.to(device)
            y = y.to(device).float()

            o = model(X)
            o_act = criterion.activate(o)

            val_loss.append(criterion.evaluate(o, y))# NOTE: in CNN is CrossEntropyLoss and requires not activeted output

            # TODO: fix this
            # It's not the best way to cast the output, but is surely the easyest
            #   Watch out that it keeps the integer part of the value, so as example
            #   bot 3.3 and 3.9 are casted to 3

            # val_acc.append(mean((o.int() == y)))
            # val_acc.append(mean((criterion.extract(o_act) == int((y + 0.5)*9))))
            val_acc.append(mean((criterion.extract(o_act) == y)))

            # print("Extracted: ", criterion.extract(o_act))
            # print("Ground truth: ", y)

            # Check if a complete sudoku is solved
            # sudoku_acc = torch.all(criterion.extract(o_act) == int((y + 0.5)*9), dim = 1) # B x 81 -> B
            sudoku_acc = torch.all(criterion.extract(o_act) == y, dim = 1) # B x 81 -> B
            val_sudoku_acc.append(mean(sudoku_acc))

    train_loss = mean(train_loss)
    val_loss = mean(val_loss)
    val_acc = mean(mean(val_acc))
    val_sudoku_acc = mean(val_sudoku_acc)

    # print("Sudoku accuracy is: {:.4f}".format(val_sudoku_acc))

    return train_loss, val_loss.item(), val_acc.item()


def plot_results(train_losses, val_losses, val_accuracies, experiment_name):
    # Plot loss during training
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'b')
    plt.plot(val_losses, 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss during training and validation')
    plt.legend(['Train', 'Validation'])

    plt.savefig('checkpoints/{}/loss.jpg'.format(experiment_name))

    # Plot each metric during training
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy during training')
    plt.savefig('checkpoints/{}/accuracy.jpg'.format(experiment_name))


def train(model, start_epoch, epochs, lr, train_loader, val_loader, criterion, device, experiment_name):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 10e-2)
    model.train()

    # training and validation
    train_losses = []
    val_losses = []
    val_acc = []

    prev_loss = float('inf')
    no_gain = 0
    val_epoch_accuracy=0

    for epoch in range(start_epoch, epochs):
        if epoch != 0:
            print('\n\n\n\n')
        print('##################      EPOCH {}      ##################\n\tLoss: {:.4f} \t Accuracy: {:.4f}\nLast improvement {} epochs ago'.format(epoch,prev_loss,val_epoch_accuracy,no_gain))
        train_epoch_loss,val_epoch_loss, val_epoch_accuracy = one_epoch(
            model, criterion, optimizer, train_loader, val_loader, device)

        # store the validation metrics
        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        val_acc.append(val_epoch_accuracy)

        # Print metrics
        #print('Current accuracy:{:.4f}'.format(val_epoch_accuracy))
        #print('Current loss:    {:.4f}'.format(val_epoch_loss))

        # Early stopping management
        if val_epoch_loss < prev_loss:
            prev_loss = val_epoch_loss
            no_gain = 0  #  Resetting early stopping counter

            save_path = os.path.join("checkpoints", experiment_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(
                save_path, "epoch_{}.pth".format(epoch)))
        else:
            no_gain += 1

        if no_gain >= EARLY_STOPPING_PATIENCE:
            print("Quitting training due to early stopping...")
            break
        scheduler.step()

    return train_losses, val_losses, val_acc


def main():
    args = parse_args()
    experiment_name = EXP_BASE_NAME + " " + strftime("%d %b %H %M", gmtime())

    print("Using main device: " + args.device)

    # Model initialization
    m = ModelZoo()
    model = m.get_model(args.model)
    model.train()
    model.to(args.device)
    summary(model,depth=5)

    # Dataset and dataloader initialization
    train_root = os.path.join(args.data, "train.txt")
    val_root = os.path.join(args.data, "val.txt")
    preprocess = v2.Compose([SudokuStandardize()])
    training_set = SudokuDataset(root=train_root, preprocess=preprocess)
    validation_set = SudokuDataset(root=val_root, preprocess=preprocess)

    train_loader = DataLoader(training_set,
                            batch_size=args.bs, shuffle=True, num_workers=args.nw)
    val_loader = DataLoader(validation_set,
                            batch_size=args.bs, shuffle=True, num_workers=args.nw) # TODO: check if is it the case to shuffle the validation set

    # Showing what we have loaded
    print("Training set:\t{} samples".format(len(training_set)))
    print("Validation set:\t{} samples".format(len(validation_set)))

    criterion = m.get_helper(args.model)
    criterion.to(args.device)
    start_epoch = 0

    if args.checkpoint is not None:
        print("Loading checkpoint {}...".format(args.checkpoint))
        model.load_state_dict(torch.load(args.checkpoint))
        # model.load_state_dict(torch.load(args.checkpoint))
        # Gatherng the starting epoch from the weights
        try:
            epoch_str = re.findall(r'_epoch_\d+', args.checkpoint)[0]
            start_epoch = int(re.findall(r'\d+', epoch_str)[0])
        except:
            print("Unable to find starting epoch...\n \
                Checkpoint file name must comprise the string '_epoch_N' in order to start from epoch N")

    train_losses, val_losses, val_accuracies = train(model, start_epoch, args.epochs, args.lr, train_loader,
                            val_loader, criterion, args.device, experiment_name)

    # Print both accuracy and loss during training
    plot_results(train_losses, val_losses, val_accuracies, experiment_name)


if __name__ == '__main__':
    main()
