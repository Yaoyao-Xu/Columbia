from ast import arg
from pyexpat import model
from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.optim as optim
import argparse
import time
from models import *
torch.manual_seed(42)
np.set_printoptions(suppress=True)


class DynamicDataset(Dataset):
    def __init__(self, dataset_dir):
        # X: (N, 9), Y: (N, 6)
        self.X = np.load(os.path.join(dataset_dir, 'X.npy')).T.astype(np.float32)
        self.Y = np.load(os.path.join(dataset_dir, 'Y.npy')).T.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]





def train(model, train_loader, test_loader, args):

    model.train()
    loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(500):
        total_loss = 0.0
        for index, data in enumerate(train_loader):
            input, target = data
            optim.zero_grad()
            outputs = model.forward(input)
            error = loss(outputs,target)
            error.backward()
            total_loss += error.item()
            optim.step()

        test_error = test(model, test_loader, loss)
        model_folder_name = f'epoch_{epoch:04d}_loss_{test_error:.8f}'
        print("TE={}, VE={}".format(total_loss / index, test_error))
        if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
            os.makedirs(os.path.join(args.save_dir, model_folder_name))
        torch.save(model.state_dict(), os.path.join(args.save_dir, model_folder_name, 'dynamics.pth'))
        print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "dynamics.pth")}\n')


 


def test(model, test_data, loss):
    model.eval()

    test_loss = 0
    for i, data in enumerate(test_data):
        input, target = data
        outputs = model.forward(input)
        result_loss = loss(outputs,target)
        test_loss += result_loss
        
    return test_loss / i


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--split', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    return args


def main():
    args = get_args()
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = DynamicDataset(args.dataset_dir)
    dataset_size = len(dataset)
    test_size = int(np.floor(args.split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 128)
    model= build_model(args.num_links, 0.01)
    train(model, train_loader, test_loader, args)
        

if __name__ == '__main__':
    main()
