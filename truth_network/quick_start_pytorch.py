import torch
from copy import deepcopy
import numpy as np
from typing import Tuple, Literal
from torch import nn
from torch.utils.data import DataLoader, Dataset

import wandb

import pandas as pd

from early_stopper import EarlyStopper

Device = Literal['cuda', 'cpu']


def create_dataloaders(training_data, testing_data, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


# allow dataset stuff
class TwitterDataset(Dataset):
    def __init__(self, file_name):
        df = pd.read_csv(file_name)

        # x = df.loc[:, ~df.columns.isin(['tweet__fake', 'tweet__possibly_sensitive'])].astype('float32').values
        x = df.loc[:, ~df.columns.isin(['tweet__fake'])].astype('float32').values
        y = df['tweet__fake'].values
        # print(f'unique values: {np.unique(y)}')
        # print(f'{x[:10] = }')

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x)
        return x


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, num_input_columns, hidden_layer_size):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_input_columns, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.linear_relu_stack(x)


def train(dataloader, model, loss_fn, optimizer, device: Device):
    size = len(dataloader.dataset)
    model.train()
    correct, train_loss = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # wandb.log({'shape X[0]': X.shape[0],
        #            'shape X[1]': X.shape[1],
        #            'shape y[0]': y.shape[0]})
        X, y = X.to(device), y.reshape((-1, 1)).to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Bits for acc and loss
        train_loss += loss.item()
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        # wandb.log({'sum of pred': torch.sum(pred)})
        correct += (pred == y).type(torch.float).sum().item()

    # wandb.log({'sum_sequential_0': torch.sum(model.linear_relu_stack[0].weight.data),
               # 'sum_sequential_2': torch.sum(model.linear_relu_stack[2].weight.data)})
    avg_loss = train_loss / len(dataloader)
    acc = correct / size

    return avg_loss, acc


def test(dataloader, model: NeuralNetwork, loss_fn, device: Device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.reshape((-1, 1)).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            correct += (pred == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    return test_loss, correct


def main():
    wandb.init(project="liebrary-of-congress-throwaway",
               entity="davisai")
    # bs = 42
    # hidden_layer_size = 219
    # lr = 0.0007447028032999965
    # optimizer_type = 'adam'
    # patience = 6
    # regularization = 0.000002793180757767303
    # manual_epoch = 11
    lr = wandb.config.lr
    bs = int(wandb.config.batch_size)
    patience = wandb.config.patience
    optimizer_type = str(wandb.config.optimizer)
    hidden_layer_size = int(wandb.config.hidden_layer_size)
    regularization = wandb.config.regularization

    device: Device = "cuda" if torch.cuda.is_available() else "cpu"

    training_data = TwitterDataset('~/datasets/do-not-delete--liebrary-of-congress/train_normalized.csv')
    test_data = TwitterDataset('~/datasets/do-not-delete--liebrary-of-congress/test_normalized.csv')

    train_dataloader, test_dataloader = create_dataloaders(training_data, test_data, batch_size=bs)

    model = NeuralNetwork(156, hidden_layer_size).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=regularization)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)

    early_stop_checker = EarlyStopper(patience=patience)

    best_val_loss = np.inf
    best_model_state = {}
    for epoch in range(200):
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, device)
        val_loss, val_acc = test(test_dataloader, model, loss_fn, device)
        if val_loss < best_val_loss:
            best_model_state = deepcopy(model.state_dict())
            best_val_loss = val_loss

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        # if epoch == manual_epoch:
        #     torch.save(model.state_dict(), f'out/{name}_manual.pth')
        if early_stop_checker.stop_early(val_loss):
            break
    torch.save(best_model_state, f'out/regularization_{regularization}_opt_{optimizer_type}_best.pth')


if __name__ == '__main__':
    main()
