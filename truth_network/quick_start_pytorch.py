import torch
from typing import Tuple, Literal
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import wandb

import pandas as pd

Device = Literal['cuda', 'cpu']


def create_dataloaders(training_data, testing_data, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


# allow dataset stuff
class TwitterDataset(Dataset):
    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        
        x = df.loc[:, ~df.columns.isin(['tweet__fake', 'tweet__possibly_sensitive'])].astype('float32').values
        # x = df.loc[:, ~df.columns.isin(['tweet__fake'])].astype('float32').values
        y = df['tweet__fake'].values
        
        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
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

    def forward(self, x):
        return self.linear_relu_stack(x)


def train(dataloader, model, loss_fn, optimizer, device: Device):
    size = len(dataloader.dataset)
    model.train()
    correct, train_loss = 0, 0
    for batch, (X, y) in enumerate(dataloader):
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
        pred[pred < 0] = 0
        pred[pred >= 0] = 1
        correct += (pred == y).type(torch.float).sum().item()

    wandb.log({'sum_sequential_0': torch.sum(model.linear_relu_stack[0].weight.data)})
    wandb.log({'sum_sequential_1': torch.sum(model.linear_relu_stack[1].weight.data)})
    wandb.log({'sum_sequential_2': torch.sum(model.linear_relu_stack[2].weight.data)})
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
            pred[pred < 0] = 0
            pred[pred >= 0] = 1
            correct += (pred == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    return test_loss, correct


def main():
    wandb.init(project="liebrary-of-congress-throwaway", entity="davisai")
    lr = .5
    bs = 32
    epochs = 10
    optimizer_type = 'adam'
    hidden_layer_size = 100
    regularization = .9
    # lr = wandb.config.lr
    # bs = int(wandb.config.batch_size)
    # epochs = int(wandb.config.epochs)
    # optimizer_type = str(wandb.config.optimizer)
    # hidden_layer_size = int(wandb.config.hidden_layer_size)
    # regularization = wandb.config.regularization

    device: Device = "cuda" if torch.cuda.is_available() else "cpu"

    training_data = TwitterDataset('/datasets/liebrary/small.csv')
    test_data = TwitterDataset('/datasets/liebrary/small.csv')

    train_dataloader, test_dataloader = create_dataloaders(training_data, test_data, batch_size=bs)

    model = NeuralNetwork(20, hidden_layer_size).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=regularization)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)

    for epoch in range(epochs):
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, device)
        val_loss, val_acc = test(test_dataloader, model, loss_fn, device)

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })


if __name__ == '__main__':
    main()

