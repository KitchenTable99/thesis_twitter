import torch
from typing import Tuple, Literal
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import wandb

Device = Literal['cuda', 'cpu']


def create_dataloaders(training_data, testing_data, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layer_size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(132),
            nn.ReLU(),
            nn.Linear(hidden_layer_size),
            nn.ReLU(),
            nn.Linear(2)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def train(dataloader, model, loss_fn, optimizer, device: Device):
    size = len(dataloader.dataset)
    model.train()
    correct, train_loss = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Bits for acc and loss
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    return test_loss, correct


def main():
    wandb.init(project="liebrary-of-congress-throwaway", entity="davisai")
    lr = wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs
    optimizer_type = wandb.config.optimizer
    hidden_layer_size = wandb.config.hidden_layer_size
    regularization = wandb.config.regularization

    device: Device = "cuda" if torch.cuda.is_available() else "cpu"

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_dataloader, test_dataloader = create_dataloaders(training_data, test_data, batch_size=bs)

    model = NeuralNetwork(hidden_layer_size).to(device)
    loss_fn = nn.CrossEntropyLoss()
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
