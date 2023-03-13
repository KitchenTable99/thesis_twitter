from quick_start_pytorch import NeuralNetwork
import pickle
import pandas as pd
import glob
from typing import Dict
import torch


def get_hidden_size_from_state(state_dict: Dict) -> int:
    for param_tensor in state_dict:
        num_out_neurons_in_layer = state_dict[param_tensor].size()[0]
        if num_out_neurons_in_layer != 1: # the final layer has 1 neuron always, so any other number is the hidden layer
            return num_out_neurons_in_layer

    raise ValueError('Hidden layer size not readable')


def get_model_from_path(path: str) -> NeuralNetwork:
    state_dict = torch.load(path)
    hidden_layer_size = get_hidden_size_from_state(state_dict)

    model = NeuralNetwork(156, hidden_layer_size)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def evaluate_model(model: NeuralNetwork, testset: pd.DataFrame, name: str) -> float:
    x_numpy = testset.loc[:, ~testset.columns.isin(['tweet__fake'])].astype('float32').values
    y_numpy = testset['tweet__fake'].to_numpy().reshape((-1, len(testset)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(x_numpy, dtype=torch.float32)
    y = torch.tensor(y_numpy, dtype=torch.float32)
    x.to(device)
    y.to(device)

    pred = model(x)
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1

    correct = torch.sum(torch.diagonal(torch.eq(pred, y)))
    acc = correct / len(testset)

    print(f'Model {name} performed with an accuracy of {acc}')

    return acc




def main():
    testset = pd.read_csv('final_nn_testset.csv')
    state_paths = glob.glob('**/*.pth', recursive=True)
    performances = {}
    for state_path in state_paths:
        name = state_path.split('/')[-1]
        model = get_model_from_path(state_path)
        acc = evaluate_model(model, testset, name)
        performances[name] = acc

    with open('performances.pickle', 'wb') as fp:
        pickle.dump(performances, fp)


if __name__ == "__main__":
    main()
