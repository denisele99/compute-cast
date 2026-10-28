"""Small PyTorch MLP; also compatible with collect_workload.py's meta factory."""
from torch import nn


def build_model(config):
    return nn.Sequential(
        nn.Linear(config['input_size'], config['hidden_size']),
        nn.ReLU(),
        nn.Linear(config['hidden_size'], config['output_size']),
    )
