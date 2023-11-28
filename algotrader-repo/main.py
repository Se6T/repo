from datetime import datetime
import os
import subprocess
import sys

import pandas as pd
import numpy as np

import torch
from torch import nn


from src.data.dataloader import get_loaders, get_X_y
from src.models.linear_model import LinearRegression, MLP
from src.models.rnn import LSTMModel, GRUModel, RNN


make_dataset_result = subprocess.run(
    [r'C:/Users/stsch/AppData/Local/Microsoft/WindowsApps/python3.9.exe', 
     r'src\\data\\make_dataset.py', '--help'], 
    capture_output=True,
)
print(make_dataset_result.stdout.decode())
print(make_dataset_result.stderr.decode())


fn = f"merged_assets_{datetime.today().date()}.csv"
train_loader, val_loader = get_loaders(os.path.join(fn))
train_set, val_set = get_X_y(os.path.join(fn))


batch_size, input_dim, seq_length, output_dim = next(iter(train_loader)).shape

lin_model = LinearRegression(input_dim=input_dim, output_dim=output_dim)
mlp = MLP(input_dim=input_dim, output_dim=output_dim)

train_losses, val_losses = lin_model.run_training(
    train_loader, 
    val_loader, 
    criterion=nn.BCEWithLogitsLoss(), 
    optimizer=torch.optim.Adam(),
    num_epochs=3,
)
performance_metrics = lin_model.run_validation(val_loader)


train_losses, val_losses = mlp.run_training(
    train_loader, 
    val_loader, 
    criterion=nn.BCEWithLogitsLoss(), 
    optimizer=torch.optim.Adam(),
    num_epochs=3,
)
performance_metrics = mlp.run_validation(val_loader)

