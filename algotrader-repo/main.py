from datetime import datetime
import os
import re
import subprocess
import sys
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

import torch
from torch import nn


from src.data.dataloader import get_loaders, get_X_y, get_df
from src.models.linear_model import LinearRegression, MLP
from src.models.rnn import LSTMModel, GRUModel, RNN
from src.models.dimensionality_reduction import CustomDownprojection
from src.features.build_features import get_returns
from src.utils.ulits import get_summary_statistics
from src.visualization import visualize
import os
import glob


directory = r'C:\Users\stsch\Documents\AlgoBoost-Trading-Solutions\repo\algotrader-repo\src\data'
filename_pattern = 'make_dataset.py'
file_paths = glob.glob(os.path.join(directory, filename_pattern))

if file_paths:
    make_dataset_result = subprocess.run(
        [r'C:/Users/stsch/AppData/Local/Microsoft/WindowsApps/python3.9.exe', file_paths[0]],
        capture_output=True,
    )
    print(make_dataset_result.stdout.decode())
    print(make_dataset_result.stderr.decode())
else:
    print(f"File '{filename_pattern}' not found in directory '{directory}'.")

fn = f"merged_assets_{datetime.today().date()}.csv"
train_loader, val_loader = get_loaders(os.path.join(fn), seq_length=1)
(X_train, y_train), (X_val, y_val) = get_X_y(os.path.join(fn))  # ToDo: use later for XGB and RFs


# (batch_size, input_dim, seq_length, output_dim) = next(iter(train_loader)).shape
# X, y = next(iter(train_loader))
# batch_size, seq_length, input_dim = X.shape
# output_dim = y.shape[-1]

# lin_model = LinearRegression(input_dim=input_dim, output_dim=output_dim)
# mlp = MLP(input_dim=input_dim, output_dim=output_dim)

# train_losses, val_losses = lin_model.run_training(
#     train_loader, 
#     val_loader, 
#     criterion=nn.BCEWithLogitsLoss(), 
#     optimizer=torch.optim.Adam(lin_model.parameters()),
#     num_epochs=1,
# )
# performance_metrics = lin_model.run_validation(val_loader)


# train_losses, val_losses = mlp.run_training(
#     train_loader, 
#     val_loader, 
#     criterion=nn.BCEWithLogitsLoss(), 
#     optimizer=torch.optim.Adam(mlp.parameters()),
#     num_epochs=3,
# )
# performance_metrics = mlp.run_validation(val_loader)

df = get_df(fn).iloc[-7000:]
downprojection = CustomDownprojection(df, dim=3)
close_times = downprojection.get_closest_times()

return_cols = [col for col in close_times.columns if re.search(r'returns_\d+d$', col)]  
returns = close_times[return_cols]

fn = "models/downprojection/expected_returns_90d.xlsx"
stats_df = get_summary_statistics(returns, df, fn)

visualize.plot_boxplots_from_samples(returns, df, asset_symbol="AAPL", store=False)
visualize.plot_boxplots_from_samples(returns, df, asset_symbol="BTC", store=False)
visualize.plot_boxplots_from_samples(returns, df, asset_symbol="^SPX", store=False)
visualize.plot_boxplots_from_samples(returns, df, asset_symbol="^IXIC", store=False)
visualize.plot_boxplots_from_samples(returns, df, asset_symbol="GLD", store=False)
visualize.plot_boxplots_from_samples(returns, df, asset_symbol="TSLA", store=False)
visualize.plot_boxplots_from_samples(returns, df, asset_symbol="ETH", store=False)

