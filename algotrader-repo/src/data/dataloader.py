"""
Load dataset, add features and return either numpy arrays for sklearn models or 
DataLoaders for pytorch models.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List

from ..features.build_features import process_df, get_returns


def get_dataset(filename: str, filepath: str = r"data\interim") -> pd.DataFrame:
    data = pd.read_csv(os.path.join(filepath, filename))
    data.set_index('Date', inplace=True)

    return data


class TradingDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            seq_length: int = 3,
            action_prob: float = 0.5,
            target_columns: Optional[List[str]] = None,
            classification: bool = True,
            binary: bool = True,
    ):

        self.df = df
        self.seq_length = seq_length
        self.action_prob = action_prob
        self.classification = classification
        self.binary = binary

        if target_columns is None:
            self.target_columns = [col for col in df.columns if 'Open' in col or 'Close' in col]
        else:
            self.target_columns = target_columns

        self.symbols = list(set([col.split('_')[0] for col in self.target_columns]))

        self.length = len(self.df) - seq_length

        self.y_labels = pd.DataFrame(columns=[f"{symbol}_label" for symbol in self.symbols])

        for symbol in self.symbols:
            symbol_open = f"{symbol}_Open"
            symbol_close = f"{symbol}_Close"
            price_diffs = df[symbol_close].shift(-1) - df[symbol_open].shift(-1)
            upper_lim = np.nanpercentile(price_diffs, (1 - self.action_prob / 2) * 100)
            lower_lim = np.nanpercentile(price_diffs, self.action_prob / 2 * 100)
            labels = np.where(price_diffs > upper_lim, -1, np.where(price_diffs < lower_lim, 1, 0))

            self.y_labels[f"{symbol}_label"] = labels

    def __getitem__(self, idx):
        X = torch.tensor(self.df.iloc[idx:idx + self.seq_length].values.astype(np.float32))

        if self.classification:
            y = torch.tensor(self.y_labels.iloc[idx:idx + self.seq_length].values.astype(np.float32))
            if self.binary:
                y = torch.where(y == -1, torch.tensor(0), y)
        else:
            y = torch.tensor(self.df.iloc[idx:idx + self.seq_length].values.astype(np.float32))

        return X, y

    def __len__(self):
        return self.length
   


def get_loaders(
        filename: str, 
        batch_size: int = 32, 
        seq_length: int = 3,
        val_split: float = 0.2, 
        noise: float = 1e-4,
        binary: bool = True,
    ) -> tuple[DataLoader, DataLoader]:
    data = get_dataset(filename)  # "merged_assets_2023-11-23.csv"
    df = process_df(data)  # yoy % change of prices, bin vol

    val_idx = int(len(df) * (1 * val_split))
    train_df, val_df = df.iloc[:val_idx], df.iloc[val_idx:]

    train_dataset = TradingDataset(train_df, seq_length, noise, binary=binary)
    val_dataset = TradingDataset(val_df, seq_length, noise, binary=binary)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    return train_loader, val_loader


def get_X_y(
        filename: str,
        seq_length: int = 1,
        val_split: float = 0.2,
        noise: float = 1e-5,
    ) -> tuple[tuple[np.array, np.array]]:
    data = get_dataset(filename)  # "merged_assets_2023-11-23.csv"
    df = process_df(data)  # yoy % change of prices, bin vol

    val_idx = int(len(df) * (1 - val_split))
    train_df, val_df = df.iloc[:val_idx], df.iloc[val_idx:]

    train_dataset = TradingDataset(train_df, seq_length, noise)
    val_dataset = TradingDataset(val_df, seq_length, noise)

    # Training data
    X_train = []
    y_train = []
    for idx in range(len(train_dataset)):
        X, y = train_dataset[idx]
        X_train.append(X.numpy())
        y_train.append(y.numpy())

    # Validation data
    X_val = []
    y_val = []
    for idx in range(len(val_dataset)):
        X, y = val_dataset[idx]
        X_val.append(X.numpy())
        y_val.append(y.numpy())

    return (np.array(X_train), np.array(y_train)), (np.array(X_val), np.array(y_val))



def get_df(filename: str) -> pd.DataFrame:
    data = get_dataset(filename)
    # df = process_df(data)
    df = get_returns(data, diff=90)
    processed_df = process_df(df)

    return processed_df.bfill()

